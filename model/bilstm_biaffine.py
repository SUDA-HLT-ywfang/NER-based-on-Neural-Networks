import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

from .biaffine import Biaffine
from .embedding import Embedding_Layer
from .encoder import Encoder


# 模型模块
class BiLSTM_Biaffine(nn.Module):
    def __init__(self, data):
        super(BiLSTM_Biaffine, self).__init__()
        # embedding part
        self.word_embedding = Embedding_Layer(
            vocab=data.vocab_word, emb_dim=data.get("emb_dim_word"), pretrain_emb=data.get("emb_path_word"))
        self.use_char = data.get("use_char")
        self.input_size = data.get("emb_dim_word")
        self.gpu = data.get("gpu")
        if self.use_char:
            self.char_encoder = Encoder(
                feature_extractor=data.get("char_model"),
                embedding_dim=data.get("emb_dim_char"),
                hidden_dim=data.get("hidden_dim_char")
            )
            self.char_embedding = Embedding_Layer(
                vocab=data.vocab_char, emb_dim=data.get("emb_dim_char"), pretrain_emb=None)
            self.charemb_dropout = nn.Dropout(data.get("dropout_embedding"))
            self.input_size += data.get("hidden_dim_char")
        # self.feature_num = len(data.feature_config)
        # self.feature_embeddings = nn.ModuleList()
        # for idx in range(self.feature_num):
        #     self.input_size += data.feature_config[idx]['emb_dim']
        #     self.feature_embeddings.append(
        #         Embedding_Layer(vocab=data.feature_config[idx]['vocab'], emb_dim=data.feature_config[idx]['emb_dim']))
        # encoder
        self.word_bilstm = Encoder(
            feature_extractor='BiLSTM',
            embedding_dim=self.input_size,
            hidden_dim=data.get("hidden_dim_lstm"),
            num_layers=data.get("lstm_layers")
        )
        self.ffnn_start = nn.Linear(data.get("hidden_dim_lstm"), data.get("hidden_dim_ffnn"))
        self.ffnn_end = nn.Linear(data.get("hidden_dim_lstm"), data.get("hidden_dim_ffnn"))
        # biaffine decoder
        self.biaffine = Biaffine(n_in=data.get("hidden_dim_ffnn"), n_out=data.vocab_label.get_size(), bias_x=True, bias_y=True)
        # dropout
        self.dropout_embedding = nn.Dropout(data.get("dropout_embedding"))
        self.dropout_lstm = nn.Dropout(data.get("dropout_lstm"))
        self.dropout_ffnn = nn.Dropout(data.get("dropout_ffnn"))
        # loss calculator
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
        self.non_entity_idx = data.vocab_label.get_id('O')

    # 根据输入，得到模型输出
    def forward(self, batch):
        word_idxs, char_idxs, feature_idxs, labels = batch
        mask = word_idxs.gt(0)
        sent_lengths = mask.sum(1)
        words_present = self.word_embedding(word_idxs)
        # some pre-process
        sorted_sentlength, indices = torch.sort(sent_lengths, dim=0, descending=True)
        sorted_words_present = words_present[indices]
        sorted_mask = mask[indices]
        sorted_labels = labels[indices]
        # # add feature info
        # sorted_feature_idxs = feature_idxs[indices]
        # if self.feature_num != 0:
        #     feature_list = [sorted_words_present]
        #     for i in range(self.feature_num):
        #         feature_idx = sorted_feature_idxs[:, :, i]
        #         feature_rep = self.feature_embeddings[i](feature_idx)
        #         feature_list.append(feature_rep)
        #     sorted_words_present = torch.cat(feature_list, dim=2)
        # add char info
        if self.use_char:
            batchsize = word_idxs.size(0)
            max_sent_length = word_idxs.size(1)
            char_all_instances = char_idxs.view(batchsize * max_sent_length, -1)
            mask = char_all_instances.gt(0)
            char_lengths = mask.sum(1)
            sorted_char_lengths, char_indices = torch.sort(char_lengths, dim=0, descending=True)
            sorted_char_idxs = char_all_instances[char_indices]
            mask_filter_null = sorted_char_lengths.gt(0)
            sorted_char_idxs4train = sorted_char_idxs[mask_filter_null]
            filtered_size = batchsize * max_sent_length - sorted_char_idxs4train.size(0)
            sorted_char_lengths4train = sorted_char_lengths[mask_filter_null]
            sorted_char_input = self.char_embedding(sorted_char_idxs4train)
            sorted_char_input = self.charemb_dropout(sorted_char_input)
            char_feature = self.char_encoder.get_last_hidden(sorted_char_input, sorted_char_lengths4train)
            # recover
            char_feature_pad = torch.zeros(filtered_size, char_feature.size(1))
            if self.gpu != "-1":
                char_feature_pad = char_feature_pad.cuda()
            char_feature_final = torch.cat((char_feature, char_feature_pad), 0)
            _, char_recover = torch.sort(char_indices, dim=0, descending=False)
            char_feature_recover = char_feature_final[char_recover]
            char_feature_final = char_feature_recover.view(batchsize, max_sent_length, -1)
            char_feature_final_align_with_word = char_feature_final[indices]
            sorted_words_present = torch.cat((sorted_words_present, char_feature_final_align_with_word), dim=2)
        # model forward
        word_embeddings_drop = self.dropout_embedding(sorted_words_present)
        encoder_out = self.word_bilstm(word_embeddings_drop, sorted_sentlength)
        lstm_out_drop = self.dropout_lstm(encoder_out)
        
        span_start_rep = self.ffnn_start(lstm_out_drop)
        span_end_rep = self.ffnn_end(lstm_out_drop)
        span_start_rep = self.dropout_ffnn(span_start_rep)
        span_end_rep = self.dropout_ffnn(span_end_rep)
        # [batch_size, sent_len, sent_len, label_size]
        score = self.biaffine(x=span_start_rep, y=span_end_rep).permute(0, 2, 3, 1)

        if self.training:        
            score_for_loss = score.reshape(-1, score.shape[-1])
            gold_score = sorted_labels.reshape(-1)
            loss = self.criterion(input=score_for_loss, target=gold_score)

            return loss
        else:
            # pred_tag_matrix = torch.argmax(score, -1)
            # recover
            _, recover = torch.sort(indices, dim=0, descending=False)
            
            return self.decode(score[recover], labels.gt(0))

    def decode(self, score_matrix, mask):
        pred_score, pred_ans = score_matrix.max(-1)
        batch_size = mask.shape[0]
        result = torch.full_like(pred_ans, self.non_entity_idx)
        for i in range(batch_size):
            pred_score_sent, pred_ans_sent, mask_sent = pred_score[i], pred_ans[i], mask[i]
            # non_entity_idx_tensor->(Tensor: first dimension index, Tensor: second dimension index)
            non_entity_idx_tensor = torch.where(condition=(pred_ans_sent.ne(self.non_entity_idx) & mask_sent))
            # non_entity_idx_list->(List: first dimension index, List: second dimension index)
            non_entity_idx_list = [idx_tensor.tolist() for idx_tensor in non_entity_idx_tensor]
            # pred_score_non_entity -> Tensor: value indexed by non_entity_idx_list
            pred_score_non_entity = pred_score_sent[non_entity_idx_list]
            pred_idx_score = list(zip(*non_entity_idx_list, pred_score_non_entity))
            pred_idx_score_sorted = sorted(pred_idx_score, key=lambda e:e[2], reverse=True)

            # final set
            entity_set = set()
            for idx_i, idx_j, _ in pred_idx_score_sorted:
                exist_conflict = False
                for entity_info in entity_set:
                    if (idx_i <= entity_info[0] and idx_j >= entity_info[0]) or (entity_info[0] <= idx_i <= entity_info[1]):
                        exist_conflict = True
                        break
                if not exist_conflict:
                    entity_set.add((idx_i, idx_j, pred_ans_sent[idx_i, idx_j]))
            # fill result chart
            for idx_i, idx_j, ans in entity_set:
                result[i, idx_i, idx_j] = ans
        # [batch_size, sent_len, sent_len]
        return result
import torch
import torch.nn as nn
from utils.utils import *
import torch.nn.functional as F
from .mfvi import MFVI
from .encoder import Encoder
from .embedding import Embedding_Layer


# 模型模块
class BiLSTM_MFVI(nn.Module):
    def __init__(self, data):
        super(BiLSTM_MFVI, self).__init__()
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
        self.feature_num = len(data.feature_config)
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.input_size += data.feature_config[idx]['emb_dim']
            self.feature_embeddings.append(
                Embedding_Layer(vocab=data.feature_config[idx]['vocab'], emb_dim=data.feature_config[idx]['emb_dim']))
        self.word_bilstm = Encoder(
            feature_extractor='BiLSTM',
            embedding_dim=self.input_size,
            hidden_dim=data.get("hidden_dim_lstm"),
            num_layers=data.get("lstm_layers")
        )
        self.word_presentation_dropout = nn.Dropout(data.get("dropout_embedding"))
        self.lstmout_dropout = nn.Dropout(data.get("dropout_lstm"))
        self.hidden2tag = nn.Linear(data.get("hidden_dim_lstm"), data.vocab_label.get_size())
        self.mfvi = MFVI(label_size=data.vocab_label.get_size(),
                         gpu=self.gpu, 
                         window_size=data.get("mfvi_window_size"), 
                         iterations=data.get("mfvi_iterations"), 
                         add_start_end=data.get("mfvi_add_start_end"))
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    # 根据输入，得到模型输出
    def forward(self, batch):
        word_idxs, char_idxs, feature_idxs, labels = batch
        mask = labels.gt(0)
        sent_lengths = mask.sum(1)
        words_present = self.word_embedding(word_idxs)
        """
        :param words: batch_size*sent_len*emb_dim
        :param sent_lengths: batch_size*1
        :param labels: batch_size*sent_len
        :return: batch_size*sent_len
        """
        # some pre-process
        sorted_sentlength, indices = torch.sort(sent_lengths, dim=0, descending=True)
        sorted_words_present = words_present[indices]
        sorted_mask = mask[indices]
        sorted_labels = labels[indices]
        # add feature info
        sorted_feature_idxs = feature_idxs[indices]
        if self.feature_num != 0:
            feature_list = [sorted_words_present]
            for i in range(self.feature_num):
                feature_idx = sorted_feature_idxs[:, :, i]
                feature_rep = self.feature_embeddings[i](feature_idx)
                feature_list.append(feature_rep)
            sorted_words_present = torch.cat(feature_list, dim=2)
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
        word_embeddings_drop = self.word_presentation_dropout(sorted_words_present)
        encoder_out = self.word_bilstm(word_embeddings_drop, sorted_sentlength)
        lstm_out_drop = self.lstmout_dropout(encoder_out)
        outputs = self.hidden2tag(lstm_out_drop)

        scores = self.mfvi.forward(unary_score=outputs, mask=sorted_mask)
        if self.training:
            scores = scores.view(-1, scores.shape[-1])
            sorted_labels = sorted_labels.view(-1)
            loss = self.criterion(scores, sorted_labels)
            return loss
        else:
            tag_seq = torch.argmax(scores, dim=2)
            # recover
            _, recover = torch.sort(indices, dim=0, descending=False)
            mask_raw = sorted_mask[recover]
            tag_seq = tag_seq[recover]
            return mask_raw, tag_seq

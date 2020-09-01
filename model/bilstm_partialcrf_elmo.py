import torch
import torch.nn as nn
from utils.utils import *
import torch.nn.functional as F
from .partial_crf import Partial_CRF
from .encoder import Encoder
from .embedding import Embedding_Layer
from .scalar_mix import *

# 模型模块
class BiLSTM_ELMo_PartialCRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_ELMo_PartialCRF, self).__init__()
        self.word_embedding = Embedding_Layer(vocab=data.word_vocab, emb_dim=data.word_emb_dim, pretrain_emb=data.pretrain_word_emb)
        self.scalar = ScalarMix(3)
        self.use_char = data.use_char
        self.gpu = data.gpu
        elmo_dim = 1024
        if self.use_char:
            self.char_encoder = Encoder(
                feature_extractor=data.char_feature_extractor,
                embedding_dim=data.char_emb_dim,
                hidden_dim=data.char_hidden_dim
            )
            self.char_embedding = Embedding_Layer(vocab=data.char_vocab, emb_dim=data.char_emb_dim, pretrain_emb=data.pretrain_char_emb)
            self.charemb_dropout = nn.Dropout(data.dropout)
        if self.use_char:
            self.word_bilstm = Encoder(
                feature_extractor='BiLSTM',
                embedding_dim=data.word_emb_dim + data.char_hidden_dim+elmo_dim,
                hidden_dim=data.hidden_dim,
                num_layers=data.lstm_layers
            )
        else:
            self.word_bilstm = Encoder(
                feature_extractor='BiLSTM',
                embedding_dim=data.word_emb_dim+elmo_dim,
                hidden_dim=data.hidden_dim,
                num_layers=data.lstm_layers
            )
        self.word_presentation_dropout = nn.Dropout(data.dropout)
        self.lstmout_dropout = nn.Dropout(data.dropout)
        self.hidden2tag = nn.Linear(data.hidden_dim, data.label_size+2)
        self.crf = Partial_CRF(data.label_size, data.gpu)

    # 根据预测值和实际值，计算loss function值
    def get_loss(self, batch):
        word_idxs, char_idxs, elmo_rep, labels = batch
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
        elmo_rep = elmo_rep[indices]
        elmo_rep = self.scalar(torch.split(elmo_rep,1,dim=2)).squeeze(2)
        sorted_words_present = torch.cat((sorted_words_present, elmo_rep), dim=2)
        # add char info
        if self.use_char:
            batchsize = word_idxs.size(0)
            max_sent_length = word_idxs.size(1)
            char_all_instances = char_idxs.view(batchsize*max_sent_length, -1)
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
            if self.gpu:
                char_feature_pad = char_feature_pad.cuda()
            char_feature_final = torch.cat((char_feature, char_feature_pad), 0)
            _, char_recover = torch.sort(char_indices,dim=0, descending=False)
            char_feature_recover = char_feature_final[char_recover]
            char_feature_final = char_feature_recover.view(batchsize, max_sent_length, -1)
            char_feature_final_align_with_word = char_feature_final[indices]
            sorted_words_present = torch.cat((sorted_words_present, char_feature_final_align_with_word), dim=2)
        # model forward
        word_embeddings_drop = self.word_presentation_dropout(sorted_words_present)
        encoder_out = self.word_bilstm(word_embeddings_drop, sorted_sentlength)
        lstm_out_drop = self.lstmout_dropout(encoder_out)
        outputs = self.hidden2tag(lstm_out_drop)
        #use crf
        loss = self.crf.neg_log_likelihood_loss(outputs, sorted_mask, sorted_labels)
        return loss

    # 根据输入，得到模型输出
    def forward(self, batch):
        word_idxs, char_idxs,elmo_rep, labels = batch
        mask = word_idxs.gt(0)
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
        elmo_rep = elmo_rep[indices]
        elmo_rep = self.scalar(torch.split(elmo_rep,1,2)).squeeze(2)
        sorted_words_present = torch.cat((sorted_words_present, elmo_rep), dim=2)
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
            if self.gpu:
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

        if self.training:
            loss = self.crf.neg_log_likelihood_loss(outputs, sorted_mask, sorted_labels)
            return loss
        else:
            tag_seq = self.crf._viterbi_decode(feats=outputs, mask=sorted_mask)
            # recover
            _, recover = torch.sort(indices, dim=0, descending=False)
            mask_raw = sorted_mask[recover]
            tag_seq = tag_seq[recover]
            return mask_raw, tag_seq

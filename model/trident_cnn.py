import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from model.embedding import Embedding_Layer
from model.crf import CRF
from model.encoder import Encoder

class AdversarialNetwork(nn.Module):
    def __init__(self, data):
        super(AdversarialNetwork, self).__init__()
        self.filter_window_3_cnn = nn.Conv1d(
            in_channels=data.common_hidden_dim,
            out_channels=100,
            kernel_size=3
        )
        self.filter_window_4_cnn = nn.Conv1d(
            in_channels=data.common_hidden_dim,
            out_channels=100,
            kernel_size=4
        )
        self.filter_window_5_cnn = nn.Conv1d(
            in_channels=data.common_hidden_dim,
            out_channels=100,
            kernel_size=5
        )
        self.hidden2domain = nn.Linear(300, 2)
        self.cls_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.5)
        self.gpu = data.gpu
    
    def forward(self, source_feature, target_feature, epoch_num):
        if epoch_num % 2 == 0:
            domain_label_source = torch.zeros(source_feature.size(0), dtype=torch.long)
            domain_label_target = torch.ones(target_feature.size(0), dtype=torch.long)
        else:    
            domain_label_source = torch.ones(source_feature.size(0), dtype=torch.long)
            domain_label_target = torch.zeros(target_feature.size(0), dtype=torch.long)
        if self.gpu:
            domain_label_source = domain_label_source.cuda()
            domain_label_target = domain_label_target.cuda()
        # CNN feature extract
        source_feature = source_feature.transpose(2,1)
        target_feature = target_feature.transpose(2,1)
        source_cnn_out1 = self.filter_window_3_cnn(source_feature)
        source_cnn_out2 = self.filter_window_4_cnn(source_feature)
        source_cnn_out3 = self.filter_window_5_cnn(source_feature)
        target_cnn_out1 = self.filter_window_3_cnn(target_feature)
        target_cnn_out2 = self.filter_window_4_cnn(target_feature)
        target_cnn_out3 = self.filter_window_5_cnn(target_feature)
        
        source_cnn_out1 = F.relu(source_cnn_out1)
        source_cnn_out2 = F.relu(source_cnn_out2)
        source_cnn_out3 = F.relu(source_cnn_out3)
        target_cnn_out1 = F.relu(target_cnn_out1)
        target_cnn_out2 = F.relu(target_cnn_out2)
        target_cnn_out3 = F.relu(target_cnn_out3)

        source_pooling_out1,_ = torch.max(source_cnn_out1, dim=2)
        source_pooling_out2,_ = torch.max(source_cnn_out2, dim=2)
        source_pooling_out3,_ = torch.max(source_cnn_out3, dim=2)
        target_pooling_out1,_ = torch.max(target_cnn_out1, dim=2)
        target_pooling_out2,_ = torch.max(target_cnn_out2, dim=2)
        target_pooling_out3,_ = torch.max(target_cnn_out3, dim=2)
        cls_source = torch.cat((source_pooling_out1, source_pooling_out2, source_pooling_out3), dim=-1)
        cls_target = torch.cat((target_pooling_out1, target_pooling_out2, target_pooling_out3), dim=-1)
        
        cls_source = self.dropout(cls_source)
        cls_target = self.dropout(cls_target)
        source_out = self.hidden2domain(cls_source)
        target_out = self.hidden2domain(cls_target)

        loss_s = self.cls_loss(source_out, domain_label_source)
        loss_t = self.cls_loss(target_out, domain_label_target)

        return loss_s + loss_t


class Trident_Transfer_Net(nn.Module):
    def __init__(self, data):
        super(Trident_Transfer_Net, self).__init__()
        self.word_embedding = Embedding_Layer(vocab=data.word_vocab, emb_dim=data.word_emb_dim, pretrain_emb=data.pretrain_word_emb)
        # private Feature Encoder
        self.source_private_lstm = Encoder(
            feature_extractor='BiLSTM',
            embedding_dim=data.word_emb_dim,
            hidden_dim=data.private_hidden_dim,
            num_layers=1
        )
        self.target_private_lstm = Encoder(
            feature_extractor='BiLSTM',
            embedding_dim=data.word_emb_dim,
            hidden_dim=data.private_hidden_dim,
            num_layers=1
        )
        # share Feature Encoder-CNN
        self.word2cnn = nn.Linear(data.word_emb_dim, data.common_hidden_dim)
        self.cnn_list = nn.ModuleList()
        self.cnn_drop_list = nn.ModuleList()
        self.cnn_norm_list = nn.ModuleList()
        for idx in range(4):
            self.cnn_list.append(nn.Conv1d(in_channels=data.common_hidden_dim, out_channels=data.common_hidden_dim, kernel_size=3, padding=1))
            self.cnn_drop_list.append(nn.Dropout(0.5))
            self.cnn_norm_list.append(nn.BatchNorm1d(data.common_hidden_dim))
        # Classification Layer
        self.hidden2tag_source = nn.Linear(data.common_hidden_dim+data.private_hidden_dim, data.source_label_size+2)
        self.hidden2tag_target = nn.Linear(data.common_hidden_dim+data.private_hidden_dim, data.target_label_size+2)
        # dropout
        self.word_dropout = nn.Dropout(data.dropout)
        self.source_private_lstm_dropout = nn.Dropout(data.dropout)
        self.common_lstm_dropout = nn.Dropout(data.dropout)
        self.target_private_lstm_dropout = nn.Dropout(data.dropout)
        # loss calculate
        self.source_crf = CRF(data.source_label_size, data.gpu)
        self.target_crf = CRF(data.target_label_size, data.gpu)

    def forward(self, batch):
        if self.training:
            batch_source, batch_target = batch
            source_word_idxs, source_char_idxs, _, source_labels = batch_source
            target_word_idxs, target_char_idx, _, target_labels = batch_target
            
            # source
            source_mask = source_word_idxs.gt(0)
            source_sent_lengths = source_mask.sum(1)
            source_words_present = self.word_embedding(source_word_idxs)

            sorted_source_sentlength, source_indices = torch.sort(source_sent_lengths, dim=0, descending=True)
            sorted_source_word_present = source_words_present[source_indices]
            sorted_source_mask = source_mask[source_indices]
            sorted_source_labels = source_labels[source_indices]

            # target
            target_mask = target_word_idxs.gt(0)
            target_sent_lengths = target_mask.sum(1)
            target_words_present = self.word_embedding(target_word_idxs)

            sorted_target_sentlength, target_indices = torch.sort(target_sent_lengths, dim=0, descending=True)
            sorted_target_word_present = target_words_present[target_indices]
            sorted_target_mask = target_mask[target_indices]
            sorted_target_labels = target_labels[target_indices]

            # feature extractor-common
            source_word_repre = torch.tanh(self.word2cnn(sorted_source_word_present)).permute(0,2,1)
            target_word_repre = torch.tanh(self.word2cnn(sorted_target_word_present)).permute(0,2,1)
            for idx in range(4):
                if idx == 0:
                    source_cnn_feature = F.relu(self.cnn_list[idx](source_word_repre))
                    target_cnn_feature = F.relu(self.cnn_list[idx](target_word_repre))
                else:
                    source_cnn_feature = F.relu(self.cnn_list[idx](source_cnn_feature))
                    target_cnn_feature = F.relu(self.cnn_list[idx](target_cnn_feature))
                source_cnn_feature = self.cnn_drop_list[idx](source_cnn_feature)
                source_cnn_feature = self.cnn_norm_list[idx](source_cnn_feature)
                target_cnn_feature = self.cnn_drop_list[idx](target_cnn_feature)
                target_cnn_feature = self.cnn_norm_list[idx](target_cnn_feature)
            common_feature4source = source_cnn_feature.permute(0,2,1)
            common_feature4target = target_cnn_feature.permute(0,2,1)
            # feature extractor-private
            private_feature_source = self.source_private_lstm(sorted_source_word_present, sorted_source_sentlength)
            private_feature_source = self.source_private_lstm_dropout(private_feature_source)
            private_feature_target = self.target_private_lstm(sorted_target_word_present, sorted_target_sentlength)
            private_feature_target = self.target_private_lstm_dropout(private_feature_target)

            source_feature = torch.cat((private_feature_source, common_feature4source), dim=-1)
            target_feature = torch.cat((private_feature_target, common_feature4target), dim=-1)

            # Fully-Connected Layer
            outputs_source = self.hidden2tag_source(source_feature)
            outputs_target = self.hidden2tag_target(target_feature)
            
            # CRF Layer
            source_sl_loss = self.source_crf.neg_log_likelihood_loss(outputs_source, sorted_source_mask, sorted_source_labels)
            target_sl_loss = self.target_crf.neg_log_likelihood_loss(outputs_target, sorted_target_mask, sorted_target_labels)

            # source_sl_average_loss = source_sl_loss/source_word_idxs.size(0)
            # target_sl_average_loss = target_sl_loss/target_word_idxs.size(0)
            return source_sl_loss, target_sl_loss, common_feature4source, common_feature4target
        else:
            word_idx, char_idx, _, real_tagseq = batch
            mask = real_tagseq.gt(0)
            word_emb = self.word_embedding(word_idx)
            sent_length = word_idx.gt(0).sum(1)
            sorted_sentlength, indices = torch.sort(sent_length, dim=0, descending=True)
            sorted_wordemb = word_emb[indices]
            sorted_mask = mask[indices]
            sorted_wordemb = self.word_dropout(sorted_wordemb)
            wordemb4cnn = torch.tanh(self.word2cnn(sorted_wordemb)).permute(0,2,1)
            for idx in range(4):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](wordemb4cnn))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_norm_list[idx](cnn_feature)
            common_feature = cnn_feature.permute(0,2,1)
            private_feature = self.target_private_lstm(sorted_wordemb, sorted_sentlength)
            feature = torch.cat((private_feature, common_feature), dim=-1)
            outputs = self.hidden2tag_target(feature)
            tag_seq = self.target_crf._viterbi_decode(feats=outputs, mask=sorted_mask)

            # recover
            _, recover = torch.sort(indices, dim=0, descending=False)
            mask_raw = sorted_mask[recover]
            tag_seq = tag_seq[recover]
            return mask_raw, tag_seq
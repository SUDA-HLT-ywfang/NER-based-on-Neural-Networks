import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Function
from model.embedding import Embedding_Layer
from model.crf import CRF
from model.encoder import Encoder

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.hidden2domain = nn.Linear(in_feature, 2)
    self.pooling = nn.AdaptiveAvgPool2d((1, in_feature))
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 100.0
    self.cls_loss = nn.CrossEntropyLoss()

  def forward(self, source_feature, target_feature):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    source_feature = source_feature * 1.0
    source_feature.register_hook(grl_hook(coeff))
    target_feature = target_feature * 1.0
    target_feature.register_hook(grl_hook(coeff))

    cls_source = self.pooling(source_feature).squeeze(dim=1)
    cls_target = self.pooling(target_feature).squeeze(dim=1)
    source_out = self.hidden2domain(cls_source)
    target_out = self.hidden2domain(cls_target)

    domain_label_source = torch.zeros(source_feature.size(0), dtype=torch.long)
    domain_label_target = torch.ones(target_feature.size(0), dtype=torch.long)
    loss_s = self.cls_loss(source_out, domain_label_source)
    loss_t = self.cls_loss(target_out, domain_label_target)

    return loss_s + loss_t

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

class DANN_lstm(nn.Module):
    def __init__(self, data):
        super(DANN_lstm, self).__init__()
        self.word_embedding = Embedding_Layer(vocab=data.word_vocab, emb_dim=data.word_emb_dim, pretrain_emb=data.pretrain_word_emb)
        self.feature_extractor = Encoder(
            feature_extractor='BiLSTM',
            embedding_dim=data.word_emb_dim,
            hidden_dim=data.hidden_dim,
            num_layers=data.lstm_layers)
        # decoder
        self.word_dropout = nn.Dropout(0.5)
        self.lstm_dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(data.hidden_dim, data.label_size+2)
        # loss calculate
        self.crf = CRF(data.label_size, data.gpu)

    def forward(self, batch):
        if self.training:
            source_batch, target_batch = batch
            source_word_idx, source_char_idx, _,source_real_tagseq = source_batch
            target_word_idx, target_char_idx, _, _= target_batch
            source_batch_size = source_word_idx.size(0)
            target_batch_size = target_word_idx.size(0)
            source_word_mask = source_word_idx.gt(0)
            target_word_mask = target_word_idx.gt(0)

            # source domain
            source_sent_length = source_word_idx.gt(0).sum(1)
            sort_source_sent_length, source_indices = torch.sort(source_sent_length, dim=0, descending=True)
            sorted_source_wordemb = self.word_embedding(source_word_idx)[source_indices]
            sorted_source_wordemb = self.word_dropout(sorted_source_wordemb)
            source_lstmout = self.feature_extractor(sorted_source_wordemb, sort_source_sent_length)
            source_feature = self.lstm_dropout(source_lstmout)

            # target domain
            target_sent_length = target_word_idx.gt(0).sum(1)
            sort_target_sent_length, target_indices = torch.sort(target_sent_length, dim=0, descending=True)
            sorted_target_wordemb = self.word_embedding(target_word_idx)[target_indices]
            sorted_target_wordemb = self.word_dropout(sorted_target_wordemb)
            target_lstmout = self.feature_extractor(sorted_target_wordemb, sort_target_sent_length)
            target_feature = self.lstm_dropout(target_lstmout)

            # ner loss
            sorted_labels = source_real_tagseq[source_indices]
            sorted_mask = source_word_mask[source_indices]
            tagprob = self.hidden2tag(source_feature)
            ner_loss = self.crf.neg_log_likelihood_loss(tagprob, sorted_mask, sorted_labels)
            
            return source_feature, target_feature, ner_loss
        else:
            word_idx, char_idx, _, real_tagseq = batch
            mask = real_tagseq.gt(0)
            word_emb = self.word_embedding(word_idx)
            sent_length = word_idx.gt(0).sum(1)
            sorted_sentlength, indices = torch.sort(sent_length, dim=0, descending=True)
            sorted_wordemb = word_emb[indices]
            sorted_mask = mask[indices]
            sorted_wordemb = self.word_dropout(sorted_wordemb)
            lstmout = self.feature_extractor(sorted_wordemb, sorted_sentlength)
            feature = self.lstm_dropout(lstmout)
            feature = self.hidden2tag(feature)
            tag_seq = self.crf._viterbi_decode(feats=feature, mask=sorted_mask)

            # recover
            _, recover = torch.sort(indices, dim=0, descending=False)
            mask_raw = sorted_mask[recover]
            tag_seq = tag_seq[recover]
            return mask_raw, tag_seq
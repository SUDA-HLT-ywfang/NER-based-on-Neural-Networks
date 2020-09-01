import torch
import torch.nn as nn
from model.bert_encoder import Bert_Encoder


class FineTuneTagger(nn.Module):
    def __init__(self, bert_path, bert_dim, tag_size):
        super(FineTuneTagger, self).__init__()
        self.bert_encoder = Bert_Encoder(bert_path, bert_dim)
        self.hidden2tag = nn.Linear(bert_dim, tag_size)
        """
        nn.CrossEntropyLoss:
            Input: N*C
            Target: N
            output = loss(Input, Target)
        """
        self.loss_calculator = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, batch):
        subword_idxs, subword_head_mask, subword_mask, real_tagseq = batch
        bert_outs = self.bert_encoder(subword_idxs, subword_head_mask)
        out = self.hidden2tag(bert_outs)

        if self.training:
            out = out.view(-1, out.shape[-1])
            real_tagseq = real_tagseq.view(-1)
            loss = self.loss_calculator(out, real_tagseq)
            return loss
        else:
            predict_tag = torch.argmax(out, 2)
            mask = real_tagseq.gt(0)
            return mask, predict_tag
import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from torch.nn.utils.rnn import pad_sequence


class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_dim, freeze=False):
        super(Bert_Encoder, self).__init__()
        self.bert_dim = bert_dim
        self.bert = BertModel.from_pretrained(bert_path)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, token_start_masks):
        sent_lens = token_start_masks.sum(dim=1)
        mask = subword_idxs.gt(0)
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=mask
        )
        # token_start_masks = token_start_masks.unsqueeze(dim=2).expand(-1, -1, self.bert_dim)
        # only_first_subword = torch.masked_select(bert_outs, token_start_masks).view(-1, self.bert_dim)
        bert_outs = torch.split(bert_outs[token_start_masks], sent_lens.tolist(), dim=0)
        bert_outs = pad_sequence(bert_outs, batch_first=True)

        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


class Bert_Embedding(nn.Module):
    def __init__(self, bertpath, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding, self).__init__()
        self.bert_layer = bert_layer
        self.bert = BertModel.from_pretrained(bertpath, output_hidden_states=True)
        self.bert_dim = bert_dim

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_mask, token_starts_masks, strategy):
        self.eval()
        sent_lengths = token_starts_masks.sum(dim=1)
        mask = subword_idxs.gt(0)
        last_layer, _, hidden_states = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=mask
        )
        bert_outs = hidden_states[len(hidden_states)-self.bert_layer:len(hidden_states)]
        if strategy == 'concat_last_4':
            concat_bert_outs = torch.cat(bert_outs, dim=2)
            final_dim = self.bert_layer*self.bert_dim
        elif strategy == 'sum_last_4':
            concat_bert_outs = sum(bert_outs)
            final_dim = self.bert_dim
        else:
            print("Wrong Bert Embedding Using strategy...")
            exit(1)
        token_start_masks = token_starts_masks.unsqueeze(dim=2).expand(-1, -1, final_dim)
        only_first_subword = torch.masked_select(concat_bert_outs, token_start_masks).view(-1, final_dim)
        bert_outs = torch.split(only_first_subword, sent_lengths.tolist(), dim=0)
        bert_outs = pad_sequence(bert_outs, batch_first=True)

        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


# 返回带start_token序列
class Bert_Embedding2(nn.Module):
    def __init__(self, bertpath, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding2, self).__init__()
        self.bert_layer = bert_layer
        self.bert = BertModel.from_pretrained(bertpath, output_hidden_states=True)
        self.bert_dim = bert_dim

        self.eval()
        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_mask, token_starts_masks):
        subtoken_lengths = subword_mask.sum(dim=1)
        mask = subword_idxs.gt(0)
        last_layer, _, hidden_states = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=mask
        )
        bert_outs = hidden_states[len(hidden_states)-self.bert_layer:len(hidden_states)]
        concat_bert_outs = torch.cat(bert_outs, dim=2)
        token_start_masks = subword_mask.unsqueeze(dim=2).expand(-1, -1, self.bert_dim*self.bert_layer)
        only_first_subword = torch.masked_select(concat_bert_outs, token_start_masks).view(-1, self.bert_dim*self.bert_layer)
        bert_outs = torch.split(only_first_subword, subtoken_lengths.tolist(), dim=0)
        bert_outs = pad_sequence(bert_outs, batch_first=True)

        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False

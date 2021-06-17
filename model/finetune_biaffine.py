import torch
import torch.nn as nn

from model.bert_encoder import Bert_Encoder

from .biaffine import Biaffine


class FineTune_Biaffine(nn.Module):
    def __init__(self, data):
        super(FineTune_Biaffine, self).__init__()
        self.bert_encoder = Bert_Encoder(data.get("bert_path"), data.get("bert_dim"))
        self.non_entity_idx = data.vocab_label.get_id("O")
        self.dropout = nn.Dropout(0.1)

        self.ffnn_start = nn.Linear(in_features=data.get("bert_dim"), out_features=150)
        self.ffnn_end = nn.Linear(in_features=data.get("bert_dim"), out_features=150)

        self.biaffine = Biaffine(n_in=150, n_out=data.vocab_label.get_size(), bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, batch):
        subword_idxs, subword_head_mask, real_tagseq = batch
        bert_outs = self.bert_encoder(subword_idxs, subword_head_mask)
        bert_outs = self.dropout(bert_outs)

        span_start_rep = self.ffnn_start(bert_outs)
        span_end_rep = self.ffnn_end(bert_outs)

        score = self.biaffine(x=span_start_rep, y=span_end_rep).permute(0, 2, 3, 1)

        if self.training:
            score_for_loss = score.reshape(-1, score.shape[-1])
            gold_score = real_tagseq.reshape(-1)
            loss = self.criterion(input=score_for_loss, target=gold_score)

            return loss
        else:
            return self.decode(score, real_tagseq.gt(0))

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
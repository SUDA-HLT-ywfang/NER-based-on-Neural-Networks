import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn

START = -2
STOP = -1


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


class CRF(nn.Module):
    def __init__(self, label_size, gpu):
        super(CRF, self).__init__()
        self.label_size = label_size
        self.use_gpu = gpu
        init_trans = torch.zeros(self.label_size+2, self.label_size+2)
        init_trans[:, START] = -10000.0
        init_trans[STOP,:] = -10000.0
        init_trans[:,0] = -10000.0
        init_trans[0,:] = -10000.0
        if self.use_gpu:
            init_trans = init_trans.cuda()
        self.transitions = nn.Parameter(init_trans)

    def _get_PZ(self, emit, mask):
        batch_size, sent_length, label_size = emit.size()
        assert label_size == self.label_size + 2
        mask = mask.transpose(1, 0)
        ins_num = batch_size * sent_length
        emits = emit.transpose(1,0).contiguous().view(ins_num,1,label_size).expand(ins_num, label_size, label_size)
        trans = self.transitions.view(1, label_size, label_size).expand(ins_num, label_size, label_size)
        scores = emits + trans
        scores = scores.view(sent_length, batch_size, label_size, label_size)
        seq_iter = enumerate(scores)
        _, init_values = next(seq_iter)
        previous = init_values[:,START,:].clone()
        for i, value in seq_iter:
            temp = previous.view(batch_size, label_size,1).expand(batch_size, label_size, label_size)
            new_score = temp + value
            new_previous = log_sum_exp(new_score, label_size)
            mask_idx = mask[i,:].view(batch_size, 1).expand(batch_size, label_size)
            previous.masked_scatter_(mask_idx, new_previous)
        temp = self.transitions.view(1, label_size, label_size).expand(batch_size, label_size, label_size) + \
               previous.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
        previous = log_sum_exp(temp, label_size)
        final = previous[:, STOP]
        return final.sum(), scores

    def _score_sentence(self, emit, mask, tags):
        batch_size = emit.size(1)
        sent_length = emit.size(0)
        label_size = emit.size(2)
        # 计算句子结束部分的分数
        end_transition = self.transitions[:,STOP].view(1, label_size).expand(batch_size, label_size)
        length = mask.sum(dim=1).view(batch_size, 1)
        end_ids = torch.gather(tags, dim=1, index=length-1)
        end_scores = torch.gather(end_transition, dim=1, index=end_ids)
        # 转换tags格式，以利用传入的emit
        new_tags = torch.zeros(batch_size, sent_length,dtype=torch.long)
        if self.use_gpu:
            new_tags = new_tags.cuda()
        for i in range(sent_length):
            if i == 0:
                new_tags[:,0] = (label_size-2)*label_size + tags[:,0]
            else:
                new_tags[:,i] = (tags[:,i-1])*label_size + tags[:,i]
        new_tags = new_tags.transpose(1, 0).view(sent_length, batch_size, 1)
        mid_scores = torch.gather(emit.view(sent_length, batch_size, -1), 2, new_tags).view(sent_length, batch_size)
        mid_scores = mid_scores.masked_select(mask.transpose(1, 0))
        gold_scores = end_scores.sum() + mid_scores.sum()
        return gold_scores

    def neg_log_likelihood_loss(self, emit, mask, tags):
        all_score, temp = self._get_PZ(emit, mask)
        gold_score = self._score_sentence(temp, mask, tags)
        return all_score - gold_score

    def forward(self, emits):
        path_score, best_path = self._viterbi_decode(emits)
        return path_score, best_path

    def _viterbi_decode(self, feats, mask):
        batch_size, sent_length, label_size = feats.shape
        length_mask = mask.sum(dim=1).view(batch_size, 1)
        mask = mask.transpose(1,0)
        reverse_mask = (1 - mask.long()).bool()     # 适用于pytorch==1.3
        # reverse_mask = (1 - mask.long()).byte()     # 适用于pytorch==1.1
        feats = feats.transpose(1,0) # sent_length, batch_size, label_size
        full_trans = self.transitions.unsqueeze(0).expand(batch_size, label_size, label_size)
        init_var = full_trans[:, START, :].view(batch_size, label_size)
        seq_iter = enumerate(feats)
        _, first_feat = next(seq_iter)
        init_var = init_var + first_feat
        best_scores_rec, best_tag_id_rec = list(), list()
        best_scores_rec.append(init_var)
        for ind, current_feat in seq_iter:
            scores = init_var.unsqueeze(2).expand(batch_size, label_size, label_size) + full_trans
            scores += current_feat.unsqueeze(1).expand(batch_size, label_size, label_size)
            init_var, best_id = torch.max(scores, dim=1)
            best_id.masked_fill_(reverse_mask[ind].view(batch_size, 1).expand(batch_size, label_size), 0)
            best_scores_rec.append(init_var)
            best_tag_id_rec.append(best_id)
        # 句子长度不一，需要获得真正的句子最后一个字的分数状态
        partition_history = torch.cat(tuple(best_scores_rec), 0).view(sent_length, batch_size, label_size).transpose(1, 0)
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, label_size) - 1
        """
        torch.gather(input, dim, index): Gather values along an axis specified by dim.
            For a 3-D tensor the output is specified by:
            out[i][j][k] = input[i][index[i][j][k]][k] ## if dim == 1
            out will have the same size as index
        """
        last_scores = torch.gather(partition_history,dim=1,index=last_position).view(batch_size, label_size, 1)
        final_scores = last_scores.expand(batch_size, label_size, label_size) + full_trans
        _, last_best_tagids = torch.max(final_scores, dim=1)
        # 既然加了转移到最后tag的概率，最后一个tag肯定是STOP，只需要看使最后一个tag为STOP的分数最高的前一个tag是什么
        pointer = last_best_tagids[:, STOP]
        # pointer2best_tag_id = pointer.view(batch_size, 1).expand(batch_size, label_size)
        # best_tag_id_rec.append(pointer2best_tag_id)
        """
        best_tag_id_rec：记得是 到当前位置，使当前位置的标签a分数最高的前一个标签。
        到第一个位置最大的前一标签肯定都是START_TAG，就不用记了。
        迭代到句尾，长度应该是（最大句长-1）。在append一个使到STOP最大的最后一个位置的标签。
        pointer强行写到back_points每句句长的位置（即最后一个位置）上，回溯时候就不会影响。
        """
        pad_zeros = torch.zeros(batch_size, label_size, dtype=torch.int64)
        if self.use_gpu:
            pad_zeros = pad_zeros.cuda()
        best_tag_id_rec.append(pad_zeros)
        back_points = torch.cat(best_tag_id_rec,0).view(sent_length, batch_size, label_size)
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, label_size)
        back_points = back_points.transpose(1, 0).contiguous()
        """
        Tensor.scatter_(dim, index, src):
        Writes all values from the tensor src into self at the indices specified in the index tensor.
            For a 3-D tensor, self is updated as:
            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
        index有多少，就要复制过去多少。
        复制的位置由index内的值决定，复制的值由source内的决定。每个句子需要复制label_size个，
        通通置成使到STOP_TAG最大的label。
        """
        # back_points转成batch_size*sent_length*lable_size，覆盖写最后一个位置
        back_points.scatter_(1, last_position, insert_last)
        # 再转成sent_length第一维，便于回溯
        back_points = back_points.transpose(1, 0).contiguous()
        #decode_idx = autograd.Variable(torch.LongTensor(sent_length, batch_size))
        decode_idx = torch.LongTensor(sent_length, batch_size)
        decode_idx[-1] = pointer
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(batch_size)
        decode_idx = decode_idx.transpose(1, 0)
        return decode_idx


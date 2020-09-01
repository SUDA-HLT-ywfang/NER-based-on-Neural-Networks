import torch
import torch.nn as nn
import torch.autograd as autograd

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


class Partial_CRF(nn.Module):
    def __init__(self, label_size, gpu):
        super(Partial_CRF, self).__init__()
        self.label_size = label_size
        self.use_gpu = gpu
        # 表示从行转到列的概率
        init_trans = torch.zeros(self.label_size+2, self.label_size+2)
        init_trans[:, START] = -100000.0
        init_trans[STOP,:] = -100000.0
        init_trans[:,0] = -100000.0
        init_trans[0,:] = -100000.0
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

    def _score_sentence(self, sum_scores, mask, uncertain_mask):
        """
        :param scores: batch_size*sent_length*label_size*label_size，句子每个位置的发射分数+转移分数
        :param mask: batch_size*sent_length, 记录句子的pad位置
        :param uncertain_mask: batch_size*sent_length*label_size, 句子的每个位置的每种标签是否有可能，1为有可能，0为没可能
        :return:
        """
        batch_size = sum_scores.size(1)
        label_size = sum_scores.size(2)
        mask = mask.transpose(1, 0)
        uncertain_mask = uncertain_mask.byte()
        uncertain_mask = ~uncertain_mask.transpose(1, 0)   # invert a mask，转成0为有可能，1为不可能
        seq_iter = enumerate(sum_scores)
        _, init_values = next(seq_iter)
        previousmax = init_values[:, START, :].clone()
        possible_tags = uncertain_mask[0]
        previousmax.masked_fill_(mask=possible_tags, value=float('-inf'))
        for i, value in seq_iter:
            temp = previousmax.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
            new_score = temp + value
            current_possible_tags = uncertain_mask[i].view(batch_size, 1, label_size).expand(batch_size, label_size, label_size)
            new_score.masked_fill_(mask=current_possible_tags, value=float('-inf'))
            new_previous = torch.logsumexp(new_score, dim=1)
            # new_previous = log_sum_exp(new_score, label_size)
            mask_idx = mask[i, :].view(batch_size, 1).expand(batch_size, label_size)
            # previous.masked_scatter_(mask_idx, new_previous)
            previousmax[mask_idx] = new_previous[mask_idx]
        temp = self.transitions.view(1, label_size, label_size).expand(batch_size, label_size, label_size) + \
               previousmax.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
        previousmax = torch.logsumexp(temp, dim=1)
        final = previousmax[:, STOP]
        return final.sum()

    def neg_log_likelihood_loss(self, emit, mask, tags):
        # 0表示确定，1表示不确定
        # emit = emit.double()
        all_score, temp = self._get_PZ(emit, mask)
        gold_score = self._score_sentence(temp, mask, tags)
        # probs, = torch.autograd.grad(all_score, emit, retain_graph=self.training)
        return all_score - gold_score

    def forward(self, emits):
        path_score, best_path = self._viterbi_decode(emits)
        return path_score, best_path

    def _viterbi_decode(self, feats, mask):
        batch_size, sent_length, label_size = feats.shape
        length_mask = mask.sum(dim=1).view(batch_size, 1)
        mask = mask.transpose(1,0)
        reverse_mask = (1 - mask.long()).byte()
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


class Partial_CRF_Setinf(nn.Module):
    def __init__(self, label_vocab, gpu):
        super(Partial_CRF_Setinf, self).__init__()
        self.label_size = label_vocab.get_size()
        self.use_gpu = gpu
        # 表示从行转到列的概率
        init_trans = torch.zeros(self.label_size + 2, self.label_size + 2)
        init_trans[:, START] = -100000.0
        init_trans[STOP, :] = -100000.0
        init_trans[:, 0] = -100000.0
        init_trans[0, :] = -100000.0
        # FIXME: 目前仅支持BIOES编码
        print("Setting all impossible transition scores to -inf......")
        # 将不可能的转移情况全部置成-inf
        # 先得到所有类别
        type_set = set()
        for i in range(self.label_size):
            label = label_vocab.get_instance(i)
            label = label.split('-')
            if len(label) == 2:
                type_set.add(label[1])
        print("There are %d kinds of entities." % (len(type_set)))
        # 遍历所有标签，把所有可能的列出来
        # FIXME: B-PER到结束符是不可以的,开始标签不能到I和E
        possible_trans_pair = set()
        for begin_type in type_set:
            possible_trans_pair.add(('S-'+begin_type, 'O'))
            possible_trans_pair.add(('E-'+begin_type, 'O'))
            for end_type in type_set:
                if begin_type == end_type:
                    possible_trans_pair.add(('B-'+begin_type, 'I-'+end_type))
                    possible_trans_pair.add(('B-'+begin_type, 'E-'+end_type))
                    possible_trans_pair.add(('I-'+begin_type, 'E-'+end_type))
                    possible_trans_pair.add(('I-'+begin_type, 'I-'+end_type))
                possible_trans_pair.add(('E-'+begin_type, 'B-'+end_type))
                possible_trans_pair.add(('E-'+begin_type, 'S-'+end_type))
                possible_trans_pair.add(('S-'+begin_type, 'B-'+end_type))
                possible_trans_pair.add(('S-'+begin_type, 'S-'+end_type))
                possible_trans_pair.add(('O', 'S-'+end_type))
                possible_trans_pair.add(('O', 'B-'+end_type))
        possible_trans_pair.add(('O', 'O'))
        for begin_id in range(1, self.label_size):
            begin_tag = label_vocab.get_instance(begin_id)
            for end_id in range(1, self.label_size):
                end_tag = label_vocab.get_instance(end_id)
                if (begin_tag, end_tag) not in possible_trans_pair:
                    init_trans[begin_id, end_id] = -100000.0
                    # print("illegal trans: %s to %s" % (begin_tag, end_tag))
        # 特殊优化开始和结束标签
        for t in type_set:
            init_trans[label_vocab.get_id('B-'+t), STOP] = -100000.0
            init_trans[label_vocab.get_id('I-'+t), STOP] = -100000.0
            init_trans[START, label_vocab.get_id('I-'+t)] = -100000.0
            init_trans[START, label_vocab.get_id('E-'+t)] = -100000.0
        # # for debug
        # print(label_vocab.instance2id)
        # print(init_trans.numpy())
        # exit(1)
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

    def _score_sentence(self, sum_scores, mask, uncertain_mask):
        """
        :param scores: batch_size*sent_length*label_size*label_size，句子每个位置的发射分数+转移分数
        :param mask: batch_size*sent_length, 记录句子的pad位置
        :param uncertain_mask: batch_size*sent_length*label_size, 句子的每个位置的每种标签是否有可能，1为有可能，0为没可能
        :return:
        """
        batch_size = sum_scores.size(1)
        label_size = sum_scores.size(2)
        mask = mask.transpose(1, 0)
        uncertain_mask = uncertain_mask.byte()
        uncertain_mask = ~uncertain_mask.transpose(1, 0)   # invert a mask，转成0为有可能，1为不可能
        seq_iter = enumerate(sum_scores)
        _, init_values = next(seq_iter)
        previousmax = init_values[:, START, :].clone()
        possible_tags = uncertain_mask[0]
        previousmax.masked_fill_(mask=possible_tags, value=float('-inf'))
        for i, value in seq_iter:
            temp = previousmax.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
            new_score = temp + value
            current_possible_tags = uncertain_mask[i].view(batch_size, 1, label_size).expand(batch_size, label_size, label_size)
            new_score.masked_fill_(mask=current_possible_tags, value=float('-inf'))
            new_previous = torch.logsumexp(new_score, dim=1)
            # new_previous = log_sum_exp(new_score, label_size)
            mask_idx = mask[i, :].view(batch_size, 1).expand(batch_size, label_size)
            # previous.masked_scatter_(mask_idx, new_previous)
            previousmax[mask_idx] = new_previous[mask_idx]
        temp = self.transitions.view(1, label_size, label_size).expand(batch_size, label_size, label_size) + \
               previousmax.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
        previousmax = torch.logsumexp(temp, dim=1)
        final = previousmax[:, STOP]
        return final.sum()

    def neg_log_likelihood_loss(self, emit, mask, tags):
        # 0表示确定，1表示不确定
        emit = emit.double()
        all_score, temp = self._get_PZ(emit, mask)
        gold_score = self._score_sentence(temp, mask, tags)
        # probs, = torch.autograd.grad(all_score, emit, retain_graph=self.training)
        return all_score - gold_score

    def forward(self, emits):
        path_score, best_path = self._viterbi_decode(emits)
        return path_score, best_path

    def _viterbi_decode(self, feats, mask):
        batch_size, sent_length, label_size = feats.shape
        length_mask = mask.sum(dim=1).view(batch_size, 1)
        mask = mask.transpose(1,0)
        reverse_mask = (1 - mask.long()).byte()
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
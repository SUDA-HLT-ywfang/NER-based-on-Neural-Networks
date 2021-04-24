import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class MFVI(nn.Module):
    def __init__(self, label_size, gpu, window_size=1, iterations=3, add_start_end=False):
        super(MFVI, self).__init__()
        self.label_size = label_size
        self.use_gpu = gpu
        self.window_size = window_size
        self.iterations = iterations
        self.transitions = nn.Parameter(torch.randn(self.window_size,self.label_size, self.label_size))
        self.add_start_end = add_start_end
        if self.add_start_end:
            self.transitions_start = nn.Parameter(torch.randn(self.window_size, self.label_size))
            self.transitions_end = nn.Parameter(torch.randn(self.window_size, self.label_size))

    def forward(self, unary_score, mask):
        unary_score = unary_score * mask.unsqueeze(-1)
        max_sent_len = mask.shape[1]
        if max_sent_len == 1:
            return unary_score
        binary_score=[]
        for i in range(min(self.window_size, max_sent_len - 1)):
            binary_score.append(self.transitions[i])
        scores = self._mean_field_variational_inference(unary=unary_score, binary=binary_score, ternary=None, mask=mask)
        return scores

    def _mean_field_variational_inference(self, unary, binary, ternary, mask):
        unary_potential = unary.clone()
        # init q value
        q_value = unary_potential.clone()

        for iteration in range(self.iterations):
            q_value = F.softmax(q_value, dim=-1)
            s_i_minus_window_size = torch.zeros_like(q_value)
            s_i_plus_window_size = torch.zeros_like(q_value)
            for j in range(1, self.window_size+1):
                # q_value: batch_size * sent_len * label_size
                # binary[j-1]: label_size * label_size。window=j时，声明的转移矩阵参数，即考虑当前结点和前后距离j的结点的转移分数
                s_i_minus_window_size[:, j:] += torch.einsum('nsa,ab->nsb', [q_value[:, :-j], binary[j-1]])
                s_i_minus_window_size[:, :j] += self.transitions_start[j-1]
                s_i_plus_window_size[:, :-j] += torch.einsum('nsb,ab->nsa', [q_value[:, j:], binary[j-1]])
                s_i_plus_window_size[:, j:] += self.transitions_end[j-1]
            second_order_message = s_i_minus_window_size + s_i_plus_window_size
            q_value = unary_potential + second_order_message
            q_value = q_value*mask.unsqueeze(-1)
        return q_value
import torch.nn as nn
from utils.utils import *


class Embedding_Layer(nn.Module):
    def __init__(self, vocab, emb_dim, pretrain_emb=None):
        super(Embedding_Layer, self).__init__()
        self.vocab_size = vocab.get_size()
        # self.word_embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.word_embedding = nn.Embedding(self.vocab_size, emb_dim)
        if pretrain_emb != None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(load_pretrain_embedding(pretrain_emb, vocab, emb_dim)))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(random_init_embedding(self.vocab_size, emb_dim)))

    def forward(self, word_index):
        word_represent = self.word_embedding(word_index)
        return word_represent

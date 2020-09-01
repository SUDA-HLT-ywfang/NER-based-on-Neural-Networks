# @author: FrankCast1e
# @file: feature_based_tagger.py
# @Last Modified time: 19-7-9 下午8:37
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model.bert_encoder import Bert_Embedding
from .encoder import Encoder
from .crf import CRF
from .partial_crf import Partial_CRF
from pytorch_transformers import BertTokenizer
import numpy as np


class FeatureBasedTagger(nn.Module):
    def __init__(self, bert_path, bert_dim, bert_layer, tag_size, use_crf, strategy, word_emb_path, emb_dim, use_partial_crf, label_vocab):
        super(FeatureBasedTagger, self).__init__()
        self.bert_dim = bert_dim
        self.bert_layer = bert_layer
        self.bert_embedding = Bert_Embedding(bertpath=bert_path,
                                             bert_layer=self.bert_layer,
                                             bert_dim=self.bert_dim)
        self.hidden2tag = nn.Linear(self.bert_dim, tag_size+2)
        self.loss_calculator = nn.CrossEntropyLoss(ignore_index=0)
        self.bert_dropout = nn.Dropout(0.5)
        self.lstm_dropout = nn.Dropout(0.5)
        self.use_crf = use_crf
        self.use_partial_crf = use_partial_crf
        self.label_vocab = label_vocab
        self.strategy = strategy
        if self.strategy == 'concat_last_4':
            self.bilstm = Encoder(
                feature_extractor='BiLSTM',
                embedding_dim=self.bert_dim * self.bert_layer+emb_dim,
                hidden_dim=self.bert_dim,
                num_layers=1
            )
        elif self.strategy == 'sum_last_4' or self.strategy == 'mean_last_4':
            self.bilstm = Encoder(
                feature_extractor='BiLSTM',
                embedding_dim=self.bert_dim+emb_dim,
                hidden_dim=self.bert_dim,
                num_layers=1
            )
        else:
            print("Wrong Bert Embedding Using strategy...")
        if self.use_crf:
            self.crf = CRF(tag_size, gpu=torch.cuda.is_available())
        if self.use_partial_crf:
            self.crf = Partial_CRF(tag_size, gpu=torch.cuda.is_available())
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_path, do_lower_case=False)
        self.word_embedding = nn.Embedding(self.bert_tokenizer.vocab_size, emb_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.load_pretrain_embedding(word_emb_path, emb_dim)))

    def load_pretrain_embedding(self, pretrain_emb_filepath, emb_dim):
        print("Loading pretrained embedding from %s" % pretrain_emb_filepath)
        emb = np.zeros((self.bert_tokenizer.vocab_size, emb_dim))
        embedding_dict = dict()
        with open(pretrain_emb_filepath, 'r', encoding='utf-8') as fpin:
            for line in fpin:
                line = line.rstrip().split()
                word = line[0]
                embedding_dict[word] = ' '.join(line[1:])
        scale = np.sqrt(3.0 / emb_dim)
        match, oov = 0, 0
        for id in range(self.bert_tokenizer.vocab_size):
            word = self.bert_tokenizer.convert_ids_to_tokens(id)
            if word in embedding_dict:
                temp = np.fromstring(embedding_dict[word], dtype=float, sep=' ')
                emb[id] = temp
                match += 1
            elif word.lower() in embedding_dict:
                temp = np.fromstring(embedding_dict[word.lower()], dtype=float, sep=' ')
                emb[id] = temp
                match += 1
            else:
                emb[id, :] = np.random.uniform(-scale, scale, [1, emb_dim])
                oov += 1
        print("Embedding: all: %d, match: %d, oov: %d" % (len(embedding_dict), match, oov))
        return emb

    def forward(self, batch):
        # subword_idxs, subword_head_mask, subword_mask, real_tagseq = batch
        # bert_outs = self.bert_embedding(subword_idxs, subword_mask, subword_head_mask)
        #
        # real_sent_lengths = subword_head_mask.sum(dim=1)
        # sent_lengths = subword_mask.sum(dim=1)
        # sorted_sent_length, indices = torch.sort(sent_lengths, dim=0, descending=True)
        # sorted_word_present = bert_outs[indices]
        # lstm_out = self.bilstm(sorted_word_present, sorted_sent_length)
        # lstm_out = self.lstm_dropout(lstm_out)
        # # take subword head
        # max_sent_length = subword_head_mask.size(-1)
        # subword_head_mask = subword_head_mask[:, 1:max_sent_length-1] ### 去除首尾的mask，和lstm输出保持一致
        # sorted_subwordhead = subword_head_mask[indices]
        # sorted_subwordhead = sorted_subwordhead.unsqueeze(dim=2).expand(-1, -1, self.bert_dim)
        # only_first_subword = torch.masked_select(lstm_out, sorted_subwordhead).view((-1, self.bert_dim))
        # lstm_out_head = torch.split(only_first_subword, real_sent_lengths.tolist(), dim=0)
        # lstm_out_head = pad_sequence(lstm_out_head, batch_first=True)
        # ###################
        #
        # out = self.hidden2tag(lstm_out_head)
        # if self.training:
        #     sorted_labels = real_tagseq[indices]
        #     out = out.view(-1, out.shape[-1])
        #     sorted_labels = sorted_labels.view(-1)
        #
        #     loss = self.loss_calculator(out, sorted_labels)
        #     return loss
        # else:
        #     predict_tag = torch.argmax(out, 2)
        #     _, recover = torch.sort(indices, dim=0, descending=False)
        #     predict_tag = predict_tag[recover]
        #     mask = real_tagseq.gt(0)
        #     return mask, predict_tag

        ###
        if self.use_partial_crf and self.training:
            subword_idxs, subword_head_mask, subword_mask, real_tagseq, sentence_mask = batch[:5]
        else:
            subword_idxs, subword_head_mask, subword_mask, real_tagseq = batch[:4]
        sent_lengths = subword_head_mask.sum(dim=1)

        bert_outs = self.bert_embedding(subword_idxs, subword_mask, subword_head_mask, strategy=self.strategy)
        all_word_emb = self.word_embedding(subword_idxs)
        head_word_emb = torch.split(all_word_emb[subword_head_mask], sent_lengths.tolist(), dim=0)
        head_word_emb = pad_sequence(head_word_emb, batch_first=True)
        bert_outs = torch.cat((bert_outs, head_word_emb), dim=2)
        bert_outs = self.bert_dropout(bert_outs)

        sorted_sent_length, indices = torch.sort(sent_lengths, dim=0, descending=True)
        sorted_word_present = bert_outs[indices]
        lstm_out = self.bilstm(sorted_word_present, sorted_sent_length)
        lstm_out = self.lstm_dropout(lstm_out)

        out = self.hidden2tag(lstm_out)

        if self.training:
            sorted_labels = real_tagseq[indices]
            if self.use_crf:
                loss = self.crf.neg_log_likelihood_loss(out, mask=sorted_labels.gt(0), tags=sorted_labels)
            elif self.use_partial_crf:
                # sentence_mask = sentence_mask.bool()
                sentence_mask = sentence_mask.byte()
                loss = self.crf.neg_log_likelihood_loss(out, mask=sentence_mask[indices], tags=sorted_labels)
            else:
                out = out.view(-1, out.shape[-1])
                score = F.log_softmax(out, 1)
                sorted_labels = sorted_labels.view(-1)

                loss = self.loss_calculator(score, sorted_labels)
            return loss
        else:
            if self.use_crf:
                predict_tag = self.crf._viterbi_decode(feats=out, mask=real_tagseq[indices].gt(0))
            elif self.use_partial_crf:
                if len(batch) == 5:
                    # sentence_mask = batch[-1].bool()
                    sentence_mask = batch[-1].byte()
                    predict_tag = self.crf._viterbi_decode(feats=out, mask=sentence_mask[indices])
                    _, recover = torch.sort(indices, dim=0, descending=False)
                    predict_tag = predict_tag[recover]
                    return sentence_mask, predict_tag
                else:
                    predict_tag = self.crf._viterbi_decode(feats=out, mask=real_tagseq[indices].gt(0))
            else:
                predict_tag = torch.argmax(out, 2)
            _, recover = torch.sort(indices, dim=0, descending=False)
            predict_tag = predict_tag[recover]
            mask = real_tagseq.gt(0)
            return mask, predict_tag

# @author: FrankCast1e
# @file: bert_dataset.py
# @Last Modified time: 19-7-8 下午5:05
import torch
from torch.utils import data
import unicodedata
from pytorch_transformers import BertTokenizer


class BertDataSet(data.Dataset):
    def __init__(self, bert_vocab_path, corpus, label_vocab):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_vocab_path, do_lower_case=False)
        self.sents = [g[0] for g in corpus]
        self.labels = [g[-1] for g in corpus]
        self.labelvocab = label_vocab

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent, tag_seq = self.sents[index], self.labels[index]

        sent = ['[CLS]'] + sent + ['[SEP]']
        try:
            tag_seq = [self.labelvocab.pad] + tag_seq + [self.labelvocab.pad]
        except TypeError:
            print(type(tag_seq))
            print(tag_seq)

        new_sent, real_tagseq, subword_start_masks, subword_masks = self.convert_tokens_to_subtokenids(sent, tag_seq)

        return torch.tensor(new_sent), torch.tensor(subword_start_masks, dtype=torch.uint8), torch.tensor(subword_masks, dtype=torch.uint8), torch.tensor(real_tagseq)

    def convert_tokens_to_subtokenids(self, tokens, labels):
        bert_sent, subword_start_masks, real_tag_seq, subword_masks = [], [], [], []

        for w, t in zip(tokens, labels):
            if w in ('[CLS]', '[SEP]'):
                sub_tokens = [w]
            else:
                sub_tokens = self.tokenizer.tokenize(w)
                if len(sub_tokens) == 0:
                    sub_tokens = ['[UNK]']
            token_idx = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            bert_sent += token_idx

            # subword mask
            if w in ('[CLS]', '[SEP]'):
                subword_masks += [0]
            else:
                subword_masks += [1]*len(sub_tokens)
            # subword head mask
            if w in ('[CLS]', '[SEP]'):
                subword_start_masks += [0]
            else:
                subword_start_masks += [1] + [0]*(len(sub_tokens)-1)
            # real tag seq
            if w not in ('[CLS]', '[SEP]'):
                real_tag_seq.append(self.labelvocab.get_id(t))
        assert len(bert_sent) == len(subword_start_masks)
        return bert_sent, real_tag_seq, subword_start_masks, subword_masks


class BertDataSet_For_partial(data.Dataset):
    def __init__(self, bert_vocab_path, corpus, label_vocab):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_vocab_path, do_lower_case=False)
        self.sents = [g[0] for g in corpus]
        self.labels = [g[-1] for g in corpus]
        self.labelvocab = label_vocab

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent, tag_seq = self.sents[index], self.labels[index]

        sent = ['[CLS]'] + sent + ['[SEP]']

        new_sent, subword_start_masks, subword_masks = self.convert_tokens_to_subtokenids(sent)

        real_tagseq = []
        for label in tag_seq:
            if label == ['UNK']:
                label_mask = [1 for _ in range(self.labelvocab.get_size())]
                label_mask.extend([0, 0])
            else:
                label_mask = [0 for _ in range(self.labelvocab.get_size()+2)]
                for possible_label in label:
                    label_mask[self.labelvocab.get_id(possible_label)] = 1
            real_tagseq.append(label_mask)
        sentence_mask = [1 for _ in range(len(tag_seq))]
        return torch.tensor(new_sent), torch.tensor(subword_start_masks, dtype=torch.uint8), \
            torch.tensor(subword_masks, dtype=torch.uint8), torch.tensor(real_tagseq), torch.tensor(sentence_mask)

    def convert_tokens_to_subtokenids(self, tokens):
        bert_sent, subword_start_masks, subword_masks = [], [], []

        for w in tokens:
            if w in ('[CLS]', '[SEP]'):
                sub_tokens = [w]
            else:
                sub_tokens = self.tokenizer.tokenize(w)
                if len(sub_tokens) == 0:
                    sub_tokens = ['[UNK]']
            token_idx = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            bert_sent += token_idx

            # subword mask
            if w in ('[CLS]', '[SEP]'):
                subword_masks += [0]
            else:
                subword_masks += [1]*len(sub_tokens)
            # subword head mask
            if w in ('[CLS]', '[SEP]'):
                subword_start_masks += [0]
            else:
                subword_start_masks += [1] + [0]*(len(sub_tokens)-1)
        assert len(bert_sent) == len(subword_start_masks)
        return bert_sent, subword_start_masks, subword_masks


class BertDataSet_for_unlabel(data.Dataset):
    def __init__(self, bert_vocab_path, corpus, label_vocab):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_vocab_path,
                                                       do_lower_case=False)
        self.sents = [g[0] for g in corpus]
        self.labels = [g[-1] for g in corpus]
        self.labelvocab = label_vocab

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        raw_sent, tag_seq = self.sents[index], self.labels[index]

        sent = ['[CLS]'] + raw_sent + ['[SEP]']
        tag_seq = [self.labelvocab.pad] + tag_seq + [self.labelvocab.pad]

        new_sent, real_tagseq, subword_start_masks, subword_masks = self.convert_tokens_to_subtokenids(sent, tag_seq)

        return torch.tensor(new_sent), torch.tensor(subword_start_masks, dtype=torch.uint8), \
            torch.tensor(subword_masks, dtype=torch.uint8), raw_sent, torch.tensor(real_tagseq)

    def convert_tokens_to_subtokenids(self, tokens, labels):
        bert_sent, subword_start_masks, real_tag_seq, subword_masks = [], [], [], []

        for w, t in zip(tokens, labels):
            if w in ('[CLS]', '[SEP]'):
                sub_tokens = [w]
            else:
                sub_tokens = self.tokenizer.tokenize(w)
                if len(sub_tokens) == 0:
                    sub_tokens = ['[UNK]']
            token_idx = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            bert_sent += token_idx

            # subword mask
            if w in ('[CLS]', '[SEP]'):
                subword_masks += [0]
            else:
                subword_masks += [1] * len(sub_tokens)
            # subword head mask
            if w in ('[CLS]', '[SEP]'):
                subword_start_masks += [0]
            else:
                subword_start_masks += [1] + [0] * (len(sub_tokens) - 1)
            # real tag seq
            if w not in ('[CLS]', '[SEP]'):
                real_tag_seq.append(self.labelvocab.get_id(t))
        assert len(bert_sent) == len(subword_start_masks)
        return bert_sent, real_tag_seq, subword_start_masks, subword_masks


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def judge_ignore(word):
    if len(_clean_text(word)) == 0:
        return True
    for char in word:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            return True
    return False

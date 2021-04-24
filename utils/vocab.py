# 词表模块，负责字词和id的互相映射
try:
    import cPickle as pickle
except ImportError:
    import pickle
from utils.utils import *


class Vocab():
    def __init__(self, data, islabel=False, ischar=False, isFeature=False, feature_id=-1, freq=1):
        self.instances = []
        self.instance2id = {}
        self.islabel = islabel
        self.ischar = ischar
        self.isfeature = isFeature

        self.pad = '</pad>'
        self.unk = '</unk>'
        self.alphabet = dict()
        self.freq = freq
        self.instance2id[self.pad] = 0
        self.instances.append(self.pad)
        if not islabel:
            self.instance2id[self.unk] = 1
            self.instances.append(self.unk)
            self.begin_index = 2
        else:
            self.begin_index = 1

        self.feature_id = feature_id
        self.build(data)

    def add(self, instance):
        if instance not in self.instance2id:
            if instance not in self.alphabet:
                self.alphabet[instance] = 1
            else:
                self.alphabet[instance] += 1
            if self.alphabet[instance] >= self.freq:
                self.instance2id[instance] = len(self.instances)
                self.instances.append(instance)
                self.begin_index += 1

    def get_id(self, instance):
        if self.islabel:
            if instance in self.instance2id:
                return self.instance2id[instance]
            else:
                raise RuntimeError("Vocab ERROR: unknown label..exit...")
        else:
            if instance in self.instance2id:
                return self.instance2id[instance]
            else:
                return self.instance2id[self.unk]

    def get_instance(self, id):
        if self.islabel:
            if id < len(self.instances):
                return self.instances[id]
            else:
                raise RuntimeError("Vocab ERROR: unknown label id..exit...")
        else:
            if id < len(self.instances):
                return self.instances[id]
            else:
                return self.unk

    def get_size(self):
        return len(self.instances)

    def build(self, samples):
        for sample in samples:
            if self.islabel:
                for s in sample[-1]:
                    self.add(s)
            elif self.ischar:
                for s in sample[1]:
                    for c in s:
                        self.add(c)
            elif self.isfeature:
                for s in sample[2]:
                    self.add(s[self.feature_id])
            else:
                for s in sample[0]:
                    self.add(s)

    def save(self, filepath):
        with open(filepath, 'wb') as fpout:
            pickle.dump(self, fpout)

    def add_embedding_file(self,file,embedding_dim):
        print("Loading pretrained embedding from %s" % file)
        embed_dict = dict()
        with open(file,'r',encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip().split()
                word = line[0]
                embed_dict[word] = ' '.join(line[1:])
        for word in embed_dict.keys():
            self.add(word)


class Label_Vocab_for_partialcrf():
    def __init__(self, data, freq=1):
        self.instances = []
        self.instance2id = {}
        self.islabel = True
        self.freq = freq

        self.pad = '</pad>'
        self.alphabet = dict()
        self.instance2id[self.pad] = 0
        self.instances.append(self.pad)
        self.begin_index = 1

        self.build(data)

    def add(self, instance):
        if instance not in self.instance2id:
            if instance not in self.alphabet:
                self.alphabet[instance] = 1
            else:
                self.alphabet[instance] += 1
            if self.alphabet[instance] >= self.freq:
                self.instance2id[instance] = len(self.instances)
                self.instances.append(instance)
                self.begin_index += 1

    def get_id(self, instance):
        if instance in self.instance2id:
            return self.instance2id[instance]
        else:
            raise RuntimeError("Vocab ERROR: unknown label..exit...")

    def get_instance(self, id):
        if id < len(self.instances):
            return self.instances[id]
        else:
            raise RuntimeError("Vocab ERROR: unknown label id..exit...")

    def get_size(self):
        return len(self.instances)

    def build(self, samples):
        for sample in samples:
            for s in sample[-1]:
                if type(s) == list:
                    if s == ['UNK']: continue
                    for s_sub in s:
                        self.add(s_sub)
                else:
                    self.add(s)

    def save(self, filepath):
        with open(filepath, 'wb') as fpout:
            pickle.dump(self, fpout)


class Vocab_for_Biaffine():
    def __init__(self, data, islabel=False, ischar=False, isFeature=False, feature_id=-1, freq=1):
        self.instances = []
        self.instance2id = {}
        self.islabel = islabel
        self.ischar = ischar
        self.isfeature = isFeature

        self.pad = '</pad>'
        self.unk = '</unk>'
        self.alphabet = dict()
        self.freq = freq
        self.instance2id[self.pad] = 0
        self.instances.append(self.pad)
        if not islabel:
            self.instance2id[self.unk] = 1
            self.instances.append(self.unk)
            self.begin_index = 2
        else:
            self.begin_index = 1

        self.feature_id = feature_id
        self.build(data)

    def add(self, instance):
        if instance not in self.instance2id:
            if instance not in self.alphabet:
                self.alphabet[instance] = 1
            else:
                self.alphabet[instance] += 1
            if self.alphabet[instance] >= self.freq:
                self.instance2id[instance] = len(self.instances)
                self.instances.append(instance)
                self.begin_index += 1

    def get_id(self, instance):
        if self.islabel:
            if instance in self.instance2id:
                return self.instance2id[instance]
            else:
                raise RuntimeError("Vocab ERROR: unknown label..exit...")
        else:
            if instance in self.instance2id:
                return self.instance2id[instance]
            else:
                return self.instance2id[self.unk]

    def get_instance(self, id):
        if self.islabel:
            if id < len(self.instances):
                return self.instances[id]
            else:
                raise RuntimeError("Vocab ERROR: unknown label id..exit...")
        else:
            if id < len(self.instances):
                return self.instances[id]
            else:
                return self.unk

    def get_size(self):
        return len(self.instances)

    def build(self, samples):
        for sample in samples:
            if self.islabel:
                label_matrix = sample[-1]
                sent_len = len(label_matrix[0])
                for i in range(sent_len):
                    for j in range(i, sent_len):
                        self.add(label_matrix[i][j])
            elif self.ischar:
                for s in sample[1]:
                    for c in s:
                        self.add(c)
            elif self.isfeature:
                for s in sample[2]:
                    self.add(s[self.feature_id])
            else:
                for s in sample[0]:
                    self.add(s)

    def save(self, filepath):
        with open(filepath, 'wb') as fpout:
            pickle.dump(self, fpout)

    def add_embedding_file(self,file,embedding_dim):
        print("Loading pretrained embedding from %s" % file)
        embed_dict = dict()
        with open(file,'r',encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip().split()
                word = line[0]
                embed_dict[word] = ' '.join(line[1:])
        for word in embed_dict.keys():
            self.add(word)
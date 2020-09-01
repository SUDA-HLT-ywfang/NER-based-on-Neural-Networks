# 文件、字符串的解码，接口将在NER系统中被直接调用
import torch
from model.bilstm_crf import BiLSTM_CRF
from model.bilstm_partialcrf import BiLSTM_PartialCRF
from model.fine_tune_tagger import FineTuneTagger
from model.feature_based_tagger import FeatureBasedTagger
from .bert_dataset import BertDataSet
from .corpus import Corpus
from .utils import *
import torch.utils.data as Data
from .bert_dataset import BertDataSet
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Decoder():
    # 解码器初始化，主要包括模型加载
    def __init__(self, save_model_path, batch_size, if_decode_on_gpu, data_path, model_type):
        self.modelpath = save_model_path
        self.batchsize = batch_size
        self.gpu_available = if_decode_on_gpu
        self.modeltype = model_type.lower()
        print("        Model Type: %s" % model_type)
        print("Loading Model Info from %s..." % data_path)
        print("Loading Model Parameters from %s..." % save_model_path)
        print("Decode Batch Size: %d" % self.batchsize)
        if model_type.lower() == 'bilstm-crf' or model_type.lower() == 'bilstm-partial-crf':
            self.model_info = self.load_modelinfo(data_path)
            self.word_vocab = self.model_info.word_vocab
            self.char_vocab = self.model_info.char_vocab
            self.label_vocab = self.model_info.label_vocab
            self.feature_config = self.model_info.feature_config
            self.model_info.gpu = if_decode_on_gpu
            self.model_info.pretrain_word_emb = None
        elif model_type.lower() == 'bert-finetune' or model_type.lower() == 'bert-featurebased':
            self.model_info = self.load_modelinfo(data_path)
            self.label_vocab = self.model_info['label_vocab']
        else:
            print("Unknown Model Type...Exit...")
            exit(1)
        self.load_model()
        self.network.eval()

    # 词典加载
    def load_modelinfo(self, filepath):
        with open(filepath, 'rb') as fpin:
            return pickle.load(fpin)

    # 网络重构
    def network_reconstruct(self):
        if self.modeltype == 'bilstm-crf':
            self.network = BiLSTM_CRF(self.model_info)
        elif self.modeltype == 'bilstm-partial-crf':
            self.network = BiLSTM_PartialCRF(self.model_info)
        elif self.modeltype == 'bert-finetune':
            self.network = FineTuneTagger(bert_path=self.model_info['bert_path'],
                                          bert_dim=self.model_info['bert_dim'],
                                          tag_size=self.model_info['tag_size'])
        elif self.modeltype == 'bert-featurebased':
            self.network = FeatureBasedTagger(bert_path=self.model_info['bert_path'],
                                              bert_dim=self.model_info['bert_dim'],
                                              tag_size=self.model_info['tag_size'],
                                              bert_layer=self.model_info['bert_layer'],
                                              use_crf=self.model_info['use_crf'],
                                              strategy=self.model_info['strategy'],
                                              word_emb_path=self.model_info['word_emb_path'],
                                              emb_dim=self.model_info['word_emb_dim'],
                                              use_partial_crf=self.model_info['use_partial_crf'],
                                              label_vocab=self.label_vocab)

    # 模型加载
    def load_model(self):
        if self.gpu_available:
            # save on GPU, load on GPU
            device = torch.device("cuda")
            self.network_reconstruct()
            self.network.load_state_dict(torch.load(self.modelpath))
            self.network.to(device)
        else:
            # save on GPU, load on CPU
            device = torch.device("cpu")
            self.network_reconstruct()
            self.network.load_state_dict(torch.load(self.modelpath, map_location=device))

    def recover_label_from_id(self, id_list, mask):
        instance_list = []
        sent_num = len(id_list)
        for idx in range(sent_num):
            temp = []
            for idy in range(len(id_list[idx])):
                if mask[idx][idy] != 0:
                    temp.append(self.label_vocab.get_instance(id_list[idx][idy]))
                else:
                    break
            instance_list.append(temp)
        return instance_list

    # 文本标注，输出写入文件
    def decode_file_writeoutput(self, rawtext_dir, output_dir):
        raw_samples = Corpus(rawtext_dir, number_normal=True)
        if 'bert' in self.modeltype:
            raw_data = BertDataSet(bert_vocab_path=self.model_info['bert_path'],
                                   corpus=raw_samples.samples,
                                   label_vocab=self.label_vocab)
        else:
            raw_data = instance2tensor(raw_samples.samples, self.word_vocab, self.char_vocab,
                                       self.label_vocab, self.feature_config)
        raw_loader = Data.DataLoader(
            dataset=raw_data,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=collate_fn if not self.gpu_available else collate_fn_cuda
        )
        tagresult = []
        print("Decode File Instances Size: %d" % len(raw_loader.dataset))
        for batch in raw_loader:
            mask, tag_result_seq = self.network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            tagresult += tag_list_batch
        with open(output_dir, 'w') as fpout:
            sent_num = 0
            for word_instances in raw_samples.raw_samples:
                tag_seq = tagresult[sent_num]
                while len(word_instances) != len(tag_seq):
                    sent_num += 1
                    tag_seq += tagresult[sent_num]
                assert len(word_instances) == len(tag_seq)
                for i in range(len(word_instances)):
                    fpout.write(word_instances[i] + '\t' + tag_seq[i] + '\n')
                fpout.write('\n')
                sent_num += 1

    # 文本标注，返回标注结果
    def decode_file(self, rawtext_dir):
        raw_samples = Corpus(rawtext_dir, number_normal=True)
        if 'bert' in self.modeltype:
            raw_data = BertDataSet(bert_vocab_path=self.model_info['bert_path'],
                                   corpus=raw_samples,
                                   label_vocab=self.label_vocab)
        else:
            raw_data = instance2tensor(raw_samples.samples, self.word_vocab, self.char_vocab, self.label_vocab)
        raw_loader = Data.DataLoader(
            dataset=raw_data,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=collate_fn if not self.gpu_available else collate_fn_cuda
        )
        tagresult = []
        rawtagresult = []
        print("Decode File Instances Size: %d" % len(raw_loader.dataset))
        for batch in raw_loader:
            mask, tag_result_seq = self.network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            tagresult += tag_list_batch
        sent_num = 0
        for word_instances in raw_samples.raw_samples:
            tag_seq = tagresult[sent_num]
            while len(word_instances) != len(tag_seq):
                sent_num += 1
                tag_seq += tagresult[sent_num]
            assert len(word_instances) == len(tag_seq)
            rawtagresult.append(tag_seq)
            sent_num += 1
        return rawtagresult

    # 字符串标注
    def decode_string(self, decode_string):
        raw_samples = Corpus(decode_string, number_normal=True, isfile=False)
        if 'bert' in self.modeltype:
            raw_data = BertDataSet(bert_vocab_path=self.model_info['bert_path'],
                                   corpus=raw_samples,
                                   label_vocab=self.label_vocab)
        else:
            raw_data = instance2tensor(raw_samples.samples, self.word_vocab, self.char_vocab, self.label_vocab)
        raw_loader = Data.DataLoader(
            dataset=raw_data,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=collate_fn if not self.gpu_available else collate_fn_cuda
        )
        tagresult = []
        rawtagresult = []
        for batch in raw_loader:
            mask, tag_result_seq = self.network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            tagresult += tag_list_batch
        sent_num = 0
        for word_instances in raw_samples.raw_samples:
            tag_seq = tagresult[sent_num]
            while len(word_instances) != len(tag_seq):
                sent_num += 1
                tag_seq += tagresult[sent_num]
            assert len(word_instances) == len(tag_seq)
            rawtagresult.append(tag_seq)
            sent_num += 1
        return rawtagresult[0]

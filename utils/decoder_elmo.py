# 文件、字符串的解码，接口将在NER系统中被直接调用
import torch
from model.bilstm_crf_elmo import BiLSTM_CRF
from .corpus import Corpus
from .utils import *
import torch.utils.data as Data
from elmo_module.elmo import Embedder
from .evaluator import ner_measure
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Decoder():
    # 解码器初始化，主要包括模型加载
    def __init__(self, save_model_path, batch_size, gpu, data_path, elmo_model, save_decode_elmo=None):
        self.modelpath = save_model_path
        self.batchsize = batch_size
        self.gpu_available = True if gpu != '-1' else False
        print("Loading Model Info from %s..." % data_path)
        self.model_info = self.load_modelinfo(data_path)
        self.word_vocab = self.model_info.word_vocab
        self.char_vocab = self.model_info.char_vocab
        self.label_vocab = self.model_info.label_vocab
        self.gpu = gpu
        self.model_info.pretrain_word_emb = None
        self.model_info.gpu = self.gpu_available
        self.save_decode_elmo = save_decode_elmo
        self.embedder = elmo_model
        print("Loading Model Parameters from %s..." % save_model_path)
        # print("Decode Batch Size: %d" % self.batchsize)
        self.load_model()
        self.network.eval()

    # 词典加载
    def load_modelinfo(self, filepath):
        with open(filepath, 'rb') as fpin:
            return pickle.load(fpin)

    # 模型加载
    def load_model(self):
        if self.gpu_available:
            # save on GPU, load on GPU
            device = torch.device("cuda")
            self.network = BiLSTM_CRF(self.model_info)
            self.network.load_state_dict(torch.load(self.modelpath, map_location='cuda:'+str(self.gpu)))
            self.network.to(device)
        else:
            # save on GPU, load on CPU
            device = torch.device("cpu")
            self.network = BiLSTM_CRF(self.model_info)
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
        raw_samples = Corpus(rawtext_dir, do_lower=False, number_normal=True)
        raw_data = instance2tensor_with_elmo(raw_samples.samples, self.word_vocab,
                                             self.char_vocab, self.label_vocab, self.embedder, self.save_decode_elmo)
        raw_loader = Data.DataLoader(
            dataset=raw_data,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=collate_fn if not self.gpu_available else collate_fn_cuda
        )
        tagresult = []
        gold_all = []
        print("Decode File Instances Size: %d" % len(raw_loader.dataset))
        for batch in raw_loader:
            mask, tag_result_seq = self.network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            tagresult += tag_list_batch
            gold_list_batch = self.recover_label_from_id(batch[-1].tolist(), mask.tolist())
            gold_all += gold_list_batch
        # accuracy, precision, recall, f1, label_result = ner_measure(tagresult, gold_all, 'BIO')
        # print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (accuracy, precision, recall, f1))
        with open(output_dir, 'w') as fpout:
            sent_num = 0
            for word_instances in raw_samples.raw_samples:
                tag_seq = tagresult[sent_num]
                while len(word_instances) != len(tag_seq):
                    sent_num += 1
                    tag_seq += tagresult[sent_num]
                # assert len(word_instances) == len(tag_seq)
                for i in range(len(word_instances)):
                    fpout.write(word_instances[i] + '\t' + tag_seq[i] + '\n')
                fpout.write('\n')
                sent_num += 1

    # 文本标注，返回标注结果
    def decode_file(self, rawtext_dir):
        raw_samples = Corpus(rawtext_dir, number_normal=True)
        raw_data = instance2tensor_with_elmo(raw_samples.samples, self.word_vocab,
                                             self.char_vocab, self.label_vocab, self.embedder, self.save_decode_elmo)
        raw_loader = Data.DataLoader(
            dataset=raw_data,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=collate_fn if not self.gpu_available else collate_fn_cuda
        )
        tagresult = []
        rawtagresult = []
        gold_all = []
        print("Decode File Instances Size: %d" % len(raw_loader.dataset))
        for batch in raw_loader:
            mask, tag_result_seq = self.network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            tagresult += tag_list_batch
            gold_list_batch = self.recover_label_from_id(batch[-1].tolist(), mask.tolist())
            gold_all += gold_list_batch
        accuracy, precision, recall, f1, label_result = ner_measure(tagresult, gold_all, 'BIO')
        # print("acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (accuracy, precision, recall, f1))
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
        raw_data = instance2tensor_with_elmo(raw_samples.samples, self.word_vocab,
                                             self.char_vocab, self.label_vocab, self.embedder, None)
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

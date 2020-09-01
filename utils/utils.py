import torch
import numpy as np
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


def collate_fn_cuda(data):
    batch = zip(*data)
    return tuple([torch.tensor(x).cuda() if len(x[0].size()) < 1 else pad_sequence(x, True).cuda() for x in batch])


# 加载预训练词向量
def load_pretrain_embedding(pretrain_emb_filepath, vocab, emb_dim):
    print("Loading pretrained embedding from %s" % pretrain_emb_filepath)
    emb = np.zeros((vocab.get_size(), emb_dim))
    embedding_dict = dict()
    with open(pretrain_emb_filepath, 'r',encoding='utf-8') as fpin:
        for line in fpin:
            line = line.rstrip().split()
            word = line[0]
            embedding_dict[word] = ' '.join(line[1:])
    scale = np.sqrt(3.0 / emb_dim)
    match, oov = 0, 0
    for word in vocab.instance2id:
        if word in embedding_dict:
            temp = np.fromstring(embedding_dict[word],dtype=float,sep=' ')
            emb[vocab.get_id(word)] = temp
            match += 1
        elif word.lower() in embedding_dict:
            temp = np.fromstring(embedding_dict[word.lower()],dtype=float,sep=' ')
            emb[vocab.get_id(word)] = temp
            match += 1
        else:
            emb[vocab.get_id(word), :] = np.random.uniform(-scale, scale, [1, emb_dim])
            oov += 1
    print("Embedding: all: %d, match: %d, oov: %d" % (len(embedding_dict), match, oov))
    return emb


# 随机初始化词向量
def random_init_embedding(vocab_size, embedding_dim):
    embedding = np.zeros((vocab_size, embedding_dim))
    scale = np.sqrt(3.0 / embedding_dim)
    for i in range(vocab_size):
        embedding[i] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return embedding


#实例转tensor
def instance2tensor(instances, word_vocab, char_vocab, label_vocab, feature_config=[], max_word_len=20):
    word_idxs,char_idxs,label_idxs,feature_idxs = [],[],[],[]
    for instance in instances:
        word_idx, char_idx, label_idx, feature_idx = [], [],[], []
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append([char_vocab.get_id(i) for i in ins[:max_word_len]]+[0 for i in range(max_word_len-len(ins))])
        for ins in instance[2]:
            feature_idx.append([feature_config[i]['vocab'].get_id(ins[i]) for i in range(len(feature_config))])
        for ins in instance[-1]:
            label_idx.append(label_vocab.get_id(ins))
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
        feature_idxs.append(torch.tensor(feature_idx))
    return TensorDataSet(word_idxs, char_idxs, feature_idxs, label_idxs)


# 带特征的实例转tensor
def instance_multi_label2tensor(instances, word_vocab, char_vocab, label_vocab, feature_config=[], max_word_len=20):
    word_idxs,char_idxs,label_idxs,feature_idxs = [],[],[],[]
    for instance in instances:
        word_idx, char_idx, label_idx, feature_idx = [], [],[], []
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append([char_vocab.get_id(i) for i in ins[:max_word_len]]+[0 for i in range(max_word_len-len(ins))])
        for ins in instance[2]:
            feature_idx.append([feature_config[i]['vocab'].get_id(ins[i]) for i in range(len(feature_config))])
        for ins in instance[-1]:
            if ins == ['UNK']:
                label_mask = [1 for _ in range(label_vocab.get_size())]
                label_mask.extend([0, 0])
            else:
                label_mask = [0 for _ in range(label_vocab.get_size()+2)]
                for possible_label in ins:
                    label_mask[label_vocab.get_id(possible_label)] = 1
            label_idx.append(label_mask)
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
        feature_idxs.append(torch.tensor(feature_idx))
    return TensorDataSet(word_idxs, char_idxs, feature_idxs, label_idxs)


def instance2tensor_with_elmo(instances, word_vocab, char_vocab, label_vocab, embedder, elmo_save_path=None, max_word_len=20):
    word_idxs, char_idxs, label_idxs = [], [], []
    for instance in instances:
        word_idx, char_idx, label_idx = [], [], []
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append(
                [char_vocab.get_id(i) for i in ins[:max_word_len]] + [0 for i in range(max_word_len - len(ins))])
        for ins in instance[-1]:
            label_idx.append(label_vocab.get_id(ins))
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
    # get biLSTM language model embedding represent
    sentence = [ins[0] for ins in instances]
    #print("sentence length:", len(sentence))
    if elmo_save_path is None:  # not save elmo feats,too big
        # extract all 3 layers
        elmo_rep = [torch.from_numpy(r).permute(1,0,2) for r in embedder.sents2elmo(sentence,output_layer=-2)]
        # for an average of 3 layers
        # elmo_rep = [torch.from_numpy(r) for r in embedder.sents2elmo(sentence, output_layer=-1)]
    else:  # small test data can be saved for decoded
        if os.path.exists(elmo_save_path):
            elmo_rep = pickle.load(open(elmo_save_path, 'rb'))
        else:
            #elmo_rep = [torch.from_numpy(r).permute(1,0,2) for r in embedder.sents2elmo(sentence,output_layer=-2)]
            elmo_rep = [torch.from_numpy(r) for r in embedder.sents2elmo(sentence, output_layer=-1)]
            pickle.dump(elmo_rep, open(elmo_save_path, 'wb'))
    return TensorDataSet(word_idxs, char_idxs, elmo_rep, label_idxs)


# 混合两条tag序列（模型输出）
def compare_tag_list(ori, new):
    assert len(ori) == len(new)
    all_entity, entity, tag_list = [], [], ['O'] * len(ori)
    for i, e in enumerate(ori):
        if 'B-' in e:
            if entity:
                entity.append(ori[i - 1].split('-')[1])
                all_entity.append(entity)
            entity = [i]
        elif 'I-' in e:
            entity.append(i)
        elif 'O' in e:
            if entity:
                entity.append(ori[i - 1].split('-')[1])
                all_entity.append(entity)
                entity = []
    if entity:
        entity.append(ori[-1].split('-')[1])
        all_entity.append(entity)
        entity = []
    for i, e in enumerate(new):
        if 'B-' in e:
            if entity:
                entity.append(new[i - 1].split('-')[1])
                all_entity.append(entity)
            entity = [i]
        elif 'I-' in e:
            entity.append(i)
        elif 'O' in e:
            if entity:
                entity.append(new[i - 1].split('-')[1])
                all_entity.append(entity)
                entity = []
    if entity:
        entity.append(new[-1].split('-')[1])
        all_entity.append(entity)
        entity = []
    result, flag_list = [], [False] * len(all_entity)
    for i in range(len(all_entity)):
        flag = False
        if flag_list[i] is False:
            for j in range(i + 1, len(all_entity)):
                set_i, set_j = get_set(all_entity[i]), get_set(all_entity[j])
                if set_i.intersection(set_j):
                    flag = True
                    # print(set_i,set_j)
                    if len(set_i) > len(set_j):
                        result.append(all_entity[i])
                        flag_list[j] = True
                    else:
                        result.append(all_entity[j])
            if flag is False:
                result.append(all_entity[i])
    for r in result:
        for i, e in enumerate(r[:-1]):
            if i == 0:
                tag_list[e] = 'B-' + r[-1]
            else:
                tag_list[e] = 'I-' + r[-1]
    return tag_list


def get_set(entity):
    return {i for i in entity[:-1]}


#实例转tensor
def instance2tensor_withdomainlabel(source_instances,target_instances,word_vocab,char_vocab,label_vocab,max_word_len=20):
    word_idxs,char_idxs,label_idxs,domain_idxs = [],[],[],[]
    for instance in source_instances:
        word_idx, char_idx, label_idx = [], [],[]
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append([char_vocab.get_id(i) for i in ins[:max_word_len]]+[0 for i in range(max_word_len-len(ins))])
        for ins in instance[-1]:
            label_idx.append(label_vocab.get_id(ins))
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
        domain_idxs.append(torch.tensor(0))
    for instance in target_instances:
        word_idx, char_idx, label_idx = [], [], []
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append(
                [char_vocab.get_id(i) for i in ins[:max_word_len]] + [0 for i in range(max_word_len - len(ins))])
        for ins in instance[-1]:
            label_idx.append(label_vocab.get_id(ins))
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
        domain_idxs.append(torch.tensor(1))
    return TensorDataSet(word_idxs,char_idxs,label_idxs,domain_idxs)


# 带特征的实例转tensor+ELMo
def instance_multi_label2tensor_elmo(instances, word_vocab, char_vocab, label_vocab, embedder, feature_config=[], max_word_len=20, elmo_save_path=None):
    word_idxs,char_idxs,label_idxs,feature_idxs = [],[],[],[]
    for instance in instances:
        word_idx, char_idx, label_idx, feature_idx = [], [],[], []
        for ins in instance[0]:
            word_idx.append(word_vocab.get_id(ins))
        for ins in instance[1]:
            char_idx.append([char_vocab.get_id(i) for i in ins[:max_word_len]]+[0 for i in range(max_word_len-len(ins))])
        for ins in instance[-1]:
            if ins == ['UNK']:
                label_mask = [1 for _ in range(label_vocab.get_size())]
                label_mask.extend([0, 0])
            else:
                label_mask = [0 for _ in range(label_vocab.get_size()+2)]
                for possible_label in ins:
                    label_mask[label_vocab.get_id(possible_label)] = 1
            label_idx.append(label_mask)
        word_idxs.append(torch.tensor(word_idx))
        char_idxs.append(torch.tensor(char_idx))
        label_idxs.append(torch.tensor(label_idx))
        feature_idxs.append(torch.tensor(feature_idx))
    # get biLSTM language model embedding represent
    sentence = [ins[0] for ins in instances]
    #print("sentence length:", len(sentence))
    if elmo_save_path is None:  # not save elmo feats,too big
        # extract all 3 layers
        elmo_rep = [torch.from_numpy(r).permute(1,0,2) for r in embedder.sents2elmo(sentence,output_layer=-2)]
        # for an average of 3 layers
        # elmo_rep = [torch.from_numpy(r) for r in embedder.sents2elmo(sentence, output_layer=-1)]
    else:  # small test data can be saved for decoded
        if os.path.exists(elmo_save_path):
            elmo_rep = pickle.load(open(elmo_save_path, 'rb'))
        else:
            #elmo_rep = [torch.from_numpy(r).permute(1,0,2) for r in embedder.sents2elmo(sentence,output_layer=-2)]
            elmo_rep = [torch.from_numpy(r) for r in embedder.sents2elmo(sentence, output_layer=-1)]
            pickle.dump(elmo_rep, open(elmo_save_path, 'wb'))
    return TensorDataSet(word_idxs, char_idxs, elmo_rep, label_idxs)


def load_bio_file(filepath):
    sents = []
    sent_char = []
    sent_tag = []
    with open(filepath,'r', encoding='utf8') as fpin:
        for line in fpin:
            line = line.rstrip()
            if line == '':
                if sent_char != []:
                    sents.append([sent_char, sent_tag])
                sent_char = []
                sent_tag = []
                continue
            content = line.split('\t')
            if len(content) == 2:
                sent_char.append(content[0])
                sent_tag.append(content[1])
            else:
                print("exception:%s###" % line)
    return sents


def write_bio2file(biodata, filepath):
    with open(filepath, 'w') as fpout:
        for sent_seq, tag_seq in biodata:
            for i in range(len(sent_seq)):
                fpout.write(sent_seq[i] + '\t' + tag_seq[i] + '\n')
            fpout.write('\n')


def calulate_KL_Loss(feature1, feature2):
    # follow 2018EMNLP《Adaptive Semi-supervised Learning for Cross-domain Sentiment Classification》
    represent_mean1, _ = torch.max(feature1, dim=1)     # b_s*channel_out
    represent_mean2, _ = torch.max(feature2, dim=1)
    
    represent_mean1 = torch.mean(represent_mean1, dim=0) # channel*out
    represent_mean2 = torch.mean(represent_mean2, dim=0)

    norm1 = torch.sum(represent_mean1, dim=0, keepdim=True)
    norm2 = torch.sum(represent_mean2, dim=0, keepdim=True)
    g_s = represent_mean1/norm1
    g_t = represent_mean2/norm2
    
    kl_1 = torch.sum(g_s*torch.log(g_s/g_t), dim=0)
    kl_2 = torch.sum(g_t*torch.log(g_t/g_s), dim=0)

    kl_loss = kl_1 + kl_2

    return kl_loss

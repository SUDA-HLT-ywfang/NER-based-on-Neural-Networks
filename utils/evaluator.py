try:
    import cPickle as pickle
except ImportError:
    import pickle

# 评价模块
class Evaluator():
    def __init__(self, label_vocab, batch_size=64, classification=False):
        self.batchsize_wheneval = batch_size
        if not classification:
            self.label_vocab = label_vocab
            self.tag_style = self.get_tag_scheme()
            print("Using Tag Style: %s" % self.tag_style)

    def recover_label_from_id(self, id_list, mask):
        instance_list = []
        sent_num = len(id_list)
        for idx in range(sent_num):
            temp = []
            for idy in range(len(id_list[idx])):
                if mask[idx][idy] != 0:
                    # 用来处理输入不确定，有多个可能label: 不判断对错。触发场景：评价局部标注的train
                    if type(id_list[idx][idy]) == list:
                        temp.append('O')
                    else:
                        temp.append(self.label_vocab.get_instance(id_list[idx][idy]))
                else:
                    break
            instance_list.append(temp)
        return instance_list

    def evaluate(self, network, data):
        tag_result_all = []
        gold_all = []
        for batch in data:
            mask, tag_result_seq = network(batch)
            tag_list_batch = self.recover_label_from_id(tag_result_seq.tolist(), mask.tolist())
            gold_list_batch = self.recover_label_from_id(batch[-1].tolist(), mask.tolist())
            tag_result_all += tag_list_batch
            gold_all += gold_list_batch
        accuracy, precision, recall, f1, label_result = ner_measure(tag_result_all, gold_all, self.tag_style)
        return accuracy, precision, recall, f1

    def evaluate_classification(self, network, data):
        predict, real = [], []
        for batch in data:
            predict_class, real_class = network(batch)
            predict += predict_class.tolist()
            real += real_class.tolist()
        accuracy = classfication_accuracy(predict, real)
        return accuracy

    def get_tag_scheme(self):
        for inst in self.label_vocab.instances:
            if inst.startswith('S-') or inst.startswith('E-'):
                return 'BIOES'
        return 'BIO'


# 评价模块
class Evaluator_for_Biaffine():
    def __init__(self, label_vocab, batch_size=64):
        self.batchsize_wheneval = batch_size
        self.label_vocab = label_vocab

    def evaluate(self, network, data):
        non_entity_idx = self.label_vocab.get_id('O')
        padding_idx = self.label_vocab.get_id('</pad>')
        num_pred, num_gold, num_correct = 0, 0, 0
        for batch in data:
            label_chart_pred = network(batch)
            label_chart_gold = batch[-1]
            pred_mask = label_chart_pred.ne(non_entity_idx) & label_chart_pred.ne(padding_idx)
            gold_mask = label_chart_gold.ne(non_entity_idx) & label_chart_gold.ne(padding_idx)
            correct_mask = pred_mask & gold_mask
            num_pred += pred_mask.sum().item()
            num_gold += gold_mask.sum().item()
            num_correct += (label_chart_pred.eq(label_chart_gold) & correct_mask).sum().item()
        precision = num_correct / (num_pred + 1e-5)
        recall = num_correct / (num_gold + 1e-5)
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)
        return 0, precision, recall, f1


def classfication_accuracy(predict, gold):
    all_num = len(predict)
    right = 0
    for i in range(len(predict)):
        if predict[i] == gold[i]:
            right += 1
    acc = right / all_num
    return acc


def ner_measure(tag_list, gold_list, tag_style):
    assert len(tag_list) == len(gold_list)
    right_token_all, token_all = 0, 0
    right_num, predict_num, gold_num = 0, 0, 0
    measure4eachlabel = dict()
    for idx in range(len(tag_list)):
        assert len(tag_list[idx]) == len(gold_list[idx])
        # calculate token
        token_all += len(gold_list[idx])
        for idy in range(len(tag_list[idx])):
            if tag_list[idx][idy] == gold_list[idx][idy]:
                right_token_all += 1
        # calculate entity
        if tag_style == 'BIO':
            tag_entity_set = ner_chunk_BIO(tag_list[idx])
            gold_entity_set = ner_chunk_BIO(gold_list[idx])
        elif tag_style == 'BIOES':
            tag_entity_set = ner_chunk_BIOES(tag_list[idx])
            gold_entity_set = ner_chunk_BIOES(gold_list[idx])
        else:
            raise RuntimeError("Evaluator ERROR: wrong tag style")
        right_num += len(tag_entity_set.intersection(gold_entity_set))
        predict_num += len(tag_entity_set)
        gold_num += len(gold_entity_set)
        # calculater measure for each label
        for s in gold_entity_set:
            ind = s.find('[')
            if ind == -1:
                print("ERROR: Wrong gold matrix format")
                exit(1)
            type_label = s[1:ind]
            if type_label not in measure4eachlabel:
                measure4eachlabel[type_label] = dict()
                measure4eachlabel[type_label]['right'] = 0
                measure4eachlabel[type_label]['predict'] = 0
                measure4eachlabel[type_label]['gold'] = 0
            measure4eachlabel[type_label]['gold'] += 1
        for s in tag_entity_set.intersection(gold_entity_set):
            ind = s.find('[')
            if ind == -1:
                print("ERROR: Wrong gold matrix format")
                exit(1)
            type_label = s[1:ind]
            measure4eachlabel[type_label]['right'] += 1
        for s in tag_entity_set:
            ind = s.find('[')
            if ind == -1:
                print("ERROR: Wrong gold matrix format")
                exit(1)
            type_label = s[1:ind]
            if type_label not in measure4eachlabel:
                measure4eachlabel[type_label] = dict()
                measure4eachlabel[type_label]['right'] = 0
                measure4eachlabel[type_label]['predict'] = 0
                measure4eachlabel[type_label]['gold'] = 0
            measure4eachlabel[type_label]['predict'] += 1
    # calculate eval standard
    acc = right_token_all / token_all
    if right_num == 0:
        precision, recall, fscore = -1, -1, -1
    else:
        precision = right_num / predict_num
        recall = right_num / gold_num
        fscore = 2*precision*recall/(precision+recall)
    # fmeasure for each label
    label_level_result = dict()
    for label in measure4eachlabel:
        if measure4eachlabel[label]['right'] == 0:
            p, r, f = -1, -1, -1
        else:
            p = measure4eachlabel[label]['right'] / measure4eachlabel[label]['predict']
            r = measure4eachlabel[label]['right'] / measure4eachlabel[label]['gold']
            f = 2*p*r/(p+r)
        label_level_result[label] = dict()
        label_level_result[label]['p'] = p
        label_level_result[label]['r'] = r
        label_level_result[label]['f'] = f
    return acc, precision, recall, fscore, label_level_result


def ner_chunk_BIO(label_list):
    entity_set = set()
    current_tag, chunk_result = '', ''
    for i in range(0, len(label_list)):
        if label_list[i][0:2] == 'B-':
            if chunk_result != '':
                chunk_result += str(i-1)+']'
                entity_set.add(chunk_result)
            current_tag = label_list[i][2:]
            chunk_result = '#'+current_tag+'['+str(i)+','
        elif label_list[i][0:2] == 'I-':
            if label_list[i][2:] == current_tag:
                continue
            else:
                if chunk_result != '':
                    chunk_result += str(i-1)+']'
                    entity_set.add(chunk_result)
                chunk_result = ''
        elif label_list[i] == 'O':
            if chunk_result != '':
                chunk_result += str(i-1)+']'
                entity_set.add(chunk_result)
            chunk_result = ''
    if chunk_result != '':
        chunk_result += str(len(label_list)-1) + ']'
        entity_set.add(chunk_result)
    return entity_set


def ner_chunk_BIOES(label_list):
    entity_set = set()
    current_tag, chunk_result = '', ''
    for i in range(0, len(label_list)):
        if label_list[i][0:2] == 'B-':
            current_tag = label_list[i][2:]
            chunk_result = '#'+current_tag+'['+str(i)+','
        elif label_list[i][0:2] == 'I-' or label_list[i][0:2] == 'M-':
            if label_list[i][2:] == current_tag:
                continue
            else:
                chunk_result = ''
        elif label_list[i] == 'O':
            chunk_result = ''
            continue
        elif label_list[i][0:2] == 'E-':
            if chunk_result != '':
                chunk_result += str(i)+']'
                entity_set.add(chunk_result)
            chunk_result = ''
        elif label_list[i][0:2] == 'S-':
            current_tag = label_list[i][2:]
            chunk_result = '#'+current_tag+'['+str(i)+']'
            entity_set.add(chunk_result)
            chunk_result = ''
    return entity_set


if __name__ == '__main__':
    # BIO style test
    label_list = ['B-PER', 'I-PER', 'O', 'B-LOC', 'B-ORG', 'O', 'B-LOC', 'I-LOC','I-PER','O','I-PER','B-PER']
    print(ner_chunk_BIO(label_list))
    # eval file test
    tag_list, gold_list = [], []
    with open('../save/train-MSRA.test-MSRA.out', 'r') as fpin:
        temp_tag, temp_gold = [], []
        for line in fpin:
            line = line.rstrip()
            if line == '':
                tag_list.append(temp_tag)
                gold_list.append(temp_gold)
                temp_tag, temp_gold = [], []
                continue
            line = line.split('\t')
            temp_tag.append(line[3])
            temp_gold.append(line[1])
    a, p, r, f = ner_measure(tag_list, gold_list, 'BIO')
    print("acc: %.4f, p=%.4f, r=%.4f, f=%.4f"%(a, p, r, f))
    # BIOES style test
    label_list = ['B-PER', 'I-PER', 'O', 'S-LOC', 'S-ORG', 'O', 'B-LOC', 'I-LOC','I-PER','O','I-PER','B-PER','I-PER','E-PER']
    print(ner_chunk_BIOES(label_list))
    # eval file test
    tag_list, gold_list = [], []
    with open('../save/train-MSRA.test-MSRA.out.bmes', 'r') as fpin:
        temp_tag, temp_gold = [], []
        for line in fpin:
            line = line.rstrip()
            if line == '':
                tag_list.append(temp_tag)
                gold_list.append(temp_gold)
                temp_tag, temp_gold = [], []
                continue
            line = line.split('\t')
            temp_tag.append(line[3])
            temp_gold.append(line[1])
    a, p, r, f = ner_measure(tag_list, gold_list, 'BIOES')
    print("acc: %.4f, p=%.4f, r=%.4f, f=%.4f" % (a, p, r, f))
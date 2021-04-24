# 主要用于语料预处理，将文本数字化
class Corpus():
    def __init__(self, filepath, do_lower=False, number_normal=False, half_width=True, max_sen_len=250, isfile=True):
        self.n_line = 0
        self.n_sample = 0
        self.do_lower = do_lower
        self.number_normal = number_normal
        self.half_width = half_width
        self.max_sen_len = max_sen_len
        self.raw_samples = []
        if isfile:
            self.samples = self.read_file(filepath)
        else:
            self.samples = self.read_string(filepath)

    def read_file(self, filepath):
        samples, raw_instances, word_instances, label_instances, char_instances = [], [], [], [], []
        feature_instances = []
        count = 0
        with open(filepath, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith("-DOCSTART-"):
                    fin.readline()
                    continue
                if line != '\n':
                    info = line.rstrip().split('\t')
                    word = info[0]
                    label = info[-1]
                    feature4oneword = []
                    for i in range(1, len(info)-1):
                        feature4oneword.append(info[i].split(']')[-1])
                    raw_instances.append(word)
                    if self.do_lower:
                        word = self.__word_lower(word)
                    if self.number_normal:
                        word = self.__normalize_number(word)
                    # if self.half_width:
                    #     word = self.strQ2B(word)
                    # else:
                    #     word = self.strB2Q(word)
                    word_instances.append(word)
                    char_instances.append(list(word))
                    label_instances.append(label)
                    feature_instances.append(feature4oneword)
                else:
                    if len(word_instances) == 0:
                        continue
                    if len(word_instances) != len(label_instances):
                        raise RuntimeError('pls make sure have right instance.')
                    check_ins = self.__check_max_len([word_instances, char_instances, feature_instances, label_instances])
                    if check_ins != []:
                        samples.extend(check_ins)
                    self.raw_samples.append(raw_instances)
                    raw_instances, word_instances, label_instances, char_instances = [], [], [], []
                    feature_instances = []
                count += 1
        if len(word_instances) != 0:
            check_ins = self.__check_max_len([word_instances, char_instances, feature_instances, label_instances])
            if check_ins != []:
                samples.extend(check_ins)
            self.raw_samples.append(raw_instances)
        self.n_line = count
        self.n_sample = len(samples)
        return samples

    def read_string(self, decode_string):
        raw_instances, word_instances, char_instances, label_instances = [], [], [], []
        for word in decode_string:
            if self.do_lower:
                word = self.__word_lower(word)
            if self.number_normal:
                word = self.__normalize_number(word)
            # if self.half_width:
            #     word = self.strQ2B(word)
            # else:
            #     word = self.strB2Q(word)
            word_instances.append(word)
            char_instances.append(list(word))
            label_instances.append('O')
            raw_instances.append(word)
        chech_ins = self.__check_max_len([word_instances, char_instances, label_instances])
        self.raw_samples.append(raw_instances)
        return chech_ins

    def export_instance(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as fout:
            for sample in self.samples:
                word_instance = sample[0]
                label_instance = sample[-1]
                for idx, word in enumerate(word_instance):
                    fout.write(word + '\t' + label_instance[idx] + '\n')
                fout.write('\n')

    # word大小写
    def __word_lower(self, word):
        return word.lower()

    # 数字归零
    def __normalize_number(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    # 检查输入最大句长
    def __check_max_len(self, instance):
        word_instance = instance[0]
        char_instance = instance[1]
        label_instance = instance[-1]
        feature_instance = instance[2]
        if len(word_instance) > self.max_sen_len:
            new_instance = []
            left, right = 0, self.max_sen_len
            while right < len(word_instance):
                while label_instance[right] != 'O' and not label_instance[right].startswith('B') and right > left:
                    right -= 1
                # if this piece is full of entities, then abandon this sentence
                if right == left:
                    return []
                new_instance.append([word_instance[left:right], char_instance[left:right], feature_instance[left:right],
                                     label_instance[left:right]])
                left = right
                right += self.max_sen_len
            if left < len(word_instance) and right >= len(word_instance):
                new_instance.append([word_instance[left:], char_instance[left:], feature_instance[left:],
                                     label_instance[left:]])
            return new_instance
        else:
            return [instance]

    # 全角转半角
    def strQ2B(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    # 半角转全角
    def strB2Q(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 32:  # 半角空格直接转化
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
                inside_code += 65248
            rstring += chr(inside_code)
        return rstring


# 处理带有多种可能标签的语料
class Corpus_With_Multi_Label():
    def __init__(self, filepath, do_lower=False, number_normal=False, half_width=True, max_sen_len=250, isfile=True):
        self.n_line = 0
        self.n_sample = 0
        self.do_lower = do_lower
        self.number_normal = number_normal
        self.half_width = half_width
        self.max_sen_len = max_sen_len
        self.raw_samples = []
        if isfile:
            self.samples = self.read_file(filepath)
        else:
            self.samples = self.read_string(filepath)

    def read_file(self, filepath):
        samples, raw_instances, word_instances, label_instances, char_instances = [], [], [], [], []
        feature_instances = []
        count = 0
        with open(filepath, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith("-DOCSTART-"):
                    fin.readline()
                    continue
                if line != '\n':
                    info = line.rstrip().split('\t')
                    word = info[0]
                    label = info[-1].split(' ')
                    feature4oneword = []
                    for i in range(1, len(info) - 1):
                        feature4oneword.append(info[i].split(']')[-1])
                    raw_instances.append(word)
                    if self.do_lower:
                        word = self.__word_lower(word)
                    if self.number_normal:
                        word = self.__normalize_number(word)
                    # if self.half_width:
                    #     word = self.strQ2B(word)
                    # else:
                    #     word = self.strB2Q(word)
                    word_instances.append(word)
                    char_instances.append(list(word))
                    label_instances.append(label)
                    feature_instances.append(feature4oneword)
                else:
                    if len(word_instances) == 0:
                        continue
                    if len(word_instances) != len(label_instances):
                        raise RuntimeError('pls make sure have right instance.')
                    check_ins = self.__check_max_len(
                        [word_instances, char_instances, feature_instances, label_instances])
                    if check_ins != []:
                        samples.extend(check_ins)
                    self.raw_samples.append(raw_instances)
                    raw_instances, word_instances, label_instances, char_instances = [], [], [], []
                    feature_instances = []
                count += 1
        if len(word_instances) != 0:
            check_ins = self.__check_max_len([word_instances, char_instances, feature_instances, label_instances])
            if check_ins != []:
                samples.extend(check_ins)
            self.raw_samples.append(raw_instances)
        self.n_line = count
        self.n_sample = len(samples)
        return samples

    def read_string(self, decode_string):
        raw_instances, word_instances, char_instances, label_instances = [], [], [], []
        for word in decode_string:
            if self.do_lower:
                word = self.__word_lower(word)
            if self.number_normal:
                word = self.__normalize_number(word)
            # if self.half_width:
            #     word = self.strQ2B(word)
            # else:
            #     word = self.strB2Q(word)
            word_instances.append(word)
            char_instances.append(list(word))
            label_instances.append('O')
            raw_instances.append(word)
        chech_ins = self.__check_max_len([word_instances, char_instances, label_instances])
        self.raw_samples.append(raw_instances)
        return chech_ins

    def export_instance(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as fout:
            for sample in self.samples:
                word_instance = sample[0]
                label_instance = sample[-1]
                for idx, word in enumerate(word_instance):
                    fout.write(word + '\t' + label_instance[idx] + '\n')
                fout.write('\n')

    # word大小写
    def __word_lower(self, word):
        return word.lower()

    # 数字归零
    def __normalize_number(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    # 检查输入最大句长
    def __check_max_len(self, instance):
        word_instance = instance[0]
        char_instance = instance[1]
        label_instance = instance[-1]
        feature_instance = instance[2]
        if len(word_instance) > self.max_sen_len:
            new_instance = []
            left, right = 0, self.max_sen_len
            while right < len(word_instance):
                # while label_instance[right] != 'O' and not label_instance[right].startswith('B') and right > left:
                #     right -= 1
                # if this piece is full of entities, then abandon this sentence
                if right == left:
                    return []
                new_instance.append([word_instance[left:right], char_instance[left:right], feature_instance[left:right],
                                     label_instance[left:right]])
                left = right
                right += self.max_sen_len
            if left < len(word_instance) and right >= len(word_instance):
                new_instance.append([word_instance[left:], char_instance[left:], feature_instance[left:],
                                     label_instance[left:]])
            return new_instance
        else:
            return [instance]

    # 全角转半角
    def strQ2B(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    # 半角转全角
    def strB2Q(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 32:  # 半角空格直接转化
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
                inside_code += 65248
            rstring += chr(inside_code)
        return rstring


# 处理带有多种可能标签的语料
class Corpus_for_Biaffine():
    def __init__(self, filepath, do_lower=False, number_normal=False, half_width=True, max_sen_len=250, isfile=True):
        self.n_line = 0
        self.n_sample = 0
        self.do_lower = do_lower
        self.number_normal = number_normal
        self.half_width = half_width
        self.max_sen_len = max_sen_len
        self.raw_samples = []
        if isfile:
            self.samples = self.read_file(filepath)
        else:
            self.samples = self.read_string(filepath)

    def read_file(self, filepath):
        samples, raw_instances, word_instances, label_instances, char_instances = [], [], [], [], []
        feature_instances = []
        count = 0
        with open(filepath, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith("-DOCSTART-"):
                    fin.readline()
                    continue
                if line != '\n':
                    info = line.rstrip().split('\t')
                    word = info[0]
                    label = info[-1]
                    feature4oneword = []
                    for i in range(1, len(info) - 1):
                        feature4oneword.append(info[i].split(']')[-1])
                    raw_instances.append(word)
                    if self.do_lower:
                        word = self.__word_lower(word)
                    if self.number_normal:
                        word = self.__normalize_number(word)
                    # if self.half_width:
                    #     word = self.strQ2B(word)
                    # else:
                    #     word = self.strB2Q(word)
                    word_instances.append(word)
                    char_instances.append(list(word))
                    label_instances.append(label)
                    feature_instances.append(feature4oneword)
                else:
                    if len(word_instances) == 0:
                        continue
                    if len(word_instances) != len(label_instances):
                        raise RuntimeError('pls make sure have right instance.')
                    check_ins = self.__check_max_len([word_instances, char_instances, feature_instances, label_instances])
                    # turn label_instances to chunk format
                    for i in range(len(check_ins)):
                        instance_label = check_ins[i][-1]
                        instance_label_matrix = self.__chunk_label(instance_label)
                        check_ins[i][-1] = instance_label_matrix
                    if check_ins != []:
                        samples.extend(check_ins)
                    self.raw_samples.append(raw_instances)
                    raw_instances, word_instances, label_instances, char_instances = [], [], [], []
                    feature_instances = []
                count += 1
        if len(word_instances) != 0:
            check_ins = self.__check_max_len([word_instances, char_instances, feature_instances, label_instances])
            if check_ins != []:
                samples.extend(check_ins)
            self.raw_samples.append(raw_instances)
        self.n_line = count
        self.n_sample = len(samples)
        return samples

    def read_string(self, decode_string):
        raw_instances, word_instances, char_instances, label_instances = [], [], [], []
        for word in decode_string:
            if self.do_lower:
                word = self.__word_lower(word)
            if self.number_normal:
                word = self.__normalize_number(word)
            # if self.half_width:
            #     word = self.strQ2B(word)
            # else:
            #     word = self.strB2Q(word)
            word_instances.append(word)
            char_instances.append(list(word))
            label_instances.append('O')
            raw_instances.append(word)
        chech_ins = self.__check_max_len([word_instances, char_instances, label_instances])
        self.raw_samples.append(raw_instances)
        return chech_ins

    def export_instance(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as fout:
            for sample in self.samples:
                word_instance = sample[0]
                label_instance = sample[-1]
                for idx, word in enumerate(word_instance):
                    fout.write(word + '\t' + label_instance[idx] + '\n')
                fout.write('\n')

    def __chunk_label(self, label_list):
        # initialize label matrix, filled with "<pad>" and "O"
        label_matrix = []
        for i in range(len(label_list)):
            temp = []
            for j in range(len(label_list)):
                if j >= i: temp.append('O')
                else: temp.append('</pad>')
            label_matrix.append(temp)
        # assign
        start, end = -1, -1
        for i in range(len(label_list)):
            if label_list[i][0:2] == 'B-':
                if start == -1:
                    start = i
                else:
                    end = i-1
                    label_matrix[start][end] = label_list[end][2:]
                    start, end = -1, -1
            elif label_list[i][0:2] == 'I-':
                continue
            elif label_list[i][:2] == 'E-':
                end = i
                label_matrix[start][end] = label_list[end][2:]
                start, end = -1, -1
            elif label_list[i][:2] == 'S-':
                if start != -1:
                    end = i-1
                    label_matrix[start][end] = label_list[end][2:]
                    start, end = -1, -1
                label_matrix[i][i] = label_list[i][2:]
            elif label_list[i] == 'O':
                if start == -1:
                    continue
                else:
                    end = i-1
                    label_matrix[start][end] = label_list[end][2:]
                    start, end = -1, -1
        if start != -1:
            end = len(label_list)-1
            label_matrix[start][end] = label_list[-1][2:]
        return label_matrix
            

    # word大小写
    def __word_lower(self, word):
        return word.lower()

    # 数字归零
    def __normalize_number(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    # 检查输入最大句长
    def __check_max_len(self, instance):
        word_instance = instance[0]
        char_instance = instance[1]
        label_instance = instance[-1]
        feature_instance = instance[2]
        if len(word_instance) > self.max_sen_len:
            new_instance = []
            left, right = 0, self.max_sen_len
            while right < len(word_instance):
                # while label_instance[right] != 'O' and not label_instance[right].startswith('B') and right > left:
                #     right -= 1
                # if this piece is full of entities, then abandon this sentence
                if right == left:
                    return []
                new_instance.append([word_instance[left:right], char_instance[left:right], feature_instance[left:right],
                                     label_instance[left:right]])
                left = right
                right += self.max_sen_len
            if left < len(word_instance) and right >= len(word_instance):
                new_instance.append([word_instance[left:], char_instance[left:], feature_instance[left:],
                                     label_instance[left:]])
            return new_instance
        else:
            return [instance]

    # 全角转半角
    def strQ2B(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    # 半角转全角
    def strB2Q(self, str):
        rstring = ""
        for uchar in str:
            inside_code = ord(uchar)
            if inside_code == 32:  # 半角空格直接转化
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
                inside_code += 65248
            rstring += chr(inside_code)
        return rstring
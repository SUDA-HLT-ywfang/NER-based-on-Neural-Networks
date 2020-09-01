import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle


# 用于存储数据处理输出的数据，即模型输入的数据
class Input_Data():
    def __init__(self,word_emb_num,word_emb_dim,hidden_dim,dropout,lstm_layers,label_vocab,use_cuda,pretrain_wordemb,word_vocab,
                 char_vocab,batchsize,pretrain_charemb=None,use_char=False,char_model='CNN', feature_config=()):
        self.word_emb_num = word_emb_num
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.label_vocab = label_vocab
        self.label_size = label_vocab.get_size()
        self.gpu = use_cuda
        self.pretrain_word_emb = pretrain_wordemb
        self.word_vocab = word_vocab
        self.batchsize = batchsize

        # char embedding
        self.char_vocab = char_vocab
        self.pretrain_char_emb = pretrain_charemb
        # char encoder
        self.word_feature_extractor = 'BiLSTM'
        self.use_char = use_char
        self.char_feature_extractor = char_model
        self.char_emb_dim = 30
        self.char_hidden_dim = 50

        # extra feature
        self.feature_config = feature_config

        self.show_setup_all()

    def show_setup_all(self):
        print("=========System Set Up=========")
        print("Word feature extractor:", self.word_feature_extractor)
        print("      word    emb  dim:", self.word_emb_dim)
        print("      word  vocab size:", self.word_vocab.get_size())
        print("      word  hidden dim:", self.hidden_dim)
        print(" pretrain word emb dir:", self.pretrain_word_emb)
        if self.use_char:
            print("Char feature extractor:", self.char_feature_extractor)
            print("       char    emb dim:", self.char_emb_dim)
            print("       char hidden dim:", self.char_hidden_dim)
            print("       char vocab size:", self.char_vocab.get_size())
        if self.feature_config != None:
            print("Feature Num: %d" % len(self.feature_config))
            for i in range(len(self.feature_config)):
                print("      Feature %2d: %s" % (i, self.feature_config[i]['name']))
                print("      emb    dim: %d" % self.feature_config[i]['emb_dim'])
                print("      vocab size: %d" % self.feature_config[i]['vocab'].get_size())
        print("Label  vocab size:", self.label_size)
        print("       Batch size:", self.batchsize)
        print("================================")
        sys.stdout.flush()

    def save_model_info(self, filepath):
        with open(filepath, 'wb') as fpout:
            pickle.dump(self, fpout)

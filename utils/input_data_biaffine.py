import sys
import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle


# 用于存储数据处理输出的数据，即模型输入的数据
class Input_Data_Biaffine():
    def __init__(self, args, vocab_word, vocab_label, vocab_char=None, feature_config = []):
        self.convert_args_to_dict(args)
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.vocab_char = vocab_char
        self.feature_config = feature_config
        
        self.show_setup_all()

    def convert_args_to_dict(self, args):
        self.max_param_len = 0
        self.param_dict = vars(args)
        for arg in self.param_dict:
            if len(arg) > self.max_param_len:
                self.max_param_len = len(arg)

    def get(self, param):
        if param in self.param_dict:
            return self.param_dict[param]
        else:
            raise KeyError("The parameter %s you ask for is not defined" % param)

    def show_setup_all(self):
        print("=========System Set Up=========")
        for arg in self.param_dict:
            padding_part = "".join([" " for _ in range(self.max_param_len-len(arg))])
            print("%s%s: %s" % (arg, padding_part, self.param_dict[arg]))
        print("================================")
        sys.stdout.flush()

    def save_model_info(self, filepath):
        with open(filepath, 'wb') as fpout:
            pickle.dump(self, fpout)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='BiLSTM-CRF Model Training')
#     parser.add_argument('--train', required=True, help='filepath of train corpus')
#     parser.add_argument('--dev', required=True, help='filepath of dev corpus')
#     parser.add_argument('--test', required=True, help='filepath of test corpus')
#     parser.add_argument('--pretrain_emb', default=None, help='filepath of pretrained embedding')
#     parser.add_argument('--save_model_dir', default='save/best.baseline.model', help='path of saved network parameters')
#     parser.add_argument('--save_model_info_dir', default='save/bestmodel_info.model', help='path of saved info for decode')
#     parser.add_argument('--epoch_num', default=100, type=int, help='epoch num of model training')
#     parser.add_argument('--batch_size', default=10, type=int, help='batch size when training')
#     parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of lstm')
#     parser.add_argument('--optimizer', default='SGD', help='')
#     parser.add_argument('--gpu', type=str, default='-1', help='gpu id, set to -1 if use cpu mode')
#     parser.add_argument('--lstm_layers', default=3, type=int, help='lstm layers')
#     parser.add_argument('--word_emb_dim', default=100, type=int, help='')
#     parser.add_argument('--dropout_rate_lstm', default=0.4, type=float, help='')
#     parser.add_argument('--dropout_rate_ffnn', default=0.2, type=float)
#     parser.add_argument('--learning_rate', default=0.015, type=float, help='')
#     parser.add_argument('--lr_decay', default=0.05, type=float)
#     parser.add_argument('--patience', default=10, type=int, help='')
#     # char parameters
#     parser.add_argument('--emb_dim_char', default=30, type=int)
#     parser.add_argument('--hidden_dim_char', default=50, type=int)
#     parser.add_argument('--use_char', action="store_true")
#     parser.add_argument('--char_model', default='BiLSTM', type=str, help='char model for english ner', choices=["CNN","BiLSTM"])
#     # seed num
#     parser.add_argument('--seed', default=42, type=int, help='')
#     # cpu thread
#     parser.add_argument('--cpu_thread', default=6, type=int)

#     args = parser.parse_args()

#     data = Input_Data_Biaffine(args)
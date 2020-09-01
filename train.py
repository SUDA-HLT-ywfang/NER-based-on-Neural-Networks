# BiLSTM-CRF模型的训练
import argparse
from utils.corpus import Corpus
from utils.vocab import Vocab
from utils.input_data import Input_Data
from utils.utils import *
import torch.utils.data as Data
from model.bilstm_crf import *
from utils.trainer import *
from utils.evaluator import *
import random
import numpy as np
import os


def Print_Info(args):
    print("Train File: ", args.train)
    print("Dev   File: ", args.dev)
    print("Test  File: ", args.test)
    print("Batch Size: ", args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM-CRF Model Training')
    parser.add_argument('--train', required=True, help='filepath of train corpus')
    parser.add_argument('--dev', required=True, help='filepath of dev corpus')
    parser.add_argument('--test', required=True, help='filepath of test corpus')
    parser.add_argument('--pretrain_emb', default=None, help='filepath of pretrained embedding')
    parser.add_argument('--save_model_dir', default='save/best.baseline.model', help='path of saved network parameters')
    parser.add_argument('--save_model_info_dir', default='save/bestmodel_info.model', help='path of saved info for decode')
    parser.add_argument('--epoch_num', default=50, type=int, help='epoch num of model training')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size when training')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of lstm')
    parser.add_argument('--optimizer', default='SGD', help='')
    parser.add_argument('--gpu', type=str, default='-1', help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--lstm_layers', default=1, type=int, help='lstm layers')
    parser.add_argument('--word_emb_dim', default=100, type=int, help='')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='')
    parser.add_argument('--learning_rate', default=0.015, type=float, help='')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--patience', default=10, type=int, help='')
    # char parameters
    parser.add_argument('--use_char', default=False, type=bool, help='')
    parser.add_argument('--char_model', default='BiLSTM', type=str, help='')
    # seed num
    parser.add_argument('--seed', default=42, type=int, help='')
    # lookahead
    parser.add_argument('--use_lookahead', default=False, type=bool)
    # cpu thread
    parser.add_argument('--cpu_thread', default=6, type=int)
    args = parser.parse_args()

    seed_num = args.seed
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    if args.gpu != '-1':
        use_cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        use_cuda = False
    torch.set_num_threads(args.cpu_thread)

    train_samples = Corpus(args.train, do_lower=False, number_normal=True)
    dev_samples = Corpus(args.dev, do_lower=False, number_normal=True)
    test_samples = Corpus(args.test, do_lower=False, number_normal=True)

    word_vocab = Vocab(train_samples.samples+dev_samples.samples+test_samples.samples, islabel=False, freq=1)
    #word_vocab.add_embedding_file(args.pretrain_emb, embedding_dim=args.word_emb_dim)
    char_vocab = Vocab(train_samples.samples+dev_samples.samples+test_samples.samples, ischar=True, freq=1)
    label_vocab = Vocab(train_samples.samples+dev_samples.samples+test_samples.samples, islabel=True, freq=1)

    train_data = instance2tensor(train_samples.samples, word_vocab, char_vocab, label_vocab)
    dev_data = instance2tensor(dev_samples.samples, word_vocab, char_vocab, label_vocab)
    test_data = instance2tensor(test_samples.samples, word_vocab, char_vocab, label_vocab)

    data = Input_Data(
        word_vocab.get_size(),
        args.word_emb_dim,
        args.hidden_dim,
        args.dropout_rate,
        args.lstm_layers,
        label_vocab,
        use_cuda,
        args.pretrain_emb,
        word_vocab,
        char_vocab=char_vocab,
        use_char=args.use_char,
        char_model=args.char_model,
        batchsize=args.batch_size
    )
    Print_Info(args)
    # save input_data for decode
    data.save_model_info(args.save_model_info_dir)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    print("Building Model...")
    model = BiLSTM_CRF(data)
    if use_cuda:
        model.cuda()
    print(model)
    trainer = Trainer(model, args, train_loader)
    evaluator = Evaluator(label_vocab)
    trainer.train(train_loader, dev_loader, test_loader, evaluator)
    print("finish")

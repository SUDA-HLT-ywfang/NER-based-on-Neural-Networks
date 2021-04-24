# BiLSTM-CRF模型的训练
import argparse
import os
import random

import numpy as np
import torch
import torch.utils.data as Data

from model.bilstm_softmax import BiLSTM_MaxEnt
from utils.corpus import Corpus
from utils.evaluator import Evaluator
from utils.input_data_biaffine import Input_Data_Biaffine
from utils.trainer import Trainer
from utils.utils import instance2tensor, collate_fn_cuda, collate_fn
from utils.vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM-CRF Model Training')
    # files needed
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--emb_path_word', default=None)
    parser.add_argument('--save_model_dir', default='save/best.baseline.model')
    parser.add_argument('--save_model_info_dir', default='save/bestmodel_info.model')
    # model related
    parser.add_argument('--emb_dim_word', default=100, type=int)
    parser.add_argument('--lstm_layers', default=1, type=int)
    parser.add_argument('--hidden_dim_lstm', default=256, type=int)
    parser.add_argument('--dropout_embedding', default=0.5, type=float)
    parser.add_argument('--dropout_lstm', default=0.5, type=float, help='')
    # training related
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--gpu', type=str, default='-1')
    # char parameters
    parser.add_argument('--use_char', action="store_true")
    parser.add_argument('--emb_dim_char', default=30, type=int)
    parser.add_argument('--hidden_dim_char', default=50, type=int)
    parser.add_argument('--char_model', default='BiLSTM', type=str, choices=["CNN","BiLSTM"])
    # seed num
    parser.add_argument('--seed', default=42, type=int, help='')
    # cpu thread
    parser.add_argument('--cpu_thread', default=6, type=int)
    # avg batch loss
    parser.add_argument('--avg_batch_loss', action="store_true")

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

    word_vocab = Vocab(train_samples.samples+dev_samples.samples +
                       test_samples.samples, islabel=False, freq=1)
    #word_vocab.add_embedding_file(args.pretrain_emb, embedding_dim=args.word_emb_dim)
    char_vocab = Vocab(train_samples.samples+dev_samples.samples +
                       test_samples.samples, ischar=True, freq=1)
    label_vocab = Vocab(train_samples.samples+dev_samples.samples +
                        test_samples.samples, islabel=True, freq=1)
    
    train_data = instance2tensor(train_samples.samples, word_vocab, char_vocab, label_vocab)
    dev_data = instance2tensor(dev_samples.samples, word_vocab, char_vocab, label_vocab)
    test_data = instance2tensor(test_samples.samples, word_vocab, char_vocab, label_vocab)

    data = Input_Data_Biaffine(args, vocab_word=word_vocab, vocab_label=label_vocab, vocab_char=char_vocab)
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
    model = BiLSTM_MaxEnt(data)
    if use_cuda:
        model.cuda()
    print(model)
    trainer = Trainer(model, args, train_loader)
    evaluator = Evaluator(label_vocab)
    trainer.train(train_loader, dev_loader, test_loader, evaluator)
    print("finish")

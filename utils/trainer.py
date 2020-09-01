import time
import torch.optim as optim
import torch
import sys
from utils.lookahead import Lookahead
from pytorch_transformers import AdamW, WarmupLinearSchedule
import torch.utils.data as Data
from utils.utils import *

class Trainer():
    def __init__(self, network, config, trainloader):
        self.network = network
        self.config = config
        self.optimizer_name = config.optimizer.lower()
        if self.optimizer_name == 'sgd':
            print("Using SGD optimizer...learning rate set as %f" % config.learning_rate)
            self.optimizer = optim.SGD(network.parameters(), lr=config.learning_rate)
        elif self.optimizer_name == 'adam':
            print("Using Adam optimizer...learning rate set as %f" % config.learning_rate)
            self.optimizer = optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=1e-08)
        elif self.optimizer_name == 'adamw':
            print("Using AdamW optimizer...learning rate set %f" % config.learning_rate)
            print(len(trainloader))
            num_total_steps = len(trainloader) * config.epoch_num
            num_warmup_steps = int(num_total_steps*0.1)
            print("Total Steps: %d, WarmUp Steps: %d"%(num_total_steps, num_warmup_steps))
            self.optimizer = AdamW(network.parameters(), lr=config.learning_rate, correct_bias=False)
            self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)
        elif self.optimizer_name == 'rmsprop':
            print("Using RMSprop optimizer...learning rate set %f" % config.learning_rate)
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        else:
            print("Wrong optimizer, please choose among SGD/Adam/BertAdam")
        if config.use_lookahead:
            print("Using Lookahead to optimize...")
            self.optimizer = Lookahead(optimizer=self.optimizer, k=5, alpha=0.5)
        self.epoch_num = config.epoch_num
        self.patience = config.patience

    def lr_decay(self, optimizer, epoch, decay_rate, init_lr):
        lr = init_lr/(1+decay_rate*epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, traindata, devdata, testdata, evaluator, classification=False):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Train Instances Size:%d"%len(traindata.dataset))
        print("  Dev Instances Size:%d"%len(devdata.dataset))
        print(" Test Instances Size:%d"%len(testdata.dataset))
        # FIXME: 若不评价训练集，则dev、test表现会出现差异（相比于正常三个都评价）。这里提供补丁
        # prev_rng_state = torch.get_rng_state()

        for i in range(self.epoch_num):
            self.network.train()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            begin_time = time.time()
            total_loss = 0
            if self.optimizer_name == 'sgd':
                self.lr_decay(self.optimizer, i, self.config.lr_decay, self.config.learning_rate)

            # torch.set_rng_state(prev_rng_state)
            for batch in traindata:
                loss = self.network(batch)
                total_loss += loss.item()
                loss.backward()
                if self.optimizer_name == 'adamw':
                    print(self.optimizer)
                    print(self.scheduler)
                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.network.zero_grad()
                    continue
                self.optimizer.step()
                self.network.zero_grad()
            end_time = time.time()
            print("training time: %d, loss: %.4f" % (end_time-begin_time, total_loss))

            # dev, test evaluate
            # prev_rng_state = torch.get_rng_state()
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                if classification:
                    train_f = evaluator.evaluate_classification(self.network, traindata)
                    print("Train: acc: %.4f" % (train_f))
                    dev_f = evaluator.evaluate_classification(self.network, devdata)
                    print("Dev: acc: %.4f" % (dev_f))
                    test_f = evaluator.evaluate_classification(self.network, testdata)
                    print("Test: acc: %.4f" % (test_f))
                else:
                    train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, traindata)
                    print("Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                    dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, devdata)
                    print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                    test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, testdata)
                    print(" Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                if dev_f > best_dev_score:
                    best_dev_score = dev_f
                    test_when_best_dev = test_f
                    best_epoch_num = i+1
                    num_iters_nochange = 0
                    print("Saving model to ", self.config.save_model_dir)
                    torch.save(self.network.state_dict(), self.config.save_model_dir)
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > self.patience:
                        break
            sys.stdout.flush()
        print("Best epoch: %d, dev: %.4f, test: %.4f"%(best_epoch_num, best_dev_score, test_when_best_dev))

    def train_dann(self, source_traindata, source_devdata,
                   target_traindata, target_testdata, evaluator):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Source Domain:")
        print("Train Instances Size:%d"%len(source_traindata.dataset))
        print("  Dev Instances Size:%d"%len(source_devdata.dataset))
        print("Target Domain:")
        print("Train Instances Size:%d"%len(target_traindata.dataset))
        print(" Test Instances Size:%d" % len(target_testdata.dataset))
        for i in range(self.epoch_num):
            self.network.train()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            begin_time = time.time()
            if self.optimizer_name == 'sgd':
                self.lr_decay(self.optimizer, i, self.config.lr_decay, self.config.learning_rate)
            total_sl_loss, total_cls_loss, total_loss = 0, 0, 0

            n_batches = min(len(source_traindata), len(target_traindata))
            batches = list(zip(source_traindata, target_traindata))[:n_batches]
            # batches = list(zip(source_traindata, target_traindata))[:2]
            for batch1, batch2 in batches:
                batch = batch1, batch2
                loss = self.network(batch)
                total_loss += loss.item()
                loss.backward()
                if self.optimizer_name == 'adamw':
                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.network.zero_grad()
                    continue
                self.optimizer.step()
                self.network.zero_grad()
            end_time = time.time()
            print("training time: %d, ner loss: %.4f, domain classifier loss: %.4f, total loss: %.4f" %
                  (end_time-begin_time, total_sl_loss, total_cls_loss, total_loss))

            # dev, test evaluate
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, source_traindata)
                print("Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, source_devdata)
                print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, target_testdata)
                print(" Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                if dev_f > best_dev_score:
                    best_dev_score = dev_f
                    test_when_best_dev = test_f
                    best_epoch_num = i+1
                    num_iters_nochange = 0
                    print("Saving model to ", self.config.save_model_dir)
                    torch.save(self.network.state_dict(), self.config.save_model_dir)
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > self.patience:
                        break
            sys.stdout.flush()
        print("Best epoch: %d, dev: %.4f, test: %.4f"%(best_epoch_num, best_dev_score, test_when_best_dev))

    def train_wdgrl(self, source_traindata, source_devdata,
                   target_traindata, target_testdata, evaluator):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Source Domain:")
        print("Train Instances Size:%d"%len(source_traindata.dataset))
        print("  Dev Instances Size:%d"%len(source_devdata.dataset))
        print("Target Domain:")
        print("Train Instances Size:%d"%len(target_traindata.dataset))
        print(" Test Instances Size:%d" % len(target_testdata.dataset))
        for i in range(self.epoch_num):
            self.network.train()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            begin_time = time.time()
            if self.optimizer_name == 'sgd':
                self.lr_decay(self.optimizer, i, self.config.lr_decay, self.config.learning_rate)
            total_loss = 0

            n_batches = min(len(source_traindata), len(target_traindata))
            # batches = list(zip(source_traindata, target_traindata))[:n_batches]
            batches = list(zip(source_traindata, target_traindata))[:2]
            for batch1, batch2 in batches:
                batch = batch1, batch2
                loss = self.network(batch)
                total_loss += loss.item()
                print(loss.item())
                loss.backward()
                if self.optimizer_name == 'adamw':
                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.network.zero_grad()
                    continue
                self.optimizer.step()
                self.network.zero_grad()
            end_time = time.time()
            print("training time: %d, total loss: %.4f" % (end_time-begin_time, total_loss))

            # dev, test evaluate
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, source_traindata)
                print("Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, source_devdata)
                print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, target_testdata)
                print(" Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                if dev_f > best_dev_score:
                    best_dev_score = dev_f
                    test_when_best_dev = test_f
                    best_epoch_num = i+1
                    num_iters_nochange = 0
                    print("Saving model to ", self.config.save_model_dir)
                    torch.save(self.network.state_dict(), self.config.save_model_dir)
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > self.patience:
                        break
            sys.stdout.flush()
        print("Best epoch: %d, dev: %.4f, test: %.4f"%(best_epoch_num, best_dev_score, test_when_best_dev))

    def train_multi_tri(self, source_train_dataloader, source_dev_dataloader,
                        target_train_data, target_test_dataloader, evaluator):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Source Domain: ")
        print("   Train Instances Size: %d" % len(source_train_dataloader.dataset))
        print("     Dev Instances Size: %d" % len(source_dev_dataloader.dataset))
        print("Target Domain: ")
        print("   Train Instances Size: %d" % len(target_train_data))
        print("    Test Instances Size: %d" % len(target_test_dataloader.dataset))
        for i in range(self.epoch_num):
            self.network.train()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" % (i + 1, self.epoch_num))
            begin_time = time.time()
            total_loss = 0
            for batch in source_train_dataloader:
                loss = self.network(batch)
                print(loss.item())
                total_loss += loss.item()
                loss.backward()
                if self.optimizer_name == 'adamw':
                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.network.zero_grad()
                    continue
                self.optimizer.step()
                self.network.zero_grad()
            end_time = time.time()
            print("training time: %d, loss: %.4f" % (end_time - begin_time, total_loss))

            # dev, test evaluate
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, source_train_dataloader)
                print("Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, source_dev_dataloader)
                print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, target_test_dataloader)
                print(" Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                if dev_f > best_dev_score:
                    best_dev_score = dev_f
                    test_when_best_dev = test_f
                    best_epoch_num = i + 1
                    num_iters_nochange = 0
                    print("Saving model to ", self.config.save_model_dir)
                    torch.save(self.network.state_dict(), self.config.save_model_dir)
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > self.patience:
                        break
            sys.stdout.flush()
        print("Best epoch: %d, dev: %.4f, test: %.4f" % (best_epoch_num, best_dev_score, test_when_best_dev))

    def train_tri(self, train_data_0, train_data_1, train_data_2, train_data_unlabel,
                  dev_dataloader, test_dataloader, evaluator, batch_size=16):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Target Training...")
        print("   Train Instances for model0 Size:%d"%len(train_data_0))
        print("   Train Instances for model1 Size:%d"%len(train_data_1))
        print("   Train Instances for model2 Size:%d"%len(train_data_2))
        print("            Unlabel Instances Size:%d"%len(train_data_unlabel))
        print("                Dev Instances Size:%d"%len(dev_dataloader.dataset))
        print("               Test Instances Size:%d"%len(test_dataloader.dataset))
        for i in range(self.epoch_num):
            self.network.train()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            print("   Train Instances for model0 Size:%d" % len(train_data_0))
            print("   Train Instances for model1 Size:%d" % len(train_data_1))
            print("   Train Instances for model2 Size:%d" % len(train_data_2))
            print("            Unlabel Instances Size:%d" % len(train_data_unlabel))
            begin_time = time.time()
            # make sure model is converged, then predict the unlabeled
            # generate dataloader and train three models on their own training data
            # model 0 training
            while True:
                total_loss = 0
                train_data = Data.DataLoader(dataset=train_data_0,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fn if not torch.cuda.is_available() else collate_fn_cuda)
                for batch in train_data:
                    batch = batch, 0
                    loss = self.network(batch)
                    total_loss += loss.item()
                    loss.backward()
                    if self.optimizer_name == 'adamw':
                        # self.scheduler.step()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.network.zero_grad()
                        continue
                    self.optimizer.step()
                    self.network.zero_grad()
                end_time = time.time()
                print("model 0 training time: %d, loss: %.4f" % (end_time-begin_time, total_loss))
                # model 1 training
                total_loss = 0
                train_data = Data.DataLoader(dataset=train_data_1,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fn if not torch.cuda.is_available() else collate_fn_cuda)
                for batch in train_data:
                    batch = batch, 1
                    loss = self.network(batch)
                    total_loss += loss.item()
                    loss.backward()
                    if self.optimizer_name == 'adamw':
                        # self.scheduler.step()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.network.zero_grad()
                        continue
                    self.optimizer.step()
                    self.network.zero_grad()
                end_time = time.time()
                print("model 1 training time: %d, loss: %.4f" % (end_time - begin_time, total_loss))
                # model 2 training
                total_loss = 0
                train_data = Data.DataLoader(dataset=train_data_2,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fn if not torch.cuda.is_available() else collate_fn_cuda)
                for batch in train_data:
                    batch = batch, 2
                    loss = self.network(batch)
                    total_loss += loss.item()
                    loss.backward()
                    if self.optimizer_name == 'adamw':
                        # self.scheduler.step()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.network.zero_grad()
                        continue
                    self.optimizer.step()
                    self.network.zero_grad()
                end_time = time.time()
                print("model 2 training time: %d, loss: %.4f" % (end_time - begin_time, total_loss))

                self.network.eval()
                print("Evaluating...")
                with torch.no_grad():
                    # set network in "eval" model
                    self.network.eval()
                    dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, dev_dataloader)
                    print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                    test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, test_dataloader)
                    print(
                        " Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                    if dev_f > best_dev_score:
                        best_dev_score = dev_f
                        test_when_best_dev = test_f
                        best_epoch_num = i + 1
                        num_iters_nochange = 0
                        print("Saving model to ", self.config.save_model_dir)
                        torch.save(self.network.state_dict(), self.config.save_model_dir)
                    else:
                        num_iters_nochange += 1
                        if num_iters_nochange > self.patience:
                            break
            # set network in "eval" model
            self.network.eval()
            # add training data to 3 model
            print("Add Training data to 3 models......")
            increase_num = 0
            for j in range(len(train_data_unlabel)):
                try:
                    subword_idx, subword_head_mask, subword_mask, sent, tagseq = train_data_unlabel[j]
                except IndexError:
                    break
                belonging_num, pseudo_tagseq = self.network.add_training_data(subword_idx, subword_head_mask, subword_mask, tagseq)

                # turn tensor to python list with tag name
                if belonging_num != -1:
                    np_pseudo_tagseq = pseudo_tagseq[0].cpu().tolist()
                    for k in range(len(np_pseudo_tagseq)):
                        np_pseudo_tagseq[k] = evaluator.label_vocab.get_instance(np_pseudo_tagseq[k])
                
                if belonging_num == -1:
                    continue
                elif belonging_num == 0:
                    train_data_0.sents.append(sent)
                    train_data_0.labels.append(np_pseudo_tagseq)
                    train_data_unlabel.sents.remove(sent)
                elif belonging_num == 1:
                    train_data_1.sents.append(sent)
                    train_data_1.labels.append(np_pseudo_tagseq)
                    train_data_unlabel.sents.remove(sent)
                elif belonging_num == 2:
                    train_data_2.sents.append(sent)
                    train_data_2.labels.append(np_pseudo_tagseq)
                    train_data_unlabel.sents.remove(sent)
                increase_num += 1
            print("Total Increase Num: %d" % increase_num)

            # TODO：每一句话可能只标了一部分就被加到训练数据中了，是不是应该重新再标一遍
            # dev, test evaluate
            print("Evaluating...")
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, dev_dataloader)
                print("  Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, test_dataloader)
                print(" Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
                if dev_f > best_dev_score:
                    best_dev_score = dev_f
                    test_when_best_dev = test_f
                    best_epoch_num = i+1
                    num_iters_nochange = 0
                    print("Saving model to ", self.config.save_model_dir)
                    torch.save(self.network.state_dict(), self.config.save_model_dir)
                else:
                    num_iters_nochange += 1
                    if num_iters_nochange > self.patience:
                        break
            sys.stdout.flush()
        print("Best epoch: %d, dev: %.4f, test: %.4f"%(best_epoch_num, best_dev_score, test_when_best_dev))
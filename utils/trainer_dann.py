import time
import torch.optim as optim
import torch
import sys
import torch.utils.data as Data
from utils.utils import calulate_KL_Loss

class Trainer():
    def __init__(self, network, ad_network, config, trainloader):
        self.network = network
        self.ad_network = ad_network
        self.config = config
        self.optimizer_name = config.optimizer.lower()
        if self.optimizer_name == 'sgd':
            print("Using SGD optimizer...learning rate set as %f" % config.learning_rate)
            self.optimizer = optim.SGD(self.network.parameters(), lr=config.learning_rate)
            self.optimizer_ad = optim.SGD(self.ad_network.parameters(), lr=config.learning_rate)
        elif self.optimizer_name == 'adam':
            print("Using Adam optimizer...learning rate set as %f" % config.learning_rate)
            self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate, weight_decay=1e-08)
            self.optimizer_ad = optim.Adam(self.ad_network.parameters(), lr=config.learning_rate, weight_decay=1e-08)
        elif self.optimizer_name == 'rmsprop':
            print("Using RMSprop optimizer...learning rate set %f" % config.learning_rate)
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            self.optimizer_ad = optim.RMSprop(self.ad_network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        else:
            print("Wrong optimizer, please choose among SGD/Adam/RMSprop")
        self.epoch_num = config.epoch_num
        self.cls_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.patience = config.patience

    def lr_decay(self, optimizer, epoch, decay_rate, init_lr):
        lr = init_lr/(1+decay_rate*epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
            self.ad_network.train()
            self.ad_network.zero_grad()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            begin_time = time.time()
            if self.optimizer_name == 'sgd':
                self.lr_decay(self.optimizer, i, self.config.lr_decay, self.config.learning_rate)
            total_ner_loss, total_cls_loss, total_loss = 0, 0, 0

            n_batches = min(len(source_traindata), len(target_traindata))
            # batches = list(zip(source_traindata, target_traindata))[:n_batches]
            batches = list(zip(source_traindata, target_traindata))[:2]
            for batch1, batch2 in batches:
                batch = batch1, batch2
                s_feature, t_feature, ner_loss = self.network(batch)
                # 计算判别器的loss
                cls_loss = self.ad_network(s_feature, t_feature)
                # backprop等
                loss = cls_loss + ner_loss
                total_ner_loss += ner_loss.item()
                total_cls_loss += cls_loss.item()
                loss.backward()
                self.optimizer.step()
                if i > 0:
                    self.optimizer_ad.step()
                self.network.zero_grad()
                self.ad_network.zero_grad()
            end_time = time.time()
            print("training time: %d, ner loss: %.4f, domain classifier loss: %.4f, total loss: %.4f" %
                  (end_time-begin_time, total_ner_loss, total_cls_loss, total_cls_loss+total_ner_loss))

            # dev, test evaluate
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                self.ad_network.eval()
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

    def train_trident(self, source_traindata, target_traindata,
                    target_devdata, target_testdata, evaluator,
                    Ad_loss=False, KL_loss=False):
        num_iters_nochange = 0
        best_dev_score = -10
        test_when_best_dev = -10
        best_epoch_num = -10
        print("Model Training...")
        print("Source Domain:")
        print("Train Instances Size:%d"%len(source_traindata.dataset))
        print("Target Domain:")
        print("Train Instances Size:%d"%len(target_traindata.dataset))
        print("Dev   Instances Size:%d"%len(target_devdata.dataset))
        print(" Test Instances Size:%d" % len(target_testdata.dataset))
        for i in range(self.epoch_num):
            self.network.train()
            self.ad_network.train()
            self.ad_network.zero_grad()
            self.network.zero_grad()
            print("=========Epoch: %d / %d=========" %(i+1, self.epoch_num))
            begin_time = time.time()
            if self.optimizer_name == 'sgd':
                self.lr_decay(self.optimizer, i, self.config.lr_decay, self.config.learning_rate)
            Loss_ner_source, Loss_ner_target = 0, 0
            Loss_classify, Loss_KL = 0, 0

            n_batches = min(len(source_traindata), len(target_traindata))
            batches = list(zip(source_traindata, target_traindata))[:n_batches]
            for batch1, batch2 in batches:
                batch = batch1, batch2
                source_ner_loss, target_ner_loss, s_feature, t_feature = self.network(batch)
                loss = source_ner_loss + target_ner_loss
                if Ad_loss:
                    cls_loss = self.ad_network(s_feature, t_feature, i)
                    loss += cls_loss
                    Loss_classify += cls_loss.item()
                if KL_loss:
                    kl_loss = calulate_KL_Loss(s_feature, t_feature)
                    loss += kl_loss
                    Loss_KL += kl_loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer_ad.step()
                self.network.zero_grad()
                self.ad_network.zero_grad()
                Loss_ner_source += source_ner_loss.item()
                Loss_ner_target += target_ner_loss.item()
            end_time = time.time()
            print("training time: %d, source ner loss: %.4f, target ner loss: %.4f" % (end_time-begin_time, Loss_ner_source, Loss_ner_target))
            if Ad_loss:
                print("domain  classifier loss: %.4f" % Loss_classify)
            if KL_loss:
                print("feature adaptation Loss: %.4f" % Loss_KL)

            # dev, test evaluate
            with torch.no_grad():
                # set network in "eval" model
                self.network.eval()
                self.ad_network.eval()
                # train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, source_traindata)
                # print("Source Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                train_acc, train_precision, train_recall, train_f = evaluator.evaluate(self.network, target_traindata)
                print("Target Train: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (train_acc, train_precision, train_recall, train_f))
                dev_acc, dev_precision, dev_recall, dev_f = evaluator.evaluate(self.network, target_devdata)
                print("Target   Dev: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_acc, dev_precision, dev_recall, dev_f))
                test_acc, test_precision, test_recall, test_f = evaluator.evaluate(self.network, target_testdata)
                print("Target  Test: acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_acc, test_precision, test_recall, test_f))
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

    def frozen_Ad_Net(self):
        for para in self.ad_network.parameters():
            para.requires_grad = False

    def defrozen_Ad_Net(self):
        for para in self.ad_network.parameters():
            para.requires_grad = True
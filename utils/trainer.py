import sys
import time

import torch
import torch.optim as optim
import torch.utils.data as Data
from transformers import AdamW, get_linear_schedule_with_warmup


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
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                             num_warmup_steps=num_warmup_steps, 
                                                             num_training_steps=num_total_steps)
        elif self.optimizer_name == 'rmsprop':
            print("Using RMSprop optimizer...learning rate set %f" % config.learning_rate)
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        else:
            print("Wrong optimizer, please choose among SGD/Adam/BertAdam")
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
        # 若不评价训练集，则dev、test表现会出现差异（相比于正常三个都评价）。这里提供补丁
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
                self.network.eval()
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

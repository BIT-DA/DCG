import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import learn2learn as l2l

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *


import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        self.encoder = l2l.algorithms.MAML(self.encoder, lr = self.config["meta_lr_g"])
        self.classifier = l2l.algorithms.MAML(self.classifier, lr= self.config["meta_lr_c"])
       
        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])


        # dataloaders
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        self.encoder.train()
        self.classifier.train()
        for it, (batch, label, domain) in enumerate(self.train_loader):
            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            labels = torch.cat(label, dim=0).to(self.device)
            domains = torch.cat(domain, dim=0).to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            data_ori, data_aug = torch.split(batch, split_idx)
            labels_ori, labels_aug = torch.split(labels, split_idx)
            domains_ori, domains_aug = torch.split(domains, split_idx)
            assert data_ori.size(0) == data_aug.size(0)

            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}

            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()

            # generate groups for convex game
            if severe_unbalance_domain(domains_ori):
                continue
            ## ori domains
            rand_domain = torch.multinomial(torch.ones(3).float(), 3, replacement=False)
            meta_train_data_ori, meta_train_labels_ori, meta_train_domains_ori, meta_train_ind, meta_test_data, meta_test_labels, meta_test_ind = \
                generate_meta_data(data_ori, labels_ori, domains_ori, rand_domain)
            meta_train_data_ori = torch.autograd.Variable(meta_train_data_ori, requires_grad=True)
            meta_train_data_s, meta_train_labels_s, meta_train_data_t, meta_train_labels_t, \
            meta_train_data_p, meta_train_labels_p, meta_train_data_q, meta_train_labels_q = \
                generate_meta_train_groups(meta_train_data_ori, meta_train_labels_ori, meta_train_domains_ori, type="vector")

            ## aug domains
            meta_train_data_aug, meta_train_labels_aug, meta_train_ind_aug = filt_aug(data_aug, labels_aug, domains_aug, rand_domain)
            if meta_train_labels_aug.size(0) > 1:
                meta_train_data_aug = torch.autograd.Variable(meta_train_data_aug, requires_grad=True)
                meta_train_data_aug_s, meta_train_labels_aug_s, meta_train_data_aug_t, meta_train_labels_aug_t, \
                meta_train_data_aug_p, meta_train_labels_aug_p, meta_train_data_aug_q, meta_train_labels_aug_q = \
                    generate_meta_train_groups_aug(meta_train_data_aug, meta_train_labels_aug, type="sample")

                ## concat all domains
                meta_train_data = torch.cat((meta_train_data_ori, meta_train_data_aug),dim=0)
                meta_train_labels = torch.cat((meta_train_labels_ori, meta_train_labels_aug), dim=0)
                meta_train_ind = torch.cat((meta_train_ind, torch.tensor(meta_train_ind_aug).cuda() + int(batch.size(0) / 2)), dim=0)

                meta_train_data_s = torch.cat((meta_train_data_s, meta_train_data_aug_s), dim=0)
                meta_train_labels_s = torch.cat((meta_train_labels_s, meta_train_labels_aug_s), dim=0)
                meta_train_data_t = torch.cat((meta_train_data_t, meta_train_data_aug_t), dim=0)
                meta_train_labels_t = torch.cat((meta_train_labels_t, meta_train_labels_aug_t), dim=0)
                meta_train_data_p = torch.cat((meta_train_data_p, meta_train_data_aug_p), dim=0)
                meta_train_labels_p = torch.cat((meta_train_labels_p, meta_train_labels_aug_p), dim=0)
                meta_train_data_q = torch.cat((meta_train_data_q, meta_train_data_aug_q), dim=0)
                meta_train_labels_q = torch.cat((meta_train_labels_q, meta_train_labels_aug_q), dim=0)


            # convex game between domains --> cg_loss
            loss_s = meta_loss(self.encoder, self.classifier, meta_train_data_s, meta_train_labels_s, meta_test_data,
                               meta_test_labels)
            loss_t = meta_loss(self.encoder, self.classifier, meta_train_data_t, meta_train_labels_t, meta_test_data,
                               meta_test_labels)
            loss_p = meta_loss(self.encoder, self.classifier, meta_train_data_p, meta_train_labels_p, meta_test_data,
                               meta_test_labels)
            loss_q = meta_loss(self.encoder, self.classifier, meta_train_data_q, meta_train_labels_q, meta_test_data,
                               meta_test_labels)
            
            loss_sm = loss_p + loss_q - loss_s - loss_t
            loss_reg = F.relu(loss_sm)

            # sample filter
            grad = torch.autograd.grad(loss_sm, meta_train_data_ori, retain_graph=True)
            grad_input = torch.sum(
                meta_train_data_ori.view(meta_train_data_ori.size(0), -1) * grad[0].view(grad[0].size(0), -1), dim=1)
            if meta_train_labels_aug.size(0) > 1:
                grad_aug = torch.autograd.grad(loss_sm, meta_train_data_aug, retain_graph=True)
                grad_input_aug = torch.sum(
                    meta_train_data_aug.view(meta_train_data_aug.size(0), -1) * grad_aug[0].view(grad_aug[0].size(0), -1),
                    dim=1)
                grad_input = torch.cat((grad_input, grad_input_aug), dim=0)

            value, ind = torch.topk(grad_input, self.config["k"])
            filter = torch.ones(batch.size(0)).cuda()
            filter[meta_train_ind[ind]] = 0

            ## classification loss
            features = self.encoder(batch)
            scores = self.classifier(features)
            loss_cls = torch.sum(nn.CrossEntropyLoss(reduction='none')(scores, labels) * filter) / (
                        batch.size(0) - self.config["k"])

            # record
            loss_dict["cls"] = loss_cls.item()
            correct_dict["cls"] = calculate_correct(scores, labels)
            num_samples_dict["cls"] = int(scores.size(0))
            loss_dict["sm"] = loss_sm.item()

            # get consistency weight
            const_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                          weight=self.config["w_reg"],
                                                          rampup_length=self.config["warmup_epoch"],
                                                          rampup_type=self.config["warmup_type"])

            # calculate total loss
            total_loss = loss_cls + const_weight * loss_reg
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best val
            if self.results['val'][self.current_epoch] >= self.best_val_acc:
                self.best_val_acc = self.results['val'][self.current_epoch]
                self.best_val_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_val_acc)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_val_acc = 0
        self.best_val_epoch = 0

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch()
            print('-' * 30)
            print("Best test: %g" % (self.results['test'].max()))
            self.logger.finish_epoch()

        # save from best val
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_val_acc, self.best_val_epoch - 1)

        return self.logger


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
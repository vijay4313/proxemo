import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight

from net import classifier


class Trainer(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, C, num_classes, graph_dict, device='cuda:0'):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.device = device
        self.io = torchlight.IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.model = classifier.Classifier(C, num_classes, graph_dict)
        self.model.cuda('cuda:0')
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.best_loss = math.inf
        self.step_epochs = [
            math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args.topk)))
        self.accuracy_updated = False

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr

    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and\
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args.base_lr * (
                0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step_epochs)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self):

        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
        else:
            self.accuracy_updated = False
        self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}%.'.format(
            k, accuracy, self.best_accuracy[0, k-1]))

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output, _ = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        # for k in self.args.topk:
        #     self.calculate_topk(k, show=False)
        # if self.accuracy_updated:
        # self.model.extract_feature()

    def per_test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output, _ = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.args.topk:
                self.show_topk(k)

    def train(self):

        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test()
                self.io.print_log('Done.')

            # save model and weights
            if self.accuracy_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.work_dir,
                                        'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))
                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.args.model))
        self.io.print_log('Weights: {}.'.format(self.args.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.per_test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def save_best_feature(self, ftype, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(
                self.args.work_dir)
        else:
            best_accuracy = self.best_accuracy.item()
        filename = os.path.join(self.args.work_dir,
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
        self.model.load_state_dict(torch.load(filename))
        features = np.empty((0, 256))
        fCombined = h5py.File('../data/features'+ftype+'.h5', 'r')
        fkeys = fCombined.keys()
        dfCombined = h5py.File('../data/deepFeatures'+ftype+'.h5', 'w')
        for i, (each_data, each_key) in enumerate(zip(data, fkeys)):

            # get data
            each_data = np.reshape(
                each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.device)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]
                dfCombined.create_dataset(fname, data=feature.cpu())
                features = np.append(features, np.array(
                    feature.cpu()).reshape((1, feature.shape[0])), axis=0)
        dfCombined.close()
        return features

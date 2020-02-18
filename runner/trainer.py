import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import (find_all_substr, get_best_epoch_and_accuracy,
                               get_loss_fn, get_optimizer, weights_init)
from zoo.classifier_stgcn.classifier import Classifier
from zoo.vscnn.vscnn_base import ViewGroupPredictor
from zoo.vscnn.vscnn_base import ViewGroupFeature

MODEL_TYPE = {
    'stgcn': Classifier,
    'vscnn_view_group_predictor': ViewGroupPredictor,
    'vscnn_view_group_feature': ViewGroupFeature
}


class Trainer(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, num_classes, graph_dict):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.best_loss = math.inf
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args['TPOK'])))
        self.accuracy_updated = False
        log_dir = os.path.join(args['OUTPUT_PATH'], 'logs')
        self.logger = SummaryWriter(log_dir=log_dir)
        self.cuda = args['CUDA_DEVICE'] if args['CUDA_DEVICE'] is not None else 0
        self.graph_dict = graph_dict
        self.TERMINAL_LOG = args['TERMINAL_LOG']
        self.build_model()

    def build_model(self):
        # model
        self.model = MODEL_TYPE[self.args['MODEL']['TYPE']](
            self.args['DATA']['COORDS'],
            self.num_classes,
            self.graph_dict
            )
        self.model.cuda(self.cuda)
        self.model.apply(weights_init)
        self.loss = get_loss_fn(self.args['MODEL']['LOSS'])
        self.step_epochs = np.array([
            math.ceil(float(self.args['EPOCHS'] / x)) for x in self.args['STEP']])
        # optimizer
        optimizer_args = self.args['MODEL']['OPTIMIZER']
        self.lr = optimizer_args['LR']
        # handling multiple modes in case of feature predictor for vscnn
        if self.args['MODEL']['OPTIMIZER'] == 'vscnn_view_group_predictor':
            self.optimizer = []
            for model in self.model.models:
                self.optimizer.append(get_optimizer(optimizer_args['TYPE'])(model.parameters(),
                                                                   lr=self.lr,
                                                                   weight_decay=optimizer_args['WEIGHT_DECAY']))
        else:
            self.optimizer = get_optimizer(optimizer_args['TYPE'])(self.model.parameters(),
                                                                   lr=self.lr,
                                                                   weight_decay=optimizer_args['WEIGHT_DECAY'])
        
    def adjust_lr(self):

        # if self.args.optimizer == 'SGD' and\
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.args['MODEL']['OPTIMIZER']['LR'] * (
                0.1 ** np.sum(self.meta_info['epoch'] >= self.step_epochs))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self, mode):
        print_str = ''
        for k, v in self.epoch_info.items():
            self.logger.add_scalar(
                "-".join([mode, "Epoch", k]), v, self.meta_info['epoch'])
            if isinstance(v, float):
                print_str = print_str + ' | {}: {:.4f}'.format(k.upper(), v)
            else:
                print_str = print_str + ' | {}: {}'.format(k.upper(), v)

        if self.TERMINAL_LOG:
            print('{mode}-Epoch: {epoch} | {info}'.format(mode=mode.upper(),
                                                          epoch=self.meta_info['epoch'],
                                                          info=print_str))

    def show_iter_info(self, mode):
        info = ''
        for k, v in self.iter_info.items():
#            self.logger.add_scalar(
#                "-".join([mode, "Iter", k]), v, self.meta_info['iter'])
            if isinstance(v, float):
                info = info + ' | {}: {:.4f}'.format(k, v)
            else:
                info = info + ' | {}: {}'.format(k, v)

        if (self.meta_info['iter'] % self.args['LOG_INTERVAL'] == 0) and self.TERMINAL_LOG:
            print('{mode}\tIter: {iter}. {info}'.format(mode=mode,
                                                        iter=self.meta_info['iter'],
                                                        info=info))

    def show_topk(self, k):

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
        else:
            self.accuracy_updated = False
#        self.io.print_log('\tTop{}: {:.2f}%. Best so far: {:.2f}%.'.format(
#            k, accuracy, self.best_accuracy[0, k-1]))

    def per_train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.cuda)
            label = label.long().to(self.cuda)
            
            # TODO: get viewgroup info
            group = None

            # forward
            # handling multiple modes in case of feature predictor for vscnn
            if self.args['MODEL']['OPTIMIZER'] == 'vscnn_view_group_predictor':
                loss = 0
                outputs = self.model(data, group)
                for index in range(len(self.model.models)):
                    if outputs[index] is not None:
                        loss += self.loss(outputs[index], label)
        
                        # backward
                        self.optimizer[index].zero_grad()
                        loss.backward()
                        self.optimizer[index].step()
            else:
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
            self.show_iter_info(mode='train')
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info(mode='train')
        # self.io.print_timer()
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
            data = data.float().to(self.cuda)
            label = label.long().to(self.cuda)

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
            self.show_epoch_info(mode='test')

            # show top-k accuracy
            for k in self.args['TPOK']:
                self.show_topk(k)

    def train(self):

        for epoch in range(self.args['START_EPOCH'], self.args['EPOCHS']):
            self.meta_info['epoch'] = epoch

            # training
            self.per_train()

            # evaluation
            if (epoch % self.args['EVAL_INTERVAL'] == 0) or (
                    epoch + 1 == self.args['EPOCHS']):
                self.per_test()

            # save model and weights
            if self.accuracy_updated:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args['WORK_DIR'],
                                        'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))
                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch

    def test(self):

        # the path of weights must be appointed
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')

        self.per_test()

        # save the output of model
        if self.args.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')

    def save_best_feature(self, _features_file, _save_file, data, joints, coords):
        if self.best_epoch is None:
            self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(
                self.args['WORK_DIR'])
        else:
            best_accuracy = self.best_accuracy.item()
        filename = os.path.join(self.args['WORK_DIR'],
                                'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
        self.model.load_state_dict(torch.load(filename))
        features = np.empty((0, 256))
        fCombined = h5py.File(_features_file, 'r')
        fkeys = fCombined.keys()
        dfCombined = h5py.File(_save_file, 'w')
        for i, (each_data, each_key) in enumerate(zip(data, fkeys)):

            # get data
            each_data = np.reshape(
                each_data, (1, each_data.shape[0], joints, coords, 1))
            each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
            each_data = torch.from_numpy(each_data).float().to(self.cuda)

            # get feature
            with torch.no_grad():
                _, feature = self.model(each_data)
                fname = [each_key][0]
                dfCombined.create_dataset(fname, data=feature.cpu())
                features = np.append(features, np.array(
                    feature.cpu()).reshape((1, feature.shape[0])), axis=0)
        dfCombined.close()
        return features

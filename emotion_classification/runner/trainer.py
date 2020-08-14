#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import math
import os

from datetime import datetime
import pickle
import h5py
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import (find_all_substr, get_best_epoch_and_accuracy,
                               get_loss_fn, get_optimizer, weights_init,
                               SummaryStatistics)
from modeling.vs_gcnn import VSGCNN
from loader.loader import data_loader_base


MODEL_TYPE = {
    'vs_gcnn': VSGCNN
}


class Trainer(object):
    """Training manager class."""

    def __init__(self, gen_args, data_config, model_config):
        """Constructor.

        Args:
            gen_args (dict): General training args
            data_config (dict): Data args
            model_config (dict): Model args
            Refer modeling/config/train.yaml for details.
        """
        self.args = gen_args
        self.data_config = data_config
        self.model_config = model_config

        self.num_classes = model_config['NUM_CLASSES']
        self.num_viewgroups = model_config['NUM_GROUPS']
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.best_loss = math.inf
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args['TOPK'])))
        self.accuracy_updated = False

        self.setup()

    def setup(self):
        """Setup train/test routine."""
        now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        log_dir = os.path.join(self.args['OUTPUT_PATH'],
                               'logs', self.model_config['TYPE'], now)
        self.args['WORK_DIR'] = os.path.join(
            self.args['OUTPUT_PATH'], 'saved_models', self.model_config['TYPE'], now)
        self.args['RESULT_SAVE_DIR'] = os.path.join(
            self.args['OUTPUT_PATH'], 'test_result', self.model_config['TYPE'], now)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.cuda = self.args['CUDA_DEVICE'] if self.args['CUDA_DEVICE'] is not None else 0
        self.TERMINAL_LOG = self.args['TERMINAL_LOG']
        self.create_working_dir(self.args['WORK_DIR'])
        self.create_working_dir(self.args['RESULT_SAVE_DIR'])
        self.model_config['PRETRAIN_PATH'] = self.args['WORK_DIR']

        all_args = {'GENERAL': self.args,
                    'MODEL': self.model_config, 'DATA': self.data_config}

        with open(os.path.join(self.args['WORK_DIR'], 'settings.yaml'), 'w') as file:
            yaml.dump(all_args, file)

        # Data loader
        if self.args['MODE'] == 'train':
            test_size = 0.1
        else:
            test_size = 0.99

        if isinstance(self.data_config, dict):
            self.data_loader, num_classes, num_viewgroups = data_loader_base(
                self.args, self.data_config, test_size)
            self.num_classes = num_classes if num_classes is not None else self.num_classes
            self.num_viewgroups = num_viewgroups if num_viewgroups is not None else self.num_viewgroups
        elif self.data_config is not None:
            self.data_loader = self.data_config

        # Build Model
        self.build_model(self.model_config)

        if len(self.model_config['TARGETS']) > 1:
            self.summary_statistics = SummaryStatistics(
                self.num_classes * self.model_config['NUM_GROUPS'])
        else:
            self.summary_statistics = SummaryStatistics(self.num_classes)

    def create_working_dir(self, dir_name):
        """Create folder under given path.

        Args:
            dir_name (str): Directory path to be created
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def build_model(self, model_kwargs):
        """Build the necessary model.

        Args:
            model_kwargs (dict): model args
        """
        # model parameters
        if model_kwargs['TYPE'] == 'vs_gcnn':
            params = [self.num_classes,
                      model_kwargs['IN_CHANNELS'],
                      self.num_viewgroups,
                      model_kwargs['DROPOUT'],
                      model_kwargs['LAYER_CHANNELS']
                      ]
        else:
            raise ValueError("Invalid Model. Model Type should be \
                one of %s" % ', '.join(MODEL_TYPE.keys()))

        # model
        self.model = MODEL_TYPE[model_kwargs['TYPE']](*params)
        self.loss = get_loss_fn(model_kwargs['LOSS'])
        self.step_epochs = np.array(
            [math.ceil(float(self.args['EPOCHS'] * x)) for x in self.args['STEP']])

        # optimizer
        optimizer_args = model_kwargs['OPTIMIZER']
        self.lr = optimizer_args['LR']
        self.model.apply(weights_init)
        self.model.to(self.cuda)
        self.optimizer = get_optimizer(optimizer_args['TYPE'])(self.model.parameters(),
                                                               lr=self.lr,
                                                               weight_decay=optimizer_args['WEIGHT_DECAY'])

        if model_kwargs['PRETRAIN_NAME'] != '':
            self.load_model()

    def adjust_lr(self):
        """Adjust the learning rate.
        """
        if self.meta_info['epoch'] in self.step_epochs:
            lr = self.model_config['OPTIMIZER']['LR'] * (
                0.1 ** np.sum(self.meta_info['epoch'] >= self.step_epochs))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_epoch_info(self, mode):
        """Generate summary for a epoch.

        Args:
            mode (str): train/test
        """
        print_str = ''
        for k, v in self.epoch_info.items():
            self.logger.add_scalar(
                "-".join([mode, "epoch", k]), v, self.meta_info['epoch'])
            if isinstance(v, float):
                print_str = print_str + ' | {}: {:.4f}'.format(k.upper(), v)
            else:
                print_str = print_str + ' | {}: {}'.format(k.upper(), v)

        if self.TERMINAL_LOG:
            print('{mode}-Epoch: {epoch} | {info}'.format(mode=mode.upper(),
                                                          epoch=self.meta_info['epoch'],
                                                          info=print_str))

    def show_iter_info(self, mode):
        """Generate summary for an iteration.

        Args:
            mode (str): train/test
        """
        info = ''
        for k, v in self.iter_info.items():
            self.logger.add_scalar(
                "-".join([mode, "Iter", k]), v, self.meta_info['iter'])
            if isinstance(v, float):
                info = info + ' | {}: {:.4f}'.format(k, v)
            else:
                info = info + ' | {}: {}'.format(k, v)

        if (self.meta_info['iter'] % self.args['LOG_INTERVAL'] == 0) and self.TERMINAL_LOG:
            print('{mode}\tIter: {iter}. {info}'.format(mode=mode,
                                                        iter=self.meta_info['iter'],
                                                        info=info))

    def show_topk(self, k):
        """Show Top-k accuracy.

        Args:
            k (int): rank factor
        """
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if accuracy > self.best_accuracy[0, k-1]:
            self.best_accuracy[0, k-1] = accuracy
            self.accuracy_updated = True
        else:
            self.accuracy_updated = False
        print('\tTop{}: {:.2f}%. Best so far: {:.2f}%.'.format(
            k, accuracy, self.best_accuracy[0, k-1]))
        self.logger.add_scalar('test-Iter-accuracy',
                               accuracy,
                               self.meta_info['epoch'])
        self.summary_statistics.update(self.label, np.asarray(rank[:, -1]))

    def per_train(self):
        """Run a single training loop."""
        # put model in training mode
        if self.model_config['TYPE'] == 'vscnn_vgf':
            for _model in self.model.models:
                _model.train()
        else:
            self.model.train()

        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label_and_group in loader:
            # forward
            # get data
            data = data.float().to(self.cuda)
            label = label_and_group[0].long()  # emotion label
            group = label_and_group[1].long()  # view angle group
            if len(self.model_config['TARGETS']) > 1:
                label = (self.num_classes*label + group).to(self.cuda)
            else:
                label = label.long().to(self.cuda)

            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # statistics

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = self.lr
            loss_value.append(self.iter_info['loss'])
            self.meta_info['iter'] += 1
            self.show_iter_info(mode='train')

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info(mode='train')
        # self.io.print_timer()
        # for k in self.args.topk:
        #     self.calculate_topk(k, show=False)
        # if self.accuracy_updated:
        # self.model.extract_feature()

    def per_test(self, evaluation=True):
        """Run a single test loop.

        Args:
            evaluation (bool, optional): Evaluation/Inference. Defaults to True.
        """
        # put models in eval mode
        if self.model_config['TYPE'] == 'vscnn_vgf':
            # put models in eval mode
            for _model in self.model.models:
                _model.eval()
        else:
            self.model.eval()

        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        group_frag = []

        self.summary_statistics.reset()

        for data, label_and_group in loader:
            # get data
            data = data.float().to(self.cuda)
            label = label_and_group[0].long()  # emotion label
            group = label_and_group[1].long()  # view angle group
            if len(self.model_config['TARGETS']) > 1:
                label = (self.num_classes*label + group).to(self.cuda)
            else:
                label = label.to(self.cuda)

            # inference
            with torch.no_grad():
                output = self.model(data, apply_sfmax=False)
            result_frag.append(output.data.cpu().numpy())
            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
                group_frag.append(group.data.cpu().numpy())
#                    self.summary_statistics.update(label.data.cpu().numpy(),
#                                                   output.data.cpu().numpy().astype(int))

        self.result = np.concatenate(result_frag)

        if evaluation:
            self.group = np.concatenate(group_frag)
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info(mode='test')

            # show top-k accuracy
            for k in self.args['TOPK']:
                self.show_topk(k)

    def train(self):
        """Perform training routine."""

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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_value': self.epoch_info['mean_loss'],
                    'loss': self.loss},
                    os.path.join(self.args['WORK_DIR'], 'mdl_epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

#                    torch.save(self.model.state_dict(),
#                            os.path.join(self.args['WORK_DIR'],
#                                            'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

                if self.epoch_info['mean_loss'] < self.best_loss:
                    self.best_loss = self.epoch_info['mean_loss']
                    self.best_epoch = epoch

        print('best epoch: {}'.format(self.best_epoch))

    def test(self):
        """Perform Test routine."""
        self.per_test()
        self.result_summary = self.summary_statistics.get_metrics()
        file_name = 'test_result'
        save_file = os.path.join(
            self.args['RESULT_SAVE_DIR'], file_name+'.pkl')
        save_file_summary = os.path.join(
            self.args['RESULT_SAVE_DIR'], file_name+'_summary.txt')
        save_file_confusion = os.path.join(
            self.args['RESULT_SAVE_DIR'], file_name+'_confusion.csv')
        np.savetxt(save_file_confusion,
                   self.result_summary['conf_matrix'], delimiter=',')
        result_dict = dict(zip(self.label, self.result))
        with open(save_file, 'wb') as handle:
            pickle.dump(result_dict, handle)
        with open(save_file_summary, 'w') as handle:
            handle.write(str(self.result_summary))

    def load_model(self):
        """Load pretrained weights for model."""
        path = os.path.join(self.model_config['PRETRAIN_PATH'],
                            self.model_config['PRETRAIN_NAME'])

        checkpoint = torch.load(path, map_location=f'cuda:{self.cuda}')
        try:
            self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            self.meta_info['epoch'] = checkpoint['epoch']
            self.epoch_info['mean_loss'] = checkpoint['loss_value']
            self.loss = checkpoint['loss']
        except:
            self.model.load_state_dict(checkpoint, strict=True)

import math
import os

from datetime import datetime
import pickle
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
from zoo.vs_gcnn.vs_gcnn import VSGCNN

from utils import yaml_parser

MODEL_TYPE = {
    'stgcn': Classifier,
    'vscnn_vgp': ViewGroupPredictor,
    'vscnn_vgf': ViewGroupFeature,
    'vs_gcnn': VSGCNN
}

class Trainer(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, data_loader, num_classes, graph_dict, model_kwargs):

        self.args = args
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.best_loss = math.inf
        self.best_epoch = None
        self.best_accuracy = np.zeros((1, np.max(self.args['TOPK'])))
        self.accuracy_updated = False
        now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        log_dir = os.path.join(args['OUTPUT_PATH'], 'logs', args['MODEL']['TYPE'], now)
        self.args['WORK_DIR'] = os.path.join(args['OUTPUT_PATH'], 'saved_models', args['MODEL']['TYPE'], now)
        self.args['RESULT_SAVE_DIR'] = os.path.join(args['OUTPUT_PATH'], 'test_result', args['MODEL']['TYPE'], now)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.cuda = args['CUDA_DEVICE'] if args['CUDA_DEVICE'] is not None else 0
        self.graph_dict = graph_dict
        self.TERMINAL_LOG = args['TERMINAL_LOG']
        self.create_working_dir(self.args['WORK_DIR'])
        self.create_working_dir(self.args['RESULT_SAVE_DIR'])
        yaml_parser.copy_yaml(self.args['YAML_FILE_NAME'], self.args['WORK_DIR'])
        self.build_model(model_kwargs)
    
    def create_working_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def build_model(self, model_kwargs):
        # model parameters
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
            params = [self.args['DATA']['COORDS'],
                      self.num_classes,
                      model_kwargs['NUM_GROUPS'],
                      self.args['MODEL']['DROPOUT'],
                      self.args['MODEL']['LAYER_CHANNELS']
                      ]
        elif self.args['MODEL']['TYPE'] == 'vscnn_vgp':
            params = [self.args['DATA']['COORDS'],
                      self.num_classes
                      ]
        elif self.args['MODEL']['TYPE'] == 'vs_gcnn':
            params = [self.num_classes,
                      self.args['DATA']['COORDS'],
                      model_kwargs['NUM_GROUPS'],
                      self.args['MODEL']['DROPOUT'],
                      self.args['MODEL']['LAYER_CHANNELS']
                      ]
        elif self.args['MODEL']['TYPE'] == 'stgcn':
            params = [self.args['DATA']['COORDS'],
                      self.num_classes,
                      self.graph_dict
                      ]
        else:
            raise ValueError("Invalid Model. Model Type should be \
                one of %s" % ', '.join(MODEL_TYPE.keys()))

        # model
        self.model = MODEL_TYPE[self.args['MODEL']['TYPE']](*params)
        self.loss = get_loss_fn(self.args['MODEL']['LOSS'])
        self.step_epochs = np.array([math.ceil(float(self.args['EPOCHS'] * x)) for x in self.args['STEP']])
            
        # optimizer
        optimizer_args = self.args['MODEL']['OPTIMIZER']
        self.lr = optimizer_args['LR']
        
        # handling multiple modes in case of feature predictor for vscnn
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
            self.optimizer = []
            for model in self.model.models:
                model.apply(weights_init)
                model.to(self.cuda)
                self.optimizer.append(get_optimizer(optimizer_args['TYPE'])(model.parameters(),
                                                                   lr=self.lr,
                                                                   weight_decay=optimizer_args['WEIGHT_DECAY']))
        else:
            self.model.apply(weights_init)
            self.model.to(self.cuda)
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

    def per_train(self):
        # put model in training mode
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
            for _model in self.model.models:
                _model.train()
        else:
            self.model.train()
            
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label_and_group in loader:
            # forward
            # handling multiple modes in case of feature predictor for vscnn
            if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
                # get data and prepare according to group
                data = data.float()
                label = label_and_group[0].long() # emotion label
                group = label_and_group[1].long() # view angle group
                loss = 0
                # Train
                for index in range(len(self.model.models)):
                    # get index of specific group
                    req_index = np.where(group == index)[0]
                    if req_index.size > 0:
                        # get data according to group
                        group_data = data[req_index].to(self.cuda)
                        group_label = label[req_index].to(self.cuda)
                        # forward
                        output = self.model.models[index](group_data)
                        per_group_loss = self.loss(output, group_label)
                        loss += per_group_loss
                        # backward
                        self.optimizer[index].zero_grad()
                        per_group_loss.backward(retain_graph=True)
                        self.optimizer[index].step()
            else:
                # get data
                data = data.float().to(self.cuda)
                if len(self.args['MODEL']['TARGETS']) > 1:
                    label = label_and_group[0].long() # emotion label
                    group = label_and_group[1].long() # view angle group
                    label = (self.num_classes*label + group).to(self.cuda)
                else:
                    label = label_and_group.long().to(self.cuda)
                
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
        # put models in eval mode
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
                # put models in eval mode
                for _model in self.model.models:
                    _model.eval()
        else:
            self.model.eval()
            
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        for data, label_and_group in loader:
            if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
                # get data and prepare according to group
                data = data.float()
                label = label_and_group[0].long() # emotion label
                group = label_and_group[1].long() # view angle group
                # inference
                with torch.no_grad():
                    for index in range(len(self.model.models)):
                        # get index of specific group
                        req_index = np.where(group == index)[0]
                        if req_index.size > 0:
                            # get data according to group
                            group_data = data[req_index].to(self.cuda)
                            group_label = label[req_index].to(self.cuda)
                            # inference
                            output = self.model.models[index](group_data, apply_sfmax = True)
                            result_frag.append(output.data.cpu().numpy())
            
                            # get loss
                            if evaluation:
                                loss = self.loss(output, group_label)
                                loss_value.append(loss.item())
                                label_frag.append(group_label.data.cpu().numpy())
                    
            else:
                # get data
                data = data.float().to(self.cuda)
                if len(self.args['MODEL']['TARGETS']) > 1:
                    label = label_and_group[0].long() # emotion label
                    group = label_and_group[1].long() # view angle group
                    label = (self.num_classes*label + group).to(self.cuda)
                else:
                    label = label_and_group.long().to(self.cuda)
    
                # inference
                with torch.no_grad():
                    output = self.model(data, apply_sfmax = True)
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
            for k in self.args['TOPK']:
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
                if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
                    for idx in range(len(self.model.models)):
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.models[idx].state_dict(),
                                'optimizer_state_dict': self.optimizer[idx].state_dict(),
                                'loss_value': self.epoch_info['mean_loss'],
                                'loss': self.loss},
                                os.path.join(self.args['WORK_DIR'],'mdl{}_epoch{}_acc{:.2f}_model.pth.tar'.format(idx, epoch, self.best_accuracy.item())))
#                        torch.save(self.model.models[idx].state_dict(),
#                            os.path.join(self.args['WORK_DIR'], 
#                                            'mdl{}_epoch{}_acc{:.2f}_model.pth.tar'.format(idx, epoch, self.best_accuracy.item())))
                else:
                    
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss_value': self.epoch_info['mean_loss'],
                                'loss': self.loss},
                                os.path.join(self.args['WORK_DIR'],'mdl_epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))
                    
#                    torch.save(self.model.state_dict(),
#                            os.path.join(self.args['WORK_DIR'],
#                                            'epoch{}_acc{:.2f}_model.pth.tar'.format(epoch, self.best_accuracy.item())))

                if self.epoch_info['mean_loss'] < self.best_loss:
                        self.best_loss = self.epoch_info['mean_loss']
                        self.best_epoch = epoch
        
        print('best epoch: {}'.format(self.best_epoch))

    def test(self):
        self.per_test()
        file_name = 'test_result'
        save_file = os.path.join(self.args['RESULT_SAVE_DIR'], file_name+'.pkl') 
        result_dict = dict(zip(self.label,self.result))
        with open(save_file, 'wb') as handle:
            pickle.dump(result_dict, handle)
            
    def load_model(self):
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
            for idx in range(len(self.model.models)):
                
                
                path = os.path.join(self.args['TEST']['MODEL_FOLDER'],
                                      self.args['TEST'][f'MODEL_NAME_{idx}'])
                
                checkpoint = torch.load(path, map_location=f'cuda:{self.cuda}')
                self.model.models[idx].load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.optimizer[idx].load_state_dict(checkpoint['optimizer_state_dict'])
                self.meta_info['epoch'] = checkpoint['epoch']
                self.epoch_info['mean_loss'] = checkpoint['loss_value']
                self.loss = checkpoint['loss']
                
#                self.model.models[idx].load_state_dict(torch.load(model_path, 
#                                 map_location=f'cuda:{self.cuda}'), 
#                                 strict=True)
                self.model.models[idx].to(self.cuda)
                self.model.models[idx].eval()
        else:
            
            path = os.path.join(self.args['TEST']['MODEL_FOLDER'],
                                  self.args['TEST'][f'MODEL_NAME'])
            
            checkpoint = torch.load(path, map_location=f'cuda:{self.cuda}')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.meta_info['epoch'] = checkpoint['epoch']
            self.epoch_info['mean_loss'] = checkpoint['loss_value']
            self.loss = checkpoint['loss']
            
#            self.model.load_state_dict(torch.load(model_path, map_location=f'cuda:{self.cuda}'), 
#                                       strict=True)
            self.model.to(self.cuda)
            self.model.eval()                     
        
    def warm_start(self):
        self.load_model()
        self.test()
        if self.args['MODEL']['TYPE'] == 'vscnn_vgf':
            for idx in range(len(self.model.models)):
                self.model.models[idx].train()
        else:
            self.model.train()   
        

    # def save_best_feature(self, _features_file, _save_file, data, joints, coords):
    #     if self.best_epoch is None:
    #         self.best_epoch, best_accuracy = get_best_epoch_and_accuracy(
    #             self.args['WORK_DIR'])
    #     else:
    #         best_accuracy = self.best_accuracy.item()
    #     filename = os.path.join(self.args['WORK_DIR'],
    #                             'epoch{}_acc{:.2f}_model.pth.tar'.format(self.best_epoch, best_accuracy))
    #     self.model.load_state_dict(torch.load(filename))
    #     features = np.empty((0, 256))
    #     fCombined = h5py.File(_features_file, 'r')
    #     fkeys = fCombined.keys()
    #     dfCombined = h5py.File(_save_file, 'w')
    #     for i, (each_data, each_key) in enumerate(zip(data, fkeys)):

    #         # get data
    #         each_data = np.reshape(
    #             each_data, (1, each_data.shape[0], joints, coords, 1))
    #         each_data = np.moveaxis(each_data, [1, 2, 3], [2, 3, 1])
    #         each_data = torch.from_numpy(each_data).float().to(self.cuda)

    #         # get feature
    #         with torch.no_grad():
    #             _, feature = self.model(each_data)
    #             fname = [each_key][0]
    #             dfCombined.create_dataset(fname, data=feature.cpu())
    #             features = np.append(features, np.array(
    #                 feature.cpu()).reshape((1, feature.shape[0])), axis=0)
    #     dfCombined.close()
    #     return features

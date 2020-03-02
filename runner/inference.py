# -*- coding: utf-8 -*-

import os
import copy
import pickle
import numpy as np
import torch
from tqdm import tqdm
import yaml
from datetime import datetime

from trainer import Trainer
from utils.torch_utils import SummaryStatistics

import matplotlib.pyplot as plt

class Inference():
    def __init__(self, args, data_loader, num_classes, graph_dict, model_kwargs):
        self.num_classes = num_classes
        self.args = args
        
        if self.args['MODEL']['TYPE'] == 'vscnn':
            self.summary_statistics = SummaryStatistics(self.num_classes * model_kwargs['NUM_GROUPS'])
            
            args1 = copy.deepcopy(self.args)
            args1['MODEL']['TYPE'] = 'vscnn_vgp'
            self.trainer1 = Trainer(args1, data_loader, num_classes, graph_dict, model_kwargs)
            self.trainer1.load_model()
            
            args2 = copy.deepcopy(self.args)
            args2['MODEL']['TYPE'] = 'vscnn_vgf'
            self.trainer_vgf = Trainer(args2, data_loader, num_classes, graph_dict, model_kwargs)
            self.trainer_vgf.load_model()
            
            self.args['RESULT_SAVE_DIR'] = self.trainer1.args['RESULT_SAVE_DIR']
        else:
            self.trainer1 = Trainer(args, data_loader, num_classes, graph_dict, model_kwargs)
            self.trainer1.load_model()
        
        now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.args['RESULT_SAVE_DIR'] = os.path.join(args['OUTPUT_PATH'], 'test_result', args['MODEL']['TYPE'], now)
        self.trainer1.create_working_dir(self.args['RESULT_SAVE_DIR'])
 
    
    def inference(self):
        if self.args['MODEL']['TYPE'] == 'vscnn':
            self.per_test_vscnn()
            self.result_summary = self.summary_statistics.get_metrics()
            print(self.result_summary)
            file_name = 'test_result'
            save_file = os.path.join(self.args['RESULT_SAVE_DIR'], file_name+'.pkl') 
            save_file_summary = os.path.join(self.args['RESULT_SAVE_DIR'], file_name+'_summary.txt') 
            save_file_confusion = os.path.join(self.args['RESULT_SAVE_DIR'], file_name+'_confusion.csv') 
            plt.matshow(self.result_summary['conf_matrix'])
            np.savetxt(save_file_confusion, self.result_summary['conf_matrix'], delimiter = ',')
            result_dict = dict(zip(self.label,self.result))
            with open(save_file, 'wb') as handle:
                pickle.dump(result_dict, handle)
            with open(save_file_summary, 'w') as handle:
                handle.write(str(self.result_summary))
            print(self.result_summary)
        else:
            self.trainer1.test()
            print(self.trainer1.result_summary)
            plt.matshow(self.trainer1.result_summary['conf_matrix'])
            
            
    def per_test_vscnn(self, evaluation=True):
        # put models in eval mode
        for _model in self.trainer_vgf.model.models:
            _model.eval()
        self.trainer1.model.eval()
        
        loader = self.trainer1.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        group_frag = []
        label_16_true_frag = []
        label_16_pred_frag = []
        
        self.summary_statistics.reset()
        
        for data, label_and_group in tqdm(loader):
            with torch.no_grad():
                # get data 
                data_vgp = data.float().to(self.trainer1.cuda)
                data_vgf = data.float()
                label_true = label_and_group[0].long() # emotion label
                group_true = label_and_group[1].long().numpy() # view angle group
                # predict group
                group_pred = self.trainer1.model(data_vgp, apply_sfmax = True)
                group_pred = group_pred.data.cpu().numpy()
                group_pred = np.argmax(group_pred, axis=1)
                # inference
                for index in range(len(self.trainer_vgf.model.models)):
                    # get index of specific group
                    req_index = np.where(group_pred == index)[0]
                    if req_index.size > 0:
                        # get data according to group
                        group_data = data_vgf[req_index].to(self.trainer1.cuda)
                        group_label = label_true[req_index].to(self.trainer1.cuda)
                        # inference
                        output = self.trainer_vgf.model.models[index](group_data, apply_sfmax = True)
                        result = output.data.cpu().numpy()
                        result_frag.append(result)
        
                        # get loss
                        if evaluation:
                            loss = self.trainer_vgf.loss(output, group_label)
                            loss_value.append(loss.item())
                            group_label = group_label.data.cpu().numpy()
                            label_frag.append(group_label)
                            group_frag.append([index]*req_index.size)
                            label_16_true_frag.append(self.num_classes*group_label + group_true[req_index])
                            label_16_pred_frag.append(self.num_classes*result.argsort()[:,-1] + index)

        self.result = np.concatenate(result_frag)
        
        label_16_true_frag = np.concatenate(label_16_true_frag)
        label_16_pred_frag = np.concatenate(label_16_pred_frag)
        
        if evaluation:
            self.group = np.concatenate(group_frag)
            self.label = np.concatenate(label_frag)
            print(f'MEAN LOSS : {np.mean(loss_value):0.4f}')
            # show top-k accuracy
            for k in self.args['TOPK']:
                rank = self.result.argsort()
                hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
                accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
                print('\tTop{}: {:.2f}%'.format(k, accuracy))
                self.summary_statistics.update(label_16_true_frag,label_16_pred_frag)
                
        

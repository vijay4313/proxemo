# -*- coding: utf-8 -*-

import os
import copy
import pickle
import numpy as np
import torch
from tqdm import tqdm

from trainer import Trainer

class Inference():
    def __init__(self, args, data_loader, num_classes, graph_dict, model_kwargs):
        self.args = args
        if self.args['MODEL']['TYPE'] == 'vscnn':
            
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
    
    def inference(self):
        if self.args['MODEL']['TYPE'] == 'vscnn':
            self.per_test_vscnn()
            file_name = 'test_result'
            save_file = os.path.join(self.args['RESULT_SAVE_DIR'], file_name+'.pkl') 
            result_dict = dict(zip(self.label,self.result))
            with open(save_file, 'wb') as handle:
                pickle.dump(result_dict, handle)
            
        else:
            self.trainer1.test()
            
    def per_test_vscnn(self, evaluation=True):
        # put models in eval mode
        for _model in self.trainer_vgf.model.models:
            _model.eval()
        self.trainer1.model.eval()
        
        loader = self.trainer1.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label_and_group in tqdm(loader):
            with torch.no_grad():
                # get data 
                data_vgp = data.float().to(self.trainer1.cuda)
                data_vgf = data.float()
                label = label_and_group.long() # emotion label
                # predict group
                group = self.trainer1.model(data_vgp, apply_sfmax = True)
                group = group.data.cpu().numpy()
                group = np.argmax(group, axis=1)
                # inference
                for index in range(len(self.trainer_vgf.model.models)):
                    # get index of specific group
                    req_index = np.where(group == index)[0]
                    if req_index.size > 0:
                        # get data according to group
                        group_data = data_vgf[req_index].to(self.trainer1.cuda)
                        group_label = label[req_index].to(self.trainer1.cuda)
                        # inference
                        output = self.trainer_vgf.model.models[index](group_data, apply_sfmax = True)
                        result_frag.append(output.data.cpu().numpy())
        
                        # get loss
                        if evaluation:
                            loss = self.trainer_vgf.loss(output, group_label)
                            loss_value.append(loss.item())
                            label_frag.append(group_label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        
        if evaluation:
            self.label = np.concatenate(label_frag)
            print(f'MEAN LOSS : {np.mean(loss_value):0.4f}')
            # show top-k accuracy
            for k in self.args['TOPK']:
                rank = self.result.argsort()
                hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
                accuracy = 100. * sum(hit_top_k) * 1.0 / len(hit_top_k)
                print('\tTop{}: {:.2f}%'.format(k, accuracy))
                
        

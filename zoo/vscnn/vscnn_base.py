from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ViewGroupPredictor(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.layer_channels = [16, 32, 32]
        self.num_classes = n_classes
        self.build_net()

    def _gen_layer_name(self, layer_type, layer_num=''):
        name = '_'.join([self.__class__.__name__, layer_type, str(layer_num)])
        return name
    
    def build_net(self):
        layer1 = nn.Conv2d(
            self.in_channels, self.layer_channels[0], (3, 3), (1, 1))
        layer2 = nn.Conv2d(
            self.layer_channels[0], self.layer_channels[1], (3, 3), (1, 1))
        layer3 = nn.Conv2d(
            self.layer_channels[1], self.layer_channels[2], (3, 3), (2, 2))

        self.conv_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name('conv', 1), layer1),
            (self._gen_layer_name('conv', 2), layer2),
            (self._gen_layer_name('conv', 3), layer3)
        ]))

        self.final_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name('fc', 1), nn.Linear(453152, 128)),
            (self._gen_layer_name('relu'), nn.ReLU()),
            (self._gen_layer_name('fc', 2), nn.Linear(128, self.num_classes)),
            (self._gen_layer_name('softmax'), nn.Softmax())
        ]))

    def forward(self, input_tensor):
        # convert [N, H, W, C] to [N, C, H, W]
        input_tensor_reshaped = torch.empty((input_tensor.shape[0],
                                             input_tensor.shape[3],
                                             input_tensor.shape[1],
                                             input_tensor.shape[2]), device = input_tensor.device)
        input_tensor_reshaped[:, 0, :, :] =input_tensor[:, :, :, 0]
        input_tensor_reshaped[:, 1, :, :] =input_tensor[:, :, :, 1]
        input_tensor_reshaped[:, 2, :, :] =input_tensor[:, :, :, 2]
        
        # forward pass
        conv_out = self.conv_layers(input_tensor_reshaped)
        conv_out = conv_out.view((input_tensor_reshaped.shape[0],-1))
        final_out = self.final_layers(conv_out)

        return final_out, None


class SkCnn(nn.Module):
    def __init__(self, n_classes, in_channels, dropout, layer_channels=[32, 64]):
        super().__init__()
        self.in_channels = in_channels
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.num_classes = n_classes
        self.build_net()

    def _gen_layer_name(self, stage, layer_type, layer_num=''):
        name = '_'.join([self.__class__.__name__, 'stage', str(stage), layer_type, str(layer_num)])
        return name

    def build_net(self):
        conv1_1 = nn.Conv2d(self.in_channels, self.layer_channels[0], (3, 3))
        conv1_2 = nn.Conv2d(self.layer_channels[0], self.layer_channels[0], (3, 3))
        bn1 = nn.BatchNorm2d(self.layer_channels[0])


        conv2_1 = nn.Conv2d(
            self.layer_channels[0], self.layer_channels[1], (3, 3))
        conv2_2 = nn.Conv2d(
            self.layer_channels[1], self.layer_channels[1], (3, 3))
        bn2 = nn.BatchNorm2d(self.layer_channels[1])
        
        max_pool = nn.MaxPool2d((3, 3), (2, 2))

        dropout = nn.Dropout(self.dropout)

        self.conv_stage_1 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(1, 'conv', 1), conv1_1),
            (self._gen_layer_name(1, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 1), max_pool),
            (self._gen_layer_name(1, 'conv', 2), conv1_2),
            (self._gen_layer_name(1, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 2), max_pool),
            (self._gen_layer_name(1, 'bn'), bn1),
            (self._gen_layer_name(1, 'drop'), dropout)
        ]))

        self.conv_stage_2 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(2, 'conv', 1), conv2_1),
            (self._gen_layer_name(2, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(2, 'maxpool', 1), max_pool),
            (self._gen_layer_name(2, 'conv', 2), conv2_2),
            (self._gen_layer_name(2, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(2, 'maxpool', 2), max_pool),
            (self._gen_layer_name(2, 'bn'), bn2),
            (self._gen_layer_name(2, 'drop'), dropout)
        ]))

        self.final_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name(3, 'fc', 1), nn.Linear(9216, 128)),
            (self._gen_layer_name(3, 'relu'), nn.ReLU()),
            (self._gen_layer_name(3, 'fc', 2), nn.Linear(128, self.num_classes)),
            (self._gen_layer_name(3, 'softmax'), nn.Softmax())
        ]))

    def forward(self, input_tensor):
        
        # convert [N, H, W, C] to [N, C, H, W]
        input_tensor_reshaped = torch.empty((input_tensor.shape[0],
                                             input_tensor.shape[3],
                                             input_tensor.shape[1],
                                             input_tensor.shape[2]), device = input_tensor.device)
        input_tensor_reshaped[:, 0, :, :] =input_tensor[:, :, :, 0]
        input_tensor_reshaped[:, 1, :, :] =input_tensor[:, :, :, 1]
        input_tensor_reshaped[:, 2, :, :] =input_tensor[:, :, :, 2]
        first_conv = self.conv_stage_1(input_tensor_reshaped)
        second_conv = self.conv_stage_2(first_conv)
        second_conv = second_conv.view((input_tensor_reshaped.shape[0],-1))
        final = self.final_layers(second_conv)

        return final

class ViewGroupFeature(nn.Module):
    def __init__(self, in_channels, n_classes, num_groups, dropout, layer_channels):
        super().__init__()
        self.n_groups = num_groups
        self.in_channels = in_channels
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.n_classes = n_classes
        self.build_net()

    def build_net(self):
        self.models = [SkCnn(self.n_classes, self.in_channels,
                             self.dropout, self.layer_channels) for _ in range(self.n_groups)]

    def forward(self, input_tensor, view_group):
        view_group = np.asarray(view_group)
        outputs = []
        for group in range(self.n_groups):
            selected_indices = np.where(view_group == group)
            if selected_indices.size > 0:
                tensor = input_tensor[selected_indices, :, : , :]
                outputs.append(self.models[group](tensor))
            else:
                outputs.append(None)
        return outputs

    def forward_all(self, input_tensor):
        out_all = []
        for model in self.models:
            out_all.append(model(input_tensor))
        return out_all

class ChannelFusion(nn.Module):
    def __init__(self, num_groups, n_classes):
        self.n_groups = num_groups
        self.n_classes = n_classes

    def build_net(self):
        self.conv1d = nn.Conv2d(self.n_groups, 1, 1, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, tensor_list):
        tensors_stacked = torch.stack(tensor_list, 1).unsqueeze(-1)
        conv_out = self.conv1d(tensors_stacked)
        conv_out = conv_out.squeeze()
        out = self.softmax()

        return out
        

from collections import OrderedDict

import torch
import torch.nn as nn
import torchsummary as summary

import numpy as np


class VSGCNN(nn.Module):
    def __init__(self, n_classes, in_channels, num_groups, dropout=0.2, layer_channels=[32, 64, 16]):
        super().__init__()
        self.in_channels = in_channels
        self.n_groups = num_groups
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.num_classes = n_classes
        self.build_net()

    def _gen_layer_name(self, stage, layer_type, layer_num=''):
        name = '_'.join([self.__class__.__name__, 'stage',
                         str(stage), layer_type, str(layer_num)])
        return name

    def build_net(self):
        conv1_0 = nn.Conv2d(self.in_channels,
                            self.layer_channels[0],
                            (3, 3))
        conv1_1 = nn.Conv2d(self.layer_channels[0],
                            self.layer_channels[0]*self.n_groups,
                            (3, 3))
        conv1_2 = nn.Conv2d(self.layer_channels[0]*self.n_groups,
                             self.layer_channels[0]*self.n_groups,
                             (3, 3), groups=self.n_groups)
        bn1 = nn.BatchNorm2d(self.layer_channels[0]*self.n_groups)

        conv2_1 = nn.Conv2d(self.layer_channels[0]*self.n_groups,
                            self.layer_channels[1]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        
        conv2_2= nn.Conv2d(self.layer_channels[1]*self.n_groups,
                            self.layer_channels[1]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        bn2 = nn.BatchNorm2d(self.layer_channels[1]*self.n_groups)

        max_pool = nn.MaxPool2d((3, 3), (2, 2))

        dropout = nn.Dropout(self.dropout)

        self.conv_stage_1 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(1, 'conv', 0), conv1_0),
            (self._gen_layer_name(1, 'relu', 0),nn.ReLU()),
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

        conv3_1 = nn.Conv2d(self.layer_channels[1]*self.n_groups,
                            self.layer_channels[2]*self.n_groups,
                            (3, 3), groups=self.n_groups)
        conv3_2 = nn.Conv2d(self.layer_channels[2]*self.n_groups,
                            1,(3, 3), stride = (2, 2))

        self.final_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name(3, 'conv', 1), conv3_1),
            (self._gen_layer_name(3, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(3, 'conv', 2), conv3_2),
            (self._gen_layer_name(3, 'relu', 2), nn.ReLU())
        ]))
        self.softmax = nn.Softmax(2)

    def forward(self, input_tensor, apply_sfmax=True):

        # convert [N, H, W, C] to [N, C, H, W]
        if input_tensor.size(1) != self.in_channels:
            input_tensor = input_tensor.permute(0, 3, 2, 1)
        first_conv = self.conv_stage_1(input_tensor)
        second_conv = self.conv_stage_2(first_conv)
        final_layers = self.final_layers(second_conv)

        if apply_sfmax:
            final_layers = self.softmax(final_layers.view(*final_layers.size()[:2], -1))
            final_layers = final_layers.squeeze(1)

        return final_layers


if __name__ == "__main__":
    vgcnn = VSGCNN(4, 3, 4, 0.2)
    image = torch.rand(1, 3, 244, 244)
    a = vgcnn(image)
    print(sum([param.nelement() for param in vgcnn.parameters()]))
    print(a.data)

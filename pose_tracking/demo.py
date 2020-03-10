#!/usr/bin/env python3

import argparse
import time
import cv2
import numpy as np
import torch
import copy

from human_tracking_3D import Track_Human_Pose


# from zoo.vs_gcnn.vs_gcnn import VSGCNN



# ---------------------------------------------------------------------


from collections import OrderedDict

import torch
import torch.nn as nn

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

    def forward(self, input_tensor, apply_sfmax=False):

        # convert [N, H, W, C] to [N, C, H, W]
        if input_tensor.shape[1] != self.in_channels:
            input_tensor = input_tensor.permute(0, 3, 2, 1)
        first_conv = self.conv_stage_1(input_tensor)
        second_conv = self.conv_stage_2(first_conv)
        final_layers = self.final_layers(second_conv)

        if apply_sfmax:
            final_layers = self.softmax(final_layers.view(*final_layers.size()[:2], -1))
            final_layers = final_layers.squeeze(1)
        else:
            final_layers = final_layers.view(*final_layers.size()[:2], -1).squeeze(1)

        return final_layers


# ---------------------------------------------------------------------










PARSER = argparse.ArgumentParser(description="Emotion Classification demo.")

PARSER.add_argument("-m", "--model", type=str,
                    default="vsgcnn",
                    help="Model to be used.")
PARSER.add_argument("-l", "--load_path", type=str,
                    default="/home/emotiongroup/Desktop/models/vsgcnn_model.tar",
                    help="Trained Model parameters path.")
PARSER.add_argument("-i", "--input", type=str,
                    default="",
                    help="Path to bag files to run demo on.")
PARSER.add_argument("-d", "--cuda", type=int, default=0, help="Cuda device ID.")

MODEL_MAP = {
    'vsgcnn':   VSGCNN
}

EMOTION_MAP = {
    0: 'Angry',
    1: 'Angry',
    2: 'Angry',
    3: 'Angry',
    4: 'Happy',
    5: 'Happy',
    6: 'Happy',
    7: 'Happy',
    8: 'Sad',
    9: 'Sad',
    10: 'Sad',
    11: 'Sad',
    12: 'Neutral',
    13: 'Neutral',
    14: 'Neutral',
    15: 'Neutral'
}

def load_model(model, load_path):  
    """Load model params from training."""           
    checkpoint = torch.load(load_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except:
        model.load_state_dict(checkpoint, strict=True)

    return model


def main():
    args = PARSER.parse_args()
    model = MODEL_MAP[args.model](4, 3, 4, 0.2)
    
    model = load_model(model, args.load_path)
    model.to(args.cuda).eval()

    track_pose = Track_Human_Pose(display=True)
    skel_track = []
    while True:
        time.sleep(0.3)
        track_pose.get_pose()
        if track_pose.display:
            key = cv2.waitKey(1) & 0xFF
            # press the 'q' key to stop the video stream
            if key == ord("q"):
                break
        if track_pose.cubemos.skel3d_np.shape[0] > 0:
            print(len(skel_track))
            if len(skel_track) < 75:
                skel_track.append(track_pose.cubemos.skel3d_np[0])
            else:
                skel_track.append(track_pose.cubemos.skel3d_np[0])
                skel_track.pop(0)
                
                skel_track_np = np.asarray(skel_track)
                skel_track_np_nr = copy.deepcopy(skel_track_np)
                skel_track_np_nr = skel_track_np_nr - np.expand_dims(skel_track_np_nr[:,1,:], axis=1)
                skel_track_np_nr = np.transpose(skel_track_np_nr, (1, 0, 2))
                img = cv2.resize(skel_track_np_nr, (244, 244))
                
                kernel = np.ones((5,5),np.float32)/25
                img = cv2.filter2D(img,-1,kernel)
                
                print(skel_track_np.shape)
                print(img.shape)
                img_cv = copy.deepcopy(img)
                img_cv = img_cv.astype(np.uint8)
                cv2.imshow("skel-embedding", img_cv)
                img = np.expand_dims(img, axis=0)
                img = torch.tensor(img).float().to((args.cuda))
                emotion = EMOTION_MAP[model(img, apply_sfmax=True).detach().cpu().numpy().argmax()]
                print(emotion)


if __name__ == "__main__":
    main()

# a = vgcnn(image)
# print(sum([param.nelement() for param in vgcnn.parameters()]))
# print(a.data)

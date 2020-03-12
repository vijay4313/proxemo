#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import torch

from human_tracking_3D import Track_Human_Pose
from zoo.vs_gcnn.vs_gcnn import VSGCNN

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

def get_parameters(args):
    params = []
    if args.model == 'vsgcnn':
        # def __init__(self, n_classes, in_channels, num_groups,
        #              dropout=0.2, layer_channels=[32, 64, 16]):
        params = [4, 3, 4, 0.2]
    return params

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
    params = get_parameters(args)
    
    model = MODEL_MAP[args.model](*params)
    model = load_model(model, args.load_path)
    model.to(args.cuda).eval()

    track_pose = Track_Human_Pose(display=True)
    
    while True:
        # get pose, track the pose and get embedding
        print("-------------------")
        track_pose.get_pose()
        track_pose.track_pose()
        imgs, ids = track_pose.skel_tracker.get_embedding()
        # exit check
        if track_pose.display:
            
            key = cv2.waitKey(300) & 0xFF
            # press the 'q' key to stop the video stream
            if key == ord("q"):
                break
        # estimate emotion
        if len(ids) > 0 :
            track_pose.skel_tracker.display_embedding()
            imgs = torch.tensor(imgs).float().to((args.cuda))
            print(model(imgs, apply_sfmax=True).detach().cpu().numpy())
            pred = model(imgs, apply_sfmax=True).detach().cpu().numpy().argmax(axis = 1)
            print(pred)
            if len(ids) > 0:
                emotion = [EMOTION_MAP[emo] for emo in pred]
            else:
                emotion = EMOTION_MAP[pred]
            print(emotion)
            print(ids)


if __name__ == "__main__":
    main()

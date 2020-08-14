#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import os
import argparse
import numpy as np
import cv2
import torch
from emotion_classification.utils.yaml_parser import yaml_parser
from emotion_classification.runner.trainer import Trainer
from pose_tracking.human_tracking_3D import Track_Human_Pose
from emotion_classification.modeling.vs_gcnn import VSGCNN


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


def arg_parser():
    """CLI arg parser."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification demo.")

    parser.add_argument("-m", "--model",
                        default="./emotion_classification/modeling/config/infer.yaml",
                        help="Model config to be used.",
                        type=os.path.abspath)
    parser.add_argument("-l", "--load_path",
                        default="models/vsgcnn_model.tar",
                        help="Trained Model parameters path.",
                        type=os.path.abspath)
    parser.add_argument("-i", "--input", type=str,
                        default="",
                        help="Path to bag files to run demo on.")
    parser.add_argument("-d", "--cuda", type=int,
                        default=0, help="Cuda device ID.")
    args = parser.parse_args()

    return args


def get_model(config_file):
    """Get a Model from config file."""
    dir_path, filename = os.path.split(config_file)
    config = yaml_parser(filename, dir_path)
    model_config = config['MODEL']
    model_obj = Trainer(None, None, model_config).model
    return model_obj


def main():
    """Main demo routine."""
    args = arg_parser()
    model = get_model(args.model)
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
        if len(ids) > 0:
            track_pose.skel_tracker.display_embedding()
            imgs = torch.tensor(imgs).float().to((args.cuda))
            print(model(imgs, apply_sfmax=True).detach().cpu().numpy())
            pred = model(imgs, apply_sfmax=True).detach(
            ).cpu().numpy().argmax(axis=1)
            print(pred)
            if len(ids) > 0:
                emotion = [EMOTION_MAP[emo] for emo in pred]
            else:
                emotion = EMOTION_MAP[pred]
            print(emotion)
            print(ids)


if __name__ == "__main__":
    main()

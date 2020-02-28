#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:32:39 2019

@author: bala
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob
import re

from sklearn.model_selection import train_test_split

from transform3DPose import augment3D
from dataGenerator import readDataSingleGait
import yaml_parser


def plotSkeleton3D(data_features, label_features, fn = 0):
    "Plotting the skeleton in 3D for viewing. The human is standing along the Y - axis"
    "Order: 1-Root, 2-Spine, 3-Neck, 4-Head, 5-Left Shoulder, 6-Left Elbow, 7- Left Hand, 8- Right Shoulder"
    "     : 9-Right Elbow, 10- Right Hand, 11-Left Thigh, 12-Left Knee, 13-Left Foot, 14-Right Thigh, 15-Right Knee, 16-Right Foot"
    
    """
    1-Root,
     2-Spine,
     3-Neck,
     4-Head,
     5-Left Shoulder,
     6-Left Elbow,
     7- Left Hand,
     8- Right Shoulder
     9- Right Elbow
     10- Right Hand
     11-Left Thigh,
     12-Left Knee,
     13-Left Foot,
     14-Right Thigh,
     15-Right Knee,
     16-Right Foot"
    
    4,3,3,5,6,3,8,9 ,2,2 ,11,12,2 ,14,15
    3,2,5,6,7,8,9,10,1,11,12,13,14,15,16
    
    3,2,2,4,5,2,7,8,1,1 ,10,11,1 ,13,14
    2,1,4,5,6,7,8,9,0,10,11,12,13,14,15
    """
    plt.ion()
    fig3D = plt.figure(fn)
    ax = fig3D.add_subplot(111 , projection='3d')
    x = [-0.5 , 0.5]
    y = [-0.5 , 0.5]
    z = [-0.5 , 0.5]
    sp = ax.scatter(0, 0, 0)
    sp = ax.scatter(x, y, z)
    if label_features == 1:
        emotion = "happy"
    elif label_features == 2:
        emotion = "anger"
    elif label_features == 3:
        emotion = "sad"
    else:
        emotion = "neutral"
    ax.set_title(emotion)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    
    for ind in range(len(data_features)):
        skeletal_data = np.reshape(data_features[ind],(-1, 3))
        x = skeletal_data[:,0]
        y = skeletal_data[:,2]
        z = skeletal_data[:,1]
        sp._offsets3d = (x,y,z)
        plt.draw()
        plt.pause(0.05)
        plt.savefig('../temp/fig4')
        
def plotSkeleton3D_dataloader(data_features, label_features, fn = 0):
    plt.ion()
    fig3D = plt.figure(fn)
    ax = fig3D.add_subplot(111 , projection='3d')
    x = [-0.5 , 0.5]
    y = [-0.5 , 0.5]
    z = [-0.5 , 0.5]
    sp = ax.scatter(0, 0, 0)
    sp = ax.scatter(x, y, z)
    if label_features == 1:
        emotion = "happy"
    elif label_features == 2:
        emotion = "anger"
    elif label_features == 3:
        emotion = "sad"
    else:
        emotion = "neutral"
    ax.set_title(emotion)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    
    for ind in range(data_features.shape[1]):
        sp._offsets3d = (data_features[0,ind,:,:][:,0],
                         data_features[2,ind,:,:][:,0],
                         data_features[1,ind,:,:][:,0])
        plt.draw()
        plt.pause(0.1)
        
def disp_skeleton(settings_file, sample_idx):
    args = yaml_parser.yaml_parser(settings_file)
    model_args = args['MODEL']
    data_args = args["DATA"]
    args['YAML_FILE_NAME'] = 'stgcn'
    
    data, labels, angles, folders = \
        load_data_multiview(data_args['FEATURES_FILE'],
                            data_args['LABELS_FILE'],
                            data_args['COORDS'],
                            data_args['JOINTS'],
                            cycles=data_args['CYCLES'],
                            sample_idx = sample_idx)
        
    data = np.reshape(data, (data.shape[0], data.shape[1], data_args['JOINTS'], data_args['COORDS'], 1))
    data = np.moveaxis(data, [1, 2, 3], [2, 3, 1])
    
    for data_numpy, l , a, f in zip(data, labels, angles, folders):
        data_max = np.max(data_numpy, (1,2,3))
        data_min = np.min(data_numpy, (1,2,3))
        img_data = np.zeros((data_numpy.shape[1],
                             data_numpy.shape[2],
                             data_numpy.shape[0]))
    
        img_data[:,:,0] = (data_max[0] - data_numpy[0,:,:,0]) * ( 255 / (data_max[0] - data_min[0]) )
        img_data[:,:,1] = (data_max[1] - data_numpy[1,:,:,0]) * ( 255 / (data_max[1] - data_min[1]) )
        img_data[:,:,2] = (data_max[2] - data_numpy[2,:,:,0]) * ( 255 / (data_max[2] - data_min[2]) )
        
        img_data[:,:,0] = np.divide(img_data[:,:,0], img_data[:,:,2]+1e-9)
        img_data[:,:,1] = np.divide(img_data[:,:,1], img_data[:,:,2]+1e-9)
        img_data[:,:,2] = np.divide(img_data[:,:,2], img_data[:,:,2]+1e-9)
                        
        img_data = cv2.resize(img_data, (244,244))
#        cv2.imshow("testframe", img_data.astype(np.uint8))
#        cv2.waitKey(0)
        f_name = os.path.abspath(f'../temp/test/{f}_{int(l)}_{a}.png')
        cv2.imwrite(f_name, img_data)
    
def load_data_multiview(_path_features, _path_lables, coords, joints, cycles=3, sample_idx = [1,2,3]):
    feature_files = glob.glob(_path_features)
    label_files = glob.glob(_path_lables)
    print(f'---> Number of files = {len(feature_files)}')
    # sorting files so that features and labels files match
    feature_files.sort()
    label_files.sort()
    
    angle_regex = re.compile('(\d*).h5')
    folder_regex = re.compile('(\w*)\/')
    
    all_data = []
    all_labels = []
    all_angles = []
    all_folders = []

    for feature_file, label_file in zip(feature_files, label_files):
        ff = h5py.File(feature_file, 'r')
        fl = h5py.File(label_file, 'r')
        angle = int(angle_regex.search(feature_file).group(1))
        folder = folder_regex.findall(feature_file)[-1]
        print(f"--->> processing - {folder} - {angle}")
        
        data_list = []
        num_samples = len(sample_idx)
        time_steps = 0
        labels = np.empty(num_samples)
        for idx, si in enumerate(sample_idx):
            ff_group_key = list(ff.keys())[si]
            data_list.append(list(ff[ff_group_key]))  # Get the data
            time_steps_curr = len(ff[ff_group_key])
            if time_steps_curr > time_steps:
                time_steps = time_steps_curr
            labels[idx] = fl[list(fl.keys())[si]][()]
    
        data = np.empty((num_samples, time_steps*cycles, joints*coords))
        for si in range(num_samples):
            data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
            for ci in range(cycles):
                data[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
        
        all_data.extend(data)
        all_labels.extend(labels)
        all_angles.extend([angle]*len(labels))
        all_folders.extend([folder]*len(labels))
        
    return np.asarray(all_data), all_labels, all_angles, all_folders

    
    
    
if __name__ == "__main__":
    pts,label = readDataSingleGait("../data", "", 3, 16, 1)
#    plotSkeleton3D(pts,label,0)
    for idx, param in enumerate([(pts, 270, 2, 1)]):
#                                 (pts, 90, 0, 1),
#                                 (pts, 180, 0, 1)]):
        pts1 = augment3D(*param)
        print(abs(pts1[0][0]-pts1[0][1]))
        plotSkeleton3D(pts1,label,idx)
#    disp_skeleton('vscnn_vgf', [1,15,100])

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

from transform3DPose import augment3D
from dataGenerator import readDataSingleGait


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

if __name__ == "__main__":
    pts,label = readDataSingleGait("../data/AugDataset", "_180", 3, 16, 1)
    plotSkeleton3D(pts,label,0)
# =============================================================================
#     for idx, param in enumerate([(pts, 90, [0, 0, 0], 3),
#                                  (pts, 0, [0, 0, 0], 1),
#                                  (pts, 0, [0, 0, 0], 1)]):
#         pts1 = augment3D(*param)
#         print(abs(pts1[0][0]-pts1[0][1]))
#         plotSkeleton3D(pts1,label,idx)
# =============================================================================

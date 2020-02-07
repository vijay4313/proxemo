#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:23:55 2020

@author: bala
"""

import h5py
import os
from tqdm import tqdm

from transform3DPose import augment3D

def readDataSingleGait(_path, _ftype, coords, joints, cycles=3, dataSetNumber = 0):
    file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')
    
    ff_group_key = list(ff.keys())[dataSetNumber]
    fl_group_key = list(fl.keys())[dataSetNumber]
    
    print(f"""    --> File Name : {ff_group_key}
    --> Number of files : {len(list(ff.keys()))}""")
    
    pts = list(ff[ff_group_key])
    label = fl[fl_group_key][()]
    
    return pts, label


def generateDataSet(_path, _ftype, coords, joints, cycles=3, angles = 0):
    file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')    
    
    pbarAngle = tqdm(angles,
                     desc = "Angle")
    for angle in pbarAngle:
#        pbarDataSet = tqdm(zip(ff.keys(), fl.keys()),
#                total = len(ff.keys()),
#                desc = "DataSet")
            
        ffNew = h5py.File(os.path.join(_path + "/AugDataset", 'features' + _ftype + '_' + str(angle) +'.h5'), 'w')
        flNew = h5py.File(os.path.join(_path + "/AugDataset", 'labels'   + _ftype + '_' + str(angle) +'.h5'), 'w')
        
#        for ffKey, flKey in pbarDataSet:
        for ffKey, flKey in zip(ff.keys(), fl.keys()):
            augData = augment3D(ff[ffKey][()], angle, 0, 1)
            ffNew.create_dataset(ffKey, data=augData)
            flNew.create_dataset(flKey, data=fl[ffKey][()])
    
        ffNew.close()
        flNew.close()
    
if __name__ == "__main__":
#    pts, label = readDataSingleGait("../data", "", 3, 16, 1)
    angles = range(0,360,5)
    generateDataSet("../data", "", 3, 16, 1, angles)
    





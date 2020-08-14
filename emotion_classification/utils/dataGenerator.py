#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================

import h5py
import os
from tqdm import tqdm

from transform3DPose import augment3D


def readDataSingleGait(_path, _ftype, coords, joints, cycles=3, dataSetNumber=0):
    """Reads a single gait sequence file.

    Args:
        _path (str): path to h5 file
        _ftype (str): Dataset sub-type
        coords (int): Number of co-ordinates representing each joint in gait cycle
        joints (int)): Number of joints in the gait sequence
        cycles (int, optional): Time duration of gait cycle. Defaults to 3.
        dataSetNumber (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
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


def generateDataSet(_path, _ftype, _dpath, coords, joints, cycles=3, angles=0, trans=0):
    """[summary]

    Args:
        _path (str): path to h5 file
        _ftype (str): Dataset sub-type
        _dpath (str): sub-directory path to augmented data
        coords (int): Number of co-ordinates representing each joint in gait cycle
        joints (int)): Number of joints in the gait sequence
        cycles (int, optional): Time duration of gait cycle. Defaults to 3.
        angles (int/list, optional): List of all augmented angles. Defaults to 0.
        trans (int/list, optional): Transition matrix to be applied for reference change.
                               Defaults to 0.
    """
    file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')
    if not os.path.exists(_dpath):
        os.makedirs(_dpath)
    pbarAngle = tqdm(angles,
                     desc="Angle")
    for angle in pbarAngle:
        #        pbarDataSet = tqdm(zip(ff.keys(), fl.keys()),
        #                total = len(ff.keys()),
        #                desc = "DataSet")

        ffNew = h5py.File(os.path.join(_dpath, 'features' +
                                       _ftype + '_' + str(angle) + '.h5'), 'w')
        flNew = h5py.File(os.path.join(_dpath, 'labels' +
                                       _ftype + '_' + str(angle) + '.h5'), 'w')

#        for ffKey, flKey in pbarDataSet:
        for ffKey, flKey in zip(ff.keys(), fl.keys()):
            augData = augment3D(ff[ffKey][()], angle, trans, 1)
            ffNew.create_dataset(ffKey, data=augData)
            flNew.create_dataset(flKey, data=fl[ffKey][()])

        ffNew.close()
        flNew.close()


if __name__ == "__main__":
    #    pts, label = readDataSingleGait("../data", "", 3, 16, 1)
    angles = [33, 66, 123, 156, 213, 246, 303, 336]
    generateDataSet("../data",
                    "_ELMD",
                    "../data/AugDataset_test-custom_ELMD_150_cm",
                    3, 16, 1, angles, 150)

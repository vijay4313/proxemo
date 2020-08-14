#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import csv
import os

import h5py
import numpy as np

savepath = 'C:/Users/Uttaran/Documents/Unity3D/Gait/Data/CVAEGCN/4D/Pos'

filename = 'features4DCVAEGCN.h5'
f = h5py.File(filename, 'r')
for idx in range(len(f.keys())):
    a_group_key = list(f.keys())[idx]
    data = np.array(f[a_group_key])  # Get the data
    np.savetxt(os.path.join(savepath, a_group_key+'.csv'),
               data, delimiter=',')  # Save as csv

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:34:31 2018

@author: xavier.qiu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime        
import scipy.misc

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

stage1_solution = pd.read_csv('stage1_solution.csv')
def getNameFromTime():
    now = datetime.datetime.now()
    return (str)(now.minute) + (str)(now.second) + (str)(now.microsecond)


for index, row in stage1_solution.iterrows():
    
    id_ = row['ImageId']
    if os.path.exists('stage1_test/'+id_):
        temp = rle_decode(rle = row['EncodedPixels'],shape=[row['Height'],row['Width']])
        path_temp = 'stage1_test/'+id_+'/masks'
        if not os.path.exists(path_temp):
            os.mkdir(path_temp)
        path_temp = 'stage1_test/'+id_+'/masks/'+getNameFromTime()
        scipy.misc.imsave(path_temp, temp)

#%%

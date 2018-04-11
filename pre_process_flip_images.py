#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:58:22 2018

@author: xavier.qiu
"""

# %%
import os
from tqdm import tqdm
from skimage.io import imread, imsave
import numpy as np
import datetime

problem_ids = list()
problem_ids.append('7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
problem_ids.append('b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe')
problem_ids.append('9bb6e39d5f4415bc7554842ee5d1280403a602f2ba56122b87f453a62d37c06e')
problem_ids.append('1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca')
problem_ids.append('58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921')
problem_ids.append('adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df')
problem_ids.append('12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40')
problem_ids.append('0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9')


def getNameFromTime():
    now = datetime.datetime.now()
    return (str)(now.minute) + (str)(now.second) + (str)(now.microsecond)


import warnings

warnings.filterwarnings("ignore")

IMG_CHANNELS = 3


os.system('rm -rf stage1_train_copy')
os.system('mkdir stage1_train_copy')
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

print('Getting and resizing train images and masks ... ')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    if id_ in problem_ids:
        continue

    path = TRAIN_PATH + id_
    image__ = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    mask_imgs = list()

    temp_imgs = next(os.walk(path + '/masks/'))[2]
    assert len(temp_imgs) > 0
    for mask in temp_imgs:
        mask_img = imread(path + '/masks/' + mask)
        mask_imgs.append(mask_img)



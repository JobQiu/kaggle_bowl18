#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:58:22 2018

@author: xavier.qiu
"""

#class PreProcess(object):
#    
#    def sliceImg(image_, mask_imgs, output_shape = (256,256), img_id = '', dirPath = ''):
#        if dirPath == '':
#            dirPath = os.getcwd()
#            
#        rows = (int)((image_.shape[0]-1) / output_shape[0])+1
#        cols = (int)((image_.shape[1]-1) / output_shape[1])+1
#        
#        for y in range(rows):
#            y_end = (y+1) * output_shape[0]
#            if y == rows - 1:
#                y_end = image_.shape[0]
#            for x in range(cols):
#                x_end = (x+1) * output_shape[1]
#                if x == cols - 1:
#                    x_end = image_.shape[1]
#                print(y_end)
#                print(x_end)
#                img_temp = img[y_end - output_shape[0]: y_end, x_end - output_shape[1]: x_end]
#                
#                valid_mask = list()
#                for mask in mask_imgs:
#                    mask_temp = mask[y_end - output_shape[0]: y_end, x_end - output_shape[1]: x_end]
#                    if np.any(mask_temp):
#                        valid_mask.append(mask_temp)
#                if len(valid_mask) == 0:
#                    continue
#                else:
#                    id__ = 'i' + getNameFromTime()
#                    os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__))
#                    os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__+'/images/'))
#                    os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__+'/masks/'))
#                    
#                    path___ = os.path.join(dirPath,'stage1_train_copy/'+id__+'/images/'+id__+'.png')
#                    imsave(path___,img_temp)
#                    for mask_ in valid_mask:
#                        mask_id = 'm'+ getNameFromTime()
#                        path__m = os.path.join(dirPath,'stage1_train_copy/'+id__+'/masks/'+mask_id+'.png')
#                        imsave(path__m,mask_)
#
#    
#    def __init__(self, 
#                 remove_errorness = False,
#                 shape_after_slice = (256,256),
#                 processed = False):
#        self.remove_errorness = remove_errorness
#        self.shape_after_slice = shape_after_slice
#        self.processed = processed
#        
#    def preprocess(self, preprocess_again = False):
#        
#        IMG_CHANNELS = 3
#
#        if self.processed and not preprocess_again:
#            pass
#        
##        os.system('rm -rf stage1_train_copy')
##        os.system('mkdir stage1_train_copy')
##        dir_util.copy_tree('stage1_train/','stage1_train_copy/')
#        
#        #1. 
#        
#        os.system('rm -rf stage1_train_copy')
#        os.system('mkdir stage1_train_copy')
#        
#        cwd = os.getcwd()
#        TRAIN_PATH = 'stage1_train/'
#        TEST_PATH = 'stage1_test/'
#        
#        train_ids = next(os.walk(TRAIN_PATH))[1]
#        test_ids = next(os.walk(TEST_PATH))[1]
#        
#        print('Getting and resizing train images and masks ... ')
#        
#        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#            path = TRAIN_PATH + id_
#            image__ = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
##            print(img.shape)
#            mask_imgs = list()
#            temp_imgs = next(os.walk(path+'/masks/'))[2]
#            assert len(temp_imgs) > 0
#            for mask in temp_imgs:
#                mask_img = imread(path+'/masks/'+mask)
#                mask_imgs.append(mask_img)
#            self.sliceImg(image__,mask_imgs,img_id = id_)
#            
#
#pp = PreProcess()
#pp.preprocess()

#%%
import os
from tqdm import tqdm
from skimage.io import imread, imsave
import numpy as np
import datetime

problem_ids = list()
'''problem_ids.append('7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
problem_ids.append('b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe')
problem_ids.append('9bb6e39d5f4415bc7554842ee5d1280403a602f2ba56122b87f453a62d37c06e')
problem_ids.append('1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca')
problem_ids.append('58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921')
problem_ids.append('adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df')
problem_ids.append('12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40')
problem_ids.append('0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9')
'''

def getNameFromTime():
    now = datetime.datetime.now()
    return (str)(now.minute)+(str)(now.second) + (str)(now.microsecond)
import warnings
warnings.filterwarnings("ignore")


IMG_CHANNELS = 3
    
def sliceImg(img, mask_imgs, output_shape = (256,256), img_id = '', dirPath = ''):
    
    """
    slice an image into several 256 * 256, 
    for example, for a 360*360 image, we will divide it into 4 subimages to feed to model
    cause compression will loss some information that we can make use of, so 
    """
    
    if dirPath == '':
        dirPath = os.getcwd()
        
    rows = (int)((img.shape[0]-1) / output_shape[0])+1
    cols = (int)((img.shape[1]-1) / output_shape[1])+1
    
    for y in range(rows):
        y_end = (y+1) * output_shape[0]
        if y == rows - 1:
            y_end = img.shape[0]
        for x in range(cols):
            x_end = (x+1) * output_shape[1]
            if x == cols - 1:
                x_end = img.shape[1]
#            print(y_end)
#            print(x_end)
            img_temp = img[y_end - output_shape[0]: y_end, x_end - output_shape[1]: x_end]
#            plt.figure()
#            plt.imshow(img_temp)
            
            valid_mask = list()
            for mask in mask_imgs:
                mask_temp = mask[y_end - output_shape[0]: y_end, x_end - output_shape[1]: x_end]
                if np.any(mask_temp):
                    valid_mask.append(mask_temp)
            if len(valid_mask) == 0:
                continue
            else:
                id__ = 'i' + getNameFromTime()
                os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__))
                os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__+'/images/'))
                os.mkdir(os.path.join(dirPath,'stage1_train_copy/'+id__+'/masks/'))
                
                path___ = os.path.join(dirPath,'stage1_train_copy/'+id__+'/images/'+id__+'.png')
                imsave(path___,img_temp)
                for mask_ in valid_mask:
                    mask_id = 'm'+ getNameFromTime()
                    path__m = os.path.join(dirPath,'stage1_train_copy/'+id__+'/masks/'+mask_id+'.png')
                    imsave(path__m,mask_)
                

os.system('rm -rf stage1_train_copy')
os.system('mkdir stage1_train_copy')
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]

print('Getting and resizing train images and masks ... ')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    if id_ in problem_ids:
        continue
    
    path = TRAIN_PATH + id_
    image__ = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    if image__.shape[0] <= 256 and image__.shape[1] <= 256:
        continue
    if image__.shape[0] < 256 or image__.shape[1] < 256:
        continue
    mask_imgs = list()
    
    
    temp_imgs = next(os.walk(path+'/masks/'))[2]
    assert len(temp_imgs) > 0
    for mask in temp_imgs:
        mask_img = imread(path+'/masks/'+mask)
        mask_imgs.append(mask_img)

    sliceImg(image__,mask_imgs,img_id = id_)
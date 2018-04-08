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

def getNameFromTime():
    now = datetime.datetime.now()
    return (str)(now.minute)+(str)(now.second) + (str)(now.microsecond)
import warnings
warnings.filterwarnings("ignore")


IMG_CHANNELS = 3
#path = '/Users/xavier.qiu/Documents/comp540project/comp540/comp540Project/data/stage1_train/'
#id_ = '3a22fe593d9606d4f137461dd6802fd3918f9fbf36f4a65292be69670365e2ca'
#path = path + id_
#img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#mask_imgs = list()
#
#
#temp_imgs = next(os.walk(path+'/masks/'))[2]
#assert len(temp_imgs) > 0
#for mask in temp_imgs:
#    mask_img = imread(path+'/masks/'+mask)
#    mask_imgs.append(mask_img)
    
def sliceImg(img, mask_imgs, output_shape = (256,256), img_id = '', dirPath = ''):
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
test_ids = next(os.walk(TEST_PATH))[1]

print('Getting and resizing train images and masks ... ')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    image__ = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    mask_imgs = list()
    
    
    temp_imgs = next(os.walk(path+'/masks/'))[2]
    assert len(temp_imgs) > 0
    for mask in temp_imgs:
        mask_img = imread(path+'/masks/'+mask)
        mask_imgs.append(mask_img)

    sliceImg(image__,mask_imgs,img_id = id_)
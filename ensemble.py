import model as modellib
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
from inference_config import inference_config,inference_config101
from bowl_dataset import BowlDataset
from utils import rle_encode, rle_decode, rle_to_string
import functions as f
from u_net import *

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = 'weights/mask_rcnn_1.h5'

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

model2 = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model2_path = 'weights/mask_rcnn_2.h5'
model2.load_weights(model2_path, by_name=True)

model_res101 = modellib.MaskRCNN(mode="inference",
                          config=inference_config101,
                          model_dir=MODEL_DIR)

model101_path = 'weights/mask_rcnn_101.h5'
model_res101.load_weights(model101_path, by_name=True)


u_net = get_unet()
u_net.load_weights('u-net/u-net.h5')

dataset_test = BowlDataset()
dataset_test.load_bowl('stage2_test_final')

dataset_test.prepare()
output = []
sample_submission = pd.read_csv('stage2_sample_submission_final.csv')
ImageId = []
EncodedPixels = []
print('start predicting')
for image_id in tqdm(sample_submission.ImageId):

    image_path = os.path.join('stage2_test_final', image_id, 'images', image_id + '.png')

    original_image = cv2.imread(image_path)
    results = model.detect([original_image], verbose=0, probablymask=True)
    results2 = model2.detect([original_image], verbose=0, probablymask=True)
    results101 = model_res101.detect([original_image], verbose=0, probablymask=True)


    r = results[0]
    masks = r['masks']
    probablymasks = r['probablymasks']




    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch

f.write2csv('submission_v2.csv', ImageId, EncodedPixels)

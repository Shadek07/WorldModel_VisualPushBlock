'''
change images from 84x84 dimension to 64x64, read from npz file and also save as npz file
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
import constants
from constants import IMAGE_H, IMAGE_W
from constants import SCREEN_Y, SCREEN_X
from constants import num_npzepisode_to_use, z_size
from scipy.misc import imresize
import matplotlib.pyplot as plt

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"
DIR_NAME = 'record'
if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)
def resizeimages(raw_data):
    x = []
    for data in raw_data:
        data = data.reshape(84, 84, 3)
        img = imresize(data, (64, 64, 3))
        img.reshape((1, 64, 64, 3))
        x.append(img)
    return x

def resize_raw_data_list(filelist):

  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    raw_data_obs = raw_data['obs']
    if(raw_data_obs.shape[2] == 84): #change from 84x84 to 64x64
        raw_data_obs = resizeimages(raw_data_obs)
    filename = DIR_NAME + "/" + filename
    np.savez_compressed(filename, obs=raw_data_obs, action=raw_data['action'])
    if ((i+1) % 100 == 0):
      print("loading file", (i+1))


# Hyperparameters for ConvVAE
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:num_npzepisode_to_use]

resize_raw_data_list(filelist)


'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
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
from constants import num_npzepisode_to_use, z_size, kl_tolerance, learning_rate
from scipy.misc import imresize
import matplotlib.pyplot as plt

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"

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

def load_raw_data_list(filelist):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    raw_data_obs = raw_data['obs']
    '''if(raw_data_obs.shape[2] == 84): #change from 84x84 to 64x64
        raw_data_obs = resizeimages(raw_data_obs)
        if i == 0:
            print(raw_data_obs[0])
        print('resized')'''
    raw_data_obs = raw_data_obs*255.0
    raw_data_obs = raw_data_obs.astype(np.uint8)
    data_list.append(raw_data_obs)
    action_list.append(raw_data['action'])
    if ((i+1) % 100 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode_batch(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float) #np.copy(batch_img).astype(np.float)/255.0
  #print('obs shape',simple_obs.shape)
  simple_obs = simple_obs.reshape(simple_obs.shape[0], SCREEN_X, SCREEN_Y, 3)   #simple_obs.reshape(batch_size, SCREEN_X, SCREEN_Y, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z


def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, SCREEN_X, SCREEN_Y, 3)
  return batch_img


# Hyperparameters for ConvVAE
batch_size=1000 # treat every episode as a batch of 1000!

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:num_npzepisode_to_use]

dataset, action_dataset = load_raw_data_list(filelist)
print(len(dataset))
print(len(action_dataset))

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
action_dataset_new = []
cnt=0
for i in range(len(dataset)):
  data_batch = dataset[i] #Lx1x64x64x3 (L <= 1001)
  action_dataset_i = action_dataset[i]
  episode_len = len(data_batch)
  last_frame = data_batch[episode_len-1]
  #make all episode of same length ( i.e 1001)
  while episode_len < 1001: #fill up with last frame
      data_batch = np.concatenate((data_batch, np.array([last_frame])))
      dummy_action = np.array([[[0]]])
      #print('action shape before concatenate', action_dataset_i.shape, dummy_action)
      action_dataset_i = np.concatenate((action_dataset_i, dummy_action))
      episode_len += 1
  data_batch = data_batch.astype(np.float) / 255.0
  mu, logvar, z = encode_batch(data_batch)
  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  action_dataset_new.append(action_dataset_i)
  if ((i+1) % 100 == 0):
    print(i+1)
    print(data_batch.shape, action_dataset_i.shape)
action_dataset = np.array(action_dataset_new)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)
print('mu shape from series.py', mu_dataset.shape)
np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)

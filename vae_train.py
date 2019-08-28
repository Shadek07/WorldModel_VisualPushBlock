'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
import tensorflow as tf
import random
import numpy as np
import constants
from constants import IMAGE_H, IMAGE_W
from constants import SCREEN_Y, SCREEN_X, num_npzepisode_to_use, kl_tolerance, VAE_TRAIN_EPOCH, learning_rate
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph

# Hyperparameters for ConvVAE
#z_size=64
batch_size=100
# Parameters for training
DATA_DIR = "record"
model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return total_length

def create_dataset(filelist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  data = np.zeros((M*N, SCREEN_X, SCREEN_Y, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    #print(l)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    raw_data_reshaped = raw_data.reshape((l, SCREEN_X, SCREEN_Y, 3))
    raw_data_reshaped = raw_data_reshaped*255.0
    data[idx:idx+l] = raw_data_reshaped
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  print('data check', data[0])
  return data[0:idx]

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
#filelist = filelist[0:10000]
filelist = filelist[0:num_npzepisode_to_use] #Change is done here
#print("check total number of images:", count_length_of_filelist(filelist))
dataset = create_dataset(filelist, N=num_npzepisode_to_use)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
print('num batches', num_batches)
min_train_loss = 99999999999.0
for epoch in range(VAE_TRAIN_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0 #batch.astype(np.float)

    feed = {vae.x: obs,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
    if(train_loss < min_train_loss):
      min_train_loss = train_loss
      vae.save_json("tf_vae/vae.json")
    #print('obs shape', obs.shape)
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss, min_train_loss)

    #if ((train_step+1) % 5000 == 0):
    #  vae.save_json("tf_vae/vae.json")

# finished, final model:
vae.save_json("tf_vae/vae.json")
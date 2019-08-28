import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7,8,9"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
IMAGE_W = 64
IMAGE_H = 64

SCREEN_X = 64
SCREEN_Y = 64
num_npzepisode_to_use = 7000
z_size = 64
kl_tolerance=0.5
learning_rate=0.0001
VAE_TRAIN_EPOCH = 6

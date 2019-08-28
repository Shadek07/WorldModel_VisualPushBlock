'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams, MDNRNN
import constants

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = "series"
model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=1000, # train on sequences of 1001 (so 1000 + teacher forcing shift)
                     input_seq_width=65,    # # width of our data (64 + 1 actions) - one selected action
                     output_seq_width=64,    # width of our data is 64
                     rnn_size=256,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"), allow_pickle=True)

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
#data_mu.reshape(data_mu.shape[0], data_mu[0].shape[0], data_mu[0].shape[1])
#data_logvar.reshape(data_logvar.shape[0], data_logvar[0].shape[0], data_logvar[0].shape[1])
data_action =  raw_data["action"]
max_seq_len = hps_model.max_seq_len

N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

# save 1001 initial mu and logvars:
print('data mu shape',data_mu.shape)
'''x = np.array(data_mu[:1000])
y = np.array(data_logvar[:1000])
print(x[0].shape)
print('x[0]', x[0])
initial_mu =  np.copy(x[0]*10000).astype(np.int).tolist()
initial_logvar = np.copy(y[0]*10000).astype(np.int).tolist()
print('initial mu', initial_mu[0:2])
#print(initial_logvar)'''

initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist() # #np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist() #np.copy(y[0]*10000).astype(np.int).tolist() #np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

reset_graph()
rnn = MDNRNN(hps_model)

# train loop:
hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):

  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  raw_z, raw_a = random_batch()
  #print(raw_z.shape)
  #print(raw_a.shape[0])
  raw_a = np.reshape(raw_a, (batch_size, 1001, -1))
  #print(raw_a.shape)
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2) #100x1000x65
  outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions) #100x1000x64
  #print(inputs.shape)
  #print(outputs.shape)
  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# save the model (don't bother with tf checkpoints json all the way ...)
rnn.save_json(os.path.join(model_save_path, "rnn.json"))

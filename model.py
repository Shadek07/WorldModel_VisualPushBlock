import numpy as np
import random

import json
import sys

from env import make_env
import time
import constants
from constants import IMAGE_H, IMAGE_W, z_size
from constants import SCREEN_Y, SCREEN_X
from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

ACTION_SIZE = 1

def make_model(load_model=True):
  # can be extended in the future.
  model = Model(load_model=load_model)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)


def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self, load_model=True):
    self.env_name = './VisualPushBlock_withBlock_z_info.x86_64' #'./VisualPushBlock.x86_64'
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      self.vae.load_json('vae/vae.json')
      self.rnn.load_json('rnn/rnn.json')

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = z_size

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer ###CHANGE is made here
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, ACTION_SIZE)
      self.bias_output = np.random.randn(ACTION_SIZE)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*ACTION_SIZE+ACTION_SIZE)
    else:
      self.weight = np.random.randn(self.input_size, ACTION_SIZE)
      self.bias = np.random.randn(ACTION_SIZE)
      self.param_count = (self.input_size)*ACTION_SIZE+ACTION_SIZE

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False, full_episode=False, worker_id=0):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode, worker_id=worker_id)

  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    #result = np.copy(obs).astype(np.float)/255.0

    result = np.copy(obs).astype(np.float)
    result = result.reshape(1, IMAGE_W, IMAGE_H, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def get_action(self, z):
    h = rnn_output(self.state, z, EXP_MODE)
    #print('h', h.shape, h)
    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      '''print(h.shape)
      print(self.weight.shape)
      print(self.bias.shape)'''
      action = np.tanh(np.dot(h, self.weight) + self.bias)

    '''for i in range(ACTION_SIZE):
      action[i] = (action[i]+1.0) / 2.0 #all actions value are in range 0 to 1'''
    #action[2] = clip(action[2])
    self.state = rnn_next_state(self.rnn, z, action, self.state) #update weights of MDN-RNN
    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:ACTION_SIZE]
      self.weight_output = params_2[ACTION_SIZE:].reshape(self.hidden_size, ACTION_SIZE)
    else:
      self.bias = np.array(model_params[:ACTION_SIZE])
      self.weight = np.array(model_params[ACTION_SIZE:]).reshape(self.input_size, ACTION_SIZE)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.vae.get_random_model_params(stdev=stdev)
    self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

def clip_action(x, lo=-0.5, hi=0.5):
  return np.minimum(np.maximum(x, lo), hi)

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
  reward_list = []
  t_list = []

  max_episode_length = 1001
  recording_mode = False
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    #model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    #obs = model.env.reset()
    obs = model.env.reset(train_mode=False)[model.env.brain_names[0]]
    obs = np.asarray(obs.visual_observations[0])

    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    filename = "record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]
    recording_z = []
    recording_block_z = [] #z position of block
    recording_dist = [] #distance from agent to block
    for t in range(max_episode_length):

      '''if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')'''

      z, mu, logvar = model.encode_obs(obs)
      #recording_z.append(z)

      action = model.get_action(z)

      action[0] = clip_action(action[0])
      action_indexes = [3, 6, 1, 2, 5, 4]  # (0.33 to 0.5), (-0.163, 0.33) so on
      x = 0.5
      action_index = 4  # for value range [-1,0,65]
      for i in range(6):
        if x - 0.167 < action[0]:
          action_index = action_indexes[i]
          break
        x = x - 0.167  # 1/6 = 0.167, action value range = (1-(-1)) = 2
      action = np.array([[action_index]])
      #print(action)
      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)

      #obs, reward, done, info = model.env.step(action)
      env_info = model.env.step(action)[model.env.brain_names[0]]  # reward, done, info
      '''
      for each step without reaching goal, reward is: -0.0009
      '''
      obs, reward, done = env_info.visual_observations[0], env_info.rewards[0], env_info.local_done[0]
      recording_block_z.append(env_info.vector_observations[0, 0]) #z coordinate of block
      recording_dist.append(env_info.vector_observations[0, 1])
      obs = np.asarray(obs)
      extra_reward = 0.0
      if action_index >= 2 and action_index <= 4: #penalize backward, clockwise and anti-clockwise move
        extra_reward -= 0.001

      '''curosity_reward = 0
      l = len(recording_z)
      if(l > 1):
        for j in range(0, l - 1):
          curosity_reward += np.mean(np.abs(recording_z[l - 1] - recording_z[j]))
        curosity_reward /= (l - 1)  # average
        extra_reward += curosity_reward*0.001'''
      a = 0
      if len(recording_block_z) > 1:
        d = recording_block_z[-1] - recording_block_z[-2]
        if d > 0:
          a = d*0.1
        else:
          a = d*0.01
        extra_reward += a #if block is pushed towards goal area then +ve reward
      if len(recording_dist) > 1:
        d = recording_dist[-2] - recording_dist[-1]
        if d > 0:
          a = d*0.1 #if curr distance is smaller than prev then +ve reward
        else:
          a = d*0.01
        extra_reward += a
        #print(a)
      #print('extra reward', extra_reward, 'curosity: ', curosity_reward*0.001, 'block:', a)
      reward += extra_reward

      ###NEED TO PENALIZE for ROTATING and moving BACKWARD

       # penalize for turning too frequently
      '''if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward'''
      recording_reward.append(reward)
      #if (render_mode):
      #  print("action", action, "step reward", reward)

      total_reward += reward

      if done:
        total_reward += 1.0
        break

    #for recording:
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    action[0] = clip_action(action[0])
    action_indexes = [3, 6, 1, 2, 5, 4]  # (0.33 to 0.5), (-0.163, 0.33) so on
    x = 0.5
    action_index = 4  # for value range [-1,0,65]
    for i in range(6):
      if x - 0.167 < action[0]:
        action_index = action_indexes[i]
        break
      x = x - 0.167  # 1/6 = 0.167, action value range = (1-(-1)) = 2
    action = np.array([[action_index]])

    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)

    if not render_mode:
      if recording_mode:
        np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  assert len(sys.argv) > 1, 'python model.py render/norender path_to_mode.json [seed]'


  render_mode_string = str(sys.argv[1])
  if (render_mode_string == "render"):
    render_mode = True
  else:
    render_mode = False

  use_model = False
  if len(sys.argv) > 2:
    use_model = True
    filename = sys.argv[2]
    print("filename", filename)

  the_seed = np.random.randint(10000)
  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  if (use_model):
    model = make_model()
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.load_model(filename)
  else:
    model = make_model(load_model=False)
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = 100
  if render_mode:
    N_episode = 1
  reward_list = []
  for i in range(N_episode):
    reward, steps_taken = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=1)
    if render_mode:
      print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    else:
      print(reward[0])
    reward_list.append(reward[0])
  if not render_mode:
    print("seed", the_seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()

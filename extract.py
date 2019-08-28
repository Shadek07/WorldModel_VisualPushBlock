'''
saves ~ 200 episodes generated from a random policy
'''

'''
in this pushblock version: unity image size 64x64 is being used and vae z_size is set to 64
 '''
import numpy as np
import random
import os
import gym
import sys
from model import make_model
import matplotlib.pyplot as plt
MAX_FRAMES = 5000 #1000 # max length of carracing
MAX_TRIALS = 400 #200 just use this to extract one trial.

render_mode = False # for debugging.
IMAGE_DIR = 'images'
DIR_NAME = 'record64'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model(load_model=False)

def clip_action(x, lo=-0.5, hi=0.5):
  return np.minimum(np.maximum(x, lo), hi)

total_frames = 0
model.make_env(render_mode=render_mode, full_episode=True, worker_id=int(sys.argv[1]))

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
image_count = 0
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []

    np.random.seed(random_generated_int)
    #model.env.seed(random_generated_int)

    # random policy
    model.init_random_model_params(stdev=np.random.rand()*0.01)

    model.reset()
    obs = model.env.reset(train_mode=False)[model.env.brain_names[0]] # pixels
    obs = np.asarray(obs.visual_observations[0])
    #print('obs', np.asarray(obs).shape)
    #print(obs)
    prev_z = np.zeros((1, 64))
    for index, frame in enumerate(range(MAX_FRAMES)):
      '''if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")'''

      recording_obs.append(obs)

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)
      action[0] = clip_action(action[0])
      action_indexes = [3, 6, 1, 2, 5, 4] #(0.33 to 0.5), (-0.163, 0.33) so on
      x = 0.5
      action_index = 4 #for value range [-1,0,65]
      for i in range(6):
        if x-0.167 < action[0]:
          action_index = action_indexes[i]
          break
        x = x-0.167 #1/6 = 0.167, action value range = (1-(-1)) = 2
      '''  
      action index[1]: moving forward
      action index[2]: moving backward
      action index[3]: rotate clockwise
      action index[4]: rotate anti-clockwise
      action index[5]: moving left
      action index[6]: moving right
      '''
      #print(action.shape,action[0],action_index)
      curr_z = z
      action = np.array([[action_index]])
      if(index > 0):
        a = np.mean(np.abs(curr_z-prev_z))
        print(a)
      prev_z = z
      recording_action.append(action)
      env_info = model.env.step(action)[model.env.brain_names[0]] # reward, done, info
      obs, reward, done = env_info.visual_observations[0], env_info.rewards[0], env_info.local_done[0]
      obs = np.asarray(obs)
      #print('reward', reward) #-0.0009
      #print('obs_next', obs)
      #plt.imshow(obs[0, :, :, :])
      '''if(image_count < 1000):
        plt.savefig('./images/obs'+str(image_count)+"_"+str(action_index))'''
      image_count += 1
      if done:
        break
    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()

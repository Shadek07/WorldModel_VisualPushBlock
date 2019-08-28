from mlagents.envs import UnityEnvironment
from pushblock_env import PushBlockEnv
import matplotlib.pyplot as plt
import numpy as np
import sys
import socket
from gym import error, spaces
from matplotlib import cm
#socket.setdefaulttimeout(300)
class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass
print("Hello from ml-agents")
'''env = PushBlockEnv('./VisualPushBlock', retro=False, realtime_mode=False)
print(env.action_space)
print(env.observation_space)'''
env_name = './VisualPushBlock_withBlock_z_info.x86_64' # Name of the Unity environment binary to launch
train_mode = True  # Whether to run the environment in training or inference mode

env = UnityEnvironment(file_name=env_name, worker_id=1)

# Set the default brain to work with
'''brain_name = env.external_brain_names[0]
#default_brain = env.brain_names[0]
brain = env.brains[brain_name]
print(brain)'''
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
#print(brain)
#print(brain.camera_resolutions)

env_info = env.reset(train_mode=train_mode)[default_brain]
print(env_info.vector_observations[0])
# Examine the observation space for the default brain

'''print(type(env_info.visual_observations[0]))

for observation in env_info.visual_observations:
    observation = np.asarray(observation)
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0,:,:,:])
    else:
        plt.imshow(observation[0,:,:,0])
    #plt.show()'''
for episode in range(100):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        #print(action_size[0])
        if brain.vector_action_space_type == 'continuous':
            env_info = env.step(np.random.randn(len(env_info.agents), action_size[0]))[default_brain]
            print('continuous')
            print(np.random.randn(len(env_info.agents), action_size[0]))
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            #print(action[0])
            #action[0] = np.array([1])
            #print(np.asarray(action).shape)
            env_info = env.step(action)[default_brain]
            print(env_info.vector_observations[0, :])
            #model.env.step(action)[model.env.brain_names[0]]
        #print(env_info.vector_observations[0].shape)
        #print(env_info.visual_observations[0])
        #x = np.asarray(env_info.visual_observations[0])
        #print(x.shape)
        #print(env_info.vector_observations[0])
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

'''if brain.number_visual_observations == 0:
    raise UnityGymException("Environment provides no visual observations.")
_action_space = -100
if len(brain.vector_action_space_size) == 1:
    _action_space = spaces.Discrete(brain.vector_action_space_size[0])
print(_action_space)'''
# Reset the environment
'''env_info = env.reset(train_mode=train_mode)[brain_name]
# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

# Examine the observation space for the default brain
for observation in env_info.visual_observations:
    print("Agent observations look like:")
    #print(observation)
    ob = np.asarray(observation)
    print(ob.shape)
    if ob.shape[3] == 3:
        plt.imshow(ob[0,:,:,:])
    else:
        plt.imshow(ob[0,:,:,0])'''


'''for episode in range(10):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        if brain.vector_action_space_type == 'continuous':
            env_info = env.step(np.random.randn(len(env_info.agents),
                                                action_size[0]))[default_brain]
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            env_info = env.step(action)[default_brain]
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))'''

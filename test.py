import gym
import panda_gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer
import numpy as np
from typing import Any, Dict, Union
from panda_gym.utils import distance
import math
# env = gym.make("PandaReach-v2", render=True)
# model = DDPG.load('ddpg_panda_reach_v2', env=env)
# # model = TD3.load('td3_panda_reach_v2', env=env)
# # model = SAC.load('sac_panda_reach_v2', env=env)

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print('Done')
#         obs = env.reset()
models_dir = './simulation/1204/model'
simulation_num = 2
print(f"{models_dir}/cable_v0_{simulation_num}")

distance_threshold = np.array([0.005,0.005,0.005])
achieved_goal = np.array([0.001,0.005,0.001])
desired_goal = np.array([0.005,0.005,0.005])
print(achieved_goal[0])
d_is = np.array([0,0,0],dtype=np.float64)
is_flag = np.array([0,0,0],dtype=np.float64)
for i in range(3):
    d_is[i] = math.sqrt((achieved_goal[i]-desired_goal[i])**2)

is_flag = np.array([d_is[0]<=distance_threshold[0],
                    d_is[1]<=distance_threshold[1],
                    d_is[2]<=distance_threshold[2]])
is_flag.all()
info = {"is_success": 1}
print(np.array(is_flag.all()))
print(info['is_success'])

err_theta = np.array([math.pi/6,math.pi/6,math.pi/6])
print('array:',err_theta)
print('sin(array):',math.sin(err_theta[0]))
observationSin = np.ones((12,), dtype=np.float32)
print('obs_ones',observationSin)

replay_buffer_kwargs=dict(
        n_sampled_goal=5,
        goal_selection_strategy="future",
        # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
        # we have to manually specify the max number of steps per episode
        # max_episode_length=100,
        # online_sampling=True,
        ),

print('replay_buffer_kwargs.get',replay_buffer_kwargs.get("online_sampling", True))
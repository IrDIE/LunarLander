import time

from agent import *
import gym
import numpy as np, torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


env = gym.make('LunarLanderContinuous-v2', render_mode="human")
obs = env.reset() # 8

# print(obs)
# for _ in range(100):
#     env.render()
#     new_state, reward, term, trunc, info = env.step([1,1])
#     time.sleep(0.2)
#
#     done = term or trunc
#     print(new_state, reward, term, trunc, info)

# training_ddpg_per(env,  seq_size = 5)
inference(env)

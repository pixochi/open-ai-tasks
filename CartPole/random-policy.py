# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:03:56 2019

@author: Bruger

Inspired by:
http://kvfrans.com/simple-algoritms-for-solving-cartpole/
"""

import gym
import numpy as np

ENVIRONMENT_NAME = "CartPole-v0"

env = gym.make(ENVIRONMENT_NAME).env


def run_episode(env, weights):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):
        action = 0 if np.matmul(weights, observation) < 0 else 1
        observation, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:03:56 2019

@author: Bruger

Inspired by:
http://kvfrans.com/simple-algoritms-for-solving-cartpole/
"""

import gym

ENVIRONMENT_NAME = "CartPole-v0"
env = gym.make(ENVIRONMENT_NAME).env

# =============================================================================
# RANDOM POLICY
# =============================================================================

score_total = 0
episodes = 100

for _ in range(episodes):
    score = 0
    done = False
    state = env.reset()

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        score += reward
    
    score_total += score

env.close()
print(f"Results after {episodes} episodes:")
print(f"Average score per episode: {score_total / episodes}")


# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:25:05 2019

@author: Bruger

Inspired by: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""

import gym
from time import sleep
import numpy as np
import random

ENVIRONMENT_NAME = "Taxi-v2"

env = gym.make(ENVIRONMENT_NAME).env

# =============================================================================
# TRAINING THE AGENT
# =============================================================================

# Initializes Q-table
possible_states_count = env.observation_space.n
available_actions_count = env.action_space.n
Q_table = np.zeros([possible_states_count, available_actions_count])

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.6 # Discount factor
epsilon = 0.1 # Probability of taking a random action

for i in range(1, 100001):
    state = env.reset()

    epochs = 0
    penalties = 0 # For wrong pickup/dropoff actions
    reward = 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(Q_table[state]) # Exploit the Q-table

        next_state, reward, done, info = env.step(action) 
        
        current_Q_value = Q_table[state, action]
        next_state_max_Q_value = np.max(Q_table[next_state])
        
        new_Q_value = (1 - alpha) * current_Q_value + alpha * (reward + gamma * next_state_max_Q_value)
        Q_table[state, action] = new_Q_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")

# =============================================================================
# EVALUATION OF THE AGENT AFTER Q-LEARNING
# =============================================================================

epochs_total = 0
penalties_total = 0
episodes = 100

for _ in range(episodes):
    epochs = 0
    penalties = 0
    done = False
    state = env.reset()    
    
    while not done:
        action = np.argmax(Q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    penalties_total += penalties
    epochs_total += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {epochs_total / episodes}")
print(f"Average penalties per episode: {penalties_total / episodes}")

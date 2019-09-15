# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:26:22 2019

@author: Jakub Kozak

Inspired by: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""

import gym
from IPython.display import clear_output
from time import sleep

# =============================================================================
# The Reward Table:
# - states(0-499) x actions(0-5) matrix
# - {action: [(probability, nextstate, reward, done)]}
#   - probability: starts with 1.0
#   - action: 0-5 => south, north, east, west, pickup, dropoff
#   - reward:
#       - -1 for movement and hitting a wall
#       - +20 for the correct dropoff action
#       - -10 for the incorrect pickup/dropoff actions
#   - done: the correct dropoff
# =============================================================================


ENVIRONMENT_NAME = "Taxi-v2"

# We are using the .env on the end of make to avoid training stopping at 200 iterations.
env = gym.make(ENVIRONMENT_NAME).env

env.reset()

#print(f'Action Space {env.action_space}')
#print(f'State Space {env.observation_space}')

# filled square - the taxi
#    yellow/red - without a passenger
#    green - with a passenger
# R, G, Y, B - pickup and destination locations
#    blue letter - the current passenger pick-up location
#    purple letter - the current destination
# pipe ("|") - a wall

# (taxi row, taxi column, passenger location, destination location)
# Encoded state is a number between 0 and 499
#state = env.encode(3, 1, 2, 0)
#print("State:", state)

# Sets the state manually
#env.s = state

#env.render()

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

# =============================================================================
# EVALUATION WITHOUT Q-LEARNING
# =============================================================================
        
epochs_total = 0
penalties_total = 0
episodes = 100

for _ in range(episodes):
    epochs = 0
    penalties = 0
    done = False
    frames = []
    state = env.reset()   
    
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1
            
        frames.append({
                'action': action,
                'state': state,
                'reward': reward,
                'frame': env.render(mode='ansi')
        })
        epochs += 1
    
    epochs_total += epochs
    penalties_total += penalties
            
    #print_frames(frames)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {epochs_total / episodes}")
print(f"Average penalties per episode: {penalties_total / episodes}")








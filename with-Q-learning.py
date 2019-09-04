# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:25:05 2019

@author: Bruger

Inspired by: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""

import numpy as np


q_table = np.zeros([env.observation_space.n, env.action_space.n])
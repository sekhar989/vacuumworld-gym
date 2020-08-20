#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:46:18 2020

@author: archie
"""

import torch

import copy

from vwgym.init_utils import *
from vwgym.fun_lite import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm

torch.manual_seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'

args = {'lr': 25e-4,
        'steps': 400,
        'max_steps': 1e8,
        'env_reboot':5e5,
        'entropy_coef': 0.01,
        'gamma': 0.99,
        'alpha': 0.5,
        'eps': 0.01,
        'grid_size': 3,
        'd': 256,
        'k': 16,
        'len_hist': 10,
        'grad_clip':5.0,
        'writer': True,
        'num_worker': 1}

# env, input_shape = make_env(args['grid_size'], args['num_worker'])
random.seed(518123)
env_n = VacuumWorld(args['grid_size'])
env_v = Vectorise(copy.deepcopy(env_n))
env = StepWrapper(env_v)

num_actions_wi = range(env.action_space.n)
num_actions = range(env.action_space.n - 1)

def observation(grid):

    dirt_table = {'green':128, 'orange':255}
    orientation_table = {'north':64, 'east':128, 'south':192, 'west':255}

    obs = 0

    c = next(iter(grid._get_agents().keys()))
    obs = 255 + c[1] + c[0]

    for c,a in grid._get_agents().items(): #all agents
        obs += c[1] + c[0] + orientation_table[a.orientation]

    for c,d in grid._get_dirts().items(): #all dirts
        obs += c[1] + c[0] + dirt_table[d.colour]

    return obs

from collections import defaultdict
Q = defaultdict(float)
gamma = args['gamma']  # Discounting factor
alpha = args['alpha']  # soft update param

def update_Q(s, r, a, s_next, done, Q, actions):
    max_q_next = max([Q[s_next, a] for a in actions])
    # Do not include the next state's value if currently at the terminal state.
#     print('\n', Q, '\n')
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])

def react(state, args, actions, q):

#     if np.random.random() < args['eps']:
#         return np.
    q_values = {a: q[state, a] for a in actions}
    max_q = max(q_values.values())

    actions_with_max_q = [a for a, q in q_values.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

ob = env.reset()
env.rw_dirts = env.dirts
ob = observation(env.state())
rewards = []
reward = 0.0
history = []

for step in range(100000):
    a = react(ob, args, num_actions_wi, Q)
    ob_next, r, done, ep_info = env.step(a)
    ob_next = observation(env.state())
    update_Q(ob, r, a, ob_next, done, Q, num_actions_wi)
    reward += r
    if done:
        history.append(ep_info)
        rewards.append(reward)
        reward = 0.0
        ob = env.reset()
        env.rw_dirts = env.dirts
        ob = observation(env.state())
    else:
        ob = ob_next

    if len(history) > 50 and (history[-2] == history[-1]):
        break

import pandas as pd

h = pd.DataFrame(history)

import matplotlib.pyplot as plt

h['ep_rewards'].plot()
plt.title('Rewards vs Episodes')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()
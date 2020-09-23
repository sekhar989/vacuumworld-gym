#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:15:18 2020

@author: Krishnendu S. Kar
"""
import numpy as np

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from dqn import *

from vwgym.init_utils import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, shutil
import argparse

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=518123,
                    help='random seed')

parser.add_argument('--sparse', type=int, default=1,
                    help='less dirts or grid_size*2')

parser.add_argument('--grid_size', type=int, default=3,
                    help='grid_size')

term_args = parser.parse_args()


if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'

args = {'lr': 0.0001,
        'steps': 100000,
        'max_steps': 1e8,
        'entropy_coef': 0.01,
        'gamma': 0.999,
        'alpha': 0.5,
        'eps': 0.5,
        'grad_clip':5.0,
        'writer': True,
        'BATCH_SIZE':256,
        'EPS_START':1.0,
        'EPS_END':0.01,
        'EPS_DECAY': 50000.0,
        'TARGET_UPDATE':10}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env, input_shape = make_env(grid_size=term_args.grid_size,
                                sparse=term_args.sparse,
                                random_seed=term_args.seed)
env = env[0]
n_actions = env.action_space.n


policy_net = dqn(input_shape, n_actions, device)
target_net = dqn(input_shape, n_actions, device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

op = optim.Adam(policy_net.parameters(), lr=args['lr'])
history = ReplayMemory(2000)

steps_done = 0
ep_lens = 1000

log_dir = 'dqn_train_logs/'

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir=log_dir)

for i_episode in range(args['steps']):
    R_t = []
    d = []

    state = env.reset()

    env.rw_dirts = env.dirts
    state = torch.from_numpy(state).float()
    state = state.view(1, -1).to(device)
    print('Episode No.:\t', i_episode)

    for t in count():

        action = select_action(state, args, device, policy_net, env.action_space.n, steps_done)
        next_state, reward, done, ep_info = env.step(action.item())

        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        next_state = next_state.view(1, -1).to(device)
        reward = torch.tensor([reward], device=device)

        if done:
            print('\n', '-'*50)
            print('Steps Done:\t', steps_done)
            next_state = None
            break
        # if steps_done <= 2000:
            # print('Memory added..')
        history.push(Transition, state, action, next_state, reward)
        state = next_state
        optimize_model(args, device, history, policy_net, target_net, op, writer, steps_done, Transition)
        steps_done += 1

    plot_durations(ep_info, writer, i_episode)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % args['TARGET_UPDATE'] == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save({'model': target_net.state_dict(),
                    'args': args,
                    'optim': op.state_dict()},
                   'saved_model/dqn_ckpt.pt')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:37:10 2020

@author: archie
"""
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

from vwgym.init_utils import *
from vwgym.fun_lite import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'

class dqn(nn.Module):

    def __init__(self, input_shape, num_actions, device):
        super(dqn, self).__init__()

        channels, height, width = input_shape
        self.d = channels * height * width
        self.fc = nn.Sequential(nn.Linear(self.d, 32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU(),
                                nn.Linear(16, num_actions),
                                nn.Softmax(dim=1))
        self.to(device)

    def forward(self, x):
        return self.fc(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
        'num_worker': 1,
        'BATCH_SIZE':128,
        'EPS_START':0.9,
        'EPS_END':0.05,
        'EPS_DECAY':200,
        'TARGET_UPDATE':10}

def select_action(state, args, device):
    global steps_done

    sample = random.random()
    eps_threshold = args['EPS_END'] + (args['EPS_START'] - args['EPS_END']) * \
        math.exp(-1. * steps_done / args['EPS_DECAY'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


env, input_shape = make_env(args['grid_size'], args['num_worker'])
env = env[0]
n_actions = env.action_space.n
env, n_actions

policy_net = dqn(input_shape, n_actions, device)
target_net = dqn(input_shape, n_actions, device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0
ep_lens = []

def optimize_model(args, device):

    if len(memory) < args['BATCH_SIZE']:
        return
    transitions = memory.sample(args['BATCH_SIZE'])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(state_action_values.shape)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args['BATCH_SIZE'], device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args['gamma']) + reward_batch
    # print(expected_state_action_values.shape)

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.view(-1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations():
#     plt.figure(2)
#     plt.clf()
    durations_t = torch.tensor(ep_lens, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        print('Mean Durations..', means.nupmy())
#         plt.plot(means.numpy());

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         time.sleep(5.0)
#         display.clear_output(wait=True)
#         display.display(plt.gcf())

for i_episode in range(args['steps']):
    # Initialize the environment and state
    state = env.reset()
    # if i_episode % 10 == 0:
    #     display.clear_output(wait=True)
    env.rw_dirts = env.dirts
    state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
#     last_screen = get_screen()
#     current_screen = get_screen()
#     state = current_screen - last_screen
    for t in tqdm(count()):
        # Select and perform an action
        action = select_action(state, args, device)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state.reshape(1, -1)).float().to(device)
        reward = torch.tensor([reward], device=device)

        # Observe new state
#         last_screen = current_screen
#         current_screen = get_screen()
        if done:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(args, device)
        if done:
            ep_lens.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % args['TARGET_UPDATE'] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show();
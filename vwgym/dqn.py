#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:37:10 2020

@author: archie
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import random
import math
from vwgym.init_utils import *
from vwgym.fun_lite import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
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

        self.device = device

        channels, height, width = input_shape
        self.d = (channels * height * width) + 4

        # self.conv1 = nn.Conv1d(1, 256, kernel_size=2, stride=1)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=2)
        # self.bn2 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(self.d, 1024)
        self.fc1.weight.data.normal_(0, 0.1)
                                # nn.LeakyReLU(),
                                # nn.Linear(1024, 512),
                                # nn.LeakyReLU(),
                                # nn.Linear(512, 256),
                                # nn.LeakyReLU(),
                                # nn.Linear(256, 128),
                                # nn.LeakyReLU(),
                                # nn.Linear(128, 64),
                                # nn.LeakyReLU(),
                                # nn.Linear(64, 32),
                                # nn.LeakyReLU(),
                                # nn.Linear(32, 16),
                                # nn.LeakyReLU(),
        self.fc2 = nn.Linear(1024, num_actions)
        self.fc2.weight.data.normal_(0, 0.1)
                                # nn.Linear(1024, 8),
                                # nn.LeakyReLU(),
                                # nn.Linear(8, num_actions))

        self.to(device)

    def forward(self, x):
        # x = F.leaky_relu(self.bn1(self.conv1(x)))
        # x = F.leaky_relu(self.bn2(self.conv2(x)))

        x = self.fc1(x)
        x = F.leaky_relu(x)
        actions_value = self.fc2(x)
        return actions_value

        # return self.fc(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition_tuple, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition_tuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, args, device, net, n_actions, steps_done):

    sample = random.random()
    ep_threshold = args['EPS_END'] + (args['EPS_START'] - args['EPS_END']) * math.exp(-1. * steps_done / args['EPS_DECAY'])
    print(ep_threshold)
    if sample > ep_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = net(state)
            # print(action)
            action = action.max(1)[1].view(1, 1)
            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(args, device, memory, q_train, q_target, optimizer, w, steps, t_batch):

    if len(memory) < args['BATCH_SIZE']:
        return
    transitions = memory.sample(args['BATCH_SIZE'])

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = t_batch(*zip(*transitions))

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
    state_action_values = q_train(state_batch).gather(1, action_batch)
    # print(state_action_values.shape)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args['BATCH_SIZE'], device=device)
    next_state_values[non_final_mask] = q_target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args['gamma']) + reward_batch
    # print(expected_state_action_values.shape)

    # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    w.add_scalars('loss/mse_loss', {'loss': loss}, steps)
    # loss_cal += 1

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in q_train.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations(info, w, ep_num):

    w.add_scalars('episode', {'episode_length': info['ep_len']}, ep_num)
    w.add_scalars('episode', {'episode_rewards': info['ep_rewards']}, ep_num)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 05:19:18 2020

@author: Krishnendu S. Kar
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Actor, self).__init__()
        channels, height, width = input_shape

        self.d = (channels * height * width) + 4

        self.fc1 = nn.Linear(self.d, 1024)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.1)

        self.actor = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=1)


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        channels, height, width = input_shape

        self.d = (channels * height * width) + 4

        self.fc1 = nn.Linear(self.d, 1024)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.1)

        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.critic(x)


def compute_returns(next_value, rewards, masks, gamma=0.99):
    r_t = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        r_t = rewards[step] + (gamma * r_t * masks[step])
        returns.insert(0, r_t)
    return returns

def select_action(action_probs, steps):

    dist = Categorical(action_probs)
    ep_threshold = 10.01 + (1.0 - 0.01) * math.exp(-1. * steps/50000.0)

    if np.random.uniform(low=0.3, high=1.0, size=None) > ep_threshold:
        action = action_probs.max(1)[1]
    else:
        action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()

    return action.cpu().numpy(), logp, entropy
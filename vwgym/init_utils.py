#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 08-07-2020

    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################

    -- Storage Class (mini DB)

"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import numpy as np
from vwgym import VacuumWorld, Vectorise, StepWrapper
from torch.distributions import Categorical
# torch.manual_seed(518123)

class db:
    """A mini_db for storing values efficiently during model training"""
    def __init__(self, keys, space):
        super(mini_db, self).__init__()
        if keys is None:
            keys = []

        self.keys = keys
        self.space = space
        self.reset()

    def update(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [0] * self.space)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def normalize(self, keys):
        for key in keys:
            k = torch.stack(getattr(self, key))
            k = (k - k.mean()) / (k.std() + 1e-10)
            setattr(self, key, [i for i in k])

    def stack(self, keys):
        # for k in keys:
        #     print(k)
        #     print(getattr(self, k)[:self.space])
        data = [getattr(self, k)[:self.space] for k in keys]
        # print(data)
        return map(lambda x: torch.stack(x, dim=0), data)


def make_env(grid_size, num_env, vectorize=True, random_seed=True):

    """
    Initializing the VacuumWorld Environment
    """
    if random_seed:
        random.seed(518123)

    env = Vectorise(VacuumWorld(grid_size))
    env = StepWrapper(env)

    envs = []
    if vectorize:
        for i in range(num_env):
            envs.append(env)
            print(env.state())
        return envs, env.observation_space.shape

    else:
        env = VacuumWorld(grid_size)
        return env

# def take_action(ac, steps, greedy=False):

#     ep_threshold = 10.01 + (1.0 - 0.01) * math.exp(-1. * steps/100000.0)

#     dist = Categorical(ac)
#     if np.random.uniform(low=0.3, high=1.0, size=None) > ep_threshold:
#         action = torch.tensor([random.randrange(4)], device='cuda', dtype=torch.long)
#     else:
#         action = dist.sample()

#     logp = dist.log_prob(action)
#     entropy = dist.entropy()

#     return action.cpu().numpy(), logp, entropy

def take_action(a):

    # print(a)
    dist = Categorical(a)
    action = dist.sample()
    # print(action)
    logp = dist.log_prob(action)
    entropy = dist.entropy()

    return action.cpu().numpy(), logp, entropy

def take_step(actions, envs, device):

    x_, reward_, done_, ep_info_ = [], [], [], []
    for a, e in zip(actions, envs):
        x, reward, done, ep_info = e.step(a)
        # print(e.action_meanings[a])
        if done:
            x_.append(e.reset())
            e.rw_dirts = e.dirts
            # print(e.dirts)
            # print(e.rw_dirts, 'After Reset')
        else:
            x_.append(x)
        reward_.append(reward)
        done_.append(done)
        ep_info_.append(ep_info)

    x_ = torch.from_numpy(np.array(x_)).float().to(device)

    return x_, np.array(reward_), np.array(done_), ep_info_


def init_hidden(num_workers, dim, device, grad=False, rand=False):
    if rand:
        return torch.rand(num_workers, dim, requires_grad=grad).to(device)
    else:
        return torch.zeros(num_workers, dim, requires_grad=grad).to(device)

# def init_hidden(n_workers, h_dim, device, grad=False):
#     return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
#             torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))


def init_obj(dim, hist, device):
    goals = [torch.zeros(1, dim, requires_grad=True).to(device) for _ in range(hist)]
    m_states = [torch.zeros(1, dim).to(device) for _ in range(hist)]
    return goals, m_states


def weight_init(layer):
    if type(layer) == torch.nn.modules.conv.Conv2d or \
            type(layer) == torch.nn.Linear:
        torch.nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)


def normalize_input(p):
    return p/255


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + (gamma * R * masks[step])
        returns.insert(0, R)
    return returns
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
import numpy as np
import random
import numpy as np
from vwgym import VacuumWorld, Vectorise, StepWrapper
from torch.distributions import Categorical

class mini_db:
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
        

def make_env(grid_size, vectorize=True, random_seed=False):

    """
    Initializing the VacuumWorld Environment
    """
    if random_seed:
        random.seed(518123)

    if vectorize:
        env = Vectorise(VacuumWorld(grid_size))
        env = StepWrapper(env)

        return env, env.observation_space.shape
    else:
        env = VacuumWorld(grid_size)
        return env


def take_action(a):
    dist = Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()

    return action.cpu().numpy(), logp, entropy


def init_hidden(dim, device, grad=False, rand=False):
    if rand:
        return torch.rand(1, dim, requires_grad=grad).to(device)
    else:
        return torch.zeros(1, dim, requires_grad=grad).to(device)


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
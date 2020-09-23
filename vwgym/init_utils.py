#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import numpy as np
from vwgym import VacuumWorld, Vectorise, StepWrapper
from torch.distributions import Categorical

def make_env(grid_size, num_env=1, vectorize=True, sparse=1, random_seed=518123):

    """
    Initializing the VacuumWorld Environment
    seeds = [518123, 123518, 123, 518,  1, 2, 0]

    """
    random.seed(random_seed)

    env = Vectorise(VacuumWorld(grid_size, dirts=sparse))
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


def take_action(a):

    dist = Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().numpy(), logp, entropy

def take_step(actions, envs, device):

    x_, reward_, done_, ep_info_ = [], [], [], []
    for a, e in zip(actions, envs):
        x, reward, done, ep_info = e.step(a)
        if done:
            x_.append(e.reset())
            e.rw_dirts = e.dirts
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


def compute_returns(next_value, rewards, masks, gamma=0.99):
    r_t = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        r_t = rewards[step] + (gamma * r_t * masks[step])
        returns.insert(0, r_t)
    return returns
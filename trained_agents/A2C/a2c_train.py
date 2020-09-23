#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:10:52 2020

@author: Krishnendu S. Kar
"""

import os, shutil
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from datetime import datetime
from vwgym.init_utils import make_env
from actor_critic_a2c import *
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=518123,
                    help='random seed')

parser.add_argument('--sparse', type=int, default=1,
                    help='less dirts or grid_size*2')

parser.add_argument('--grid_size', type=int, default=3,
                    help='grid_size')

term_args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env, input_shape = make_env(grid_size=term_args.grid_size,
                            sparse=term_args.sparse,
                            random_seed=term_args.seed)

torch.manual_seed(518123)
env = env[0]
action_size = env.action_space.n
steps = 0


def trainIters(actor, critic, n_iters):
    global steps

    log_dir = 'a2c_train_logs/'

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    optimizerA = optim.RMSprop(actor.parameters(), lr=1e-4)
    optimizerC = optim.RMSprop(critic.parameters(), lr=1e-4)
    for i_episode in range(n_iters):
        state = env.reset()
        env.rw_dirts = env.dirts
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for i in count():
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
            dist, value = actor(state), critic(state)
            action, log_prob, etr = select_action(dist, steps)
            next_state, reward, done, ep_info = env.step(action.item())

            entropy += etr.mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state
            steps += 1
            if done:
                writer.add_scalars('episode/reward', {'reward': ep_info['ep_rewards']}, i_episode)
                writer.add_scalars('episode/length', {'length': ep_info['ep_len']}, i_episode)
                break

        next_state = torch.from_numpy(next_state.reshape(1, -1)).float().to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -1 * (log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        writer.add_scalars('loss/actor_loss', {'actor_loss': actor_loss}, steps)
        writer.add_scalars('loss/critic_loss', {'critic_loss': critic_loss}, steps)

        actor_loss.backward()
        critic_loss.backward()

        optimizerA.step()
        optimizerC.step()

        if i_episode % 10 == 0:
            torch.save(actor, 'saved_model/actor.pt')
            torch.save(critic, 'saved_model/critic.pt')

    torch.save(actor, 'model/actor.pt')
    torch.save(critic, 'model/critic.pt')

if __name__ == '__main__':
    if os.path.exists('model/actor.pt'):
        actor = torch.load('model/actor.pt')
        print('Actor Model loaded')
    else:
        actor = Actor(input_shape, action_size).to(device)
    if os.path.exists('model/critic.pt'):
        critic = torch.load('model/critic.pt')

        print('Critic Model loaded')
    else:
        critic = Critic(input_shape).to(device)
    trainIters(actor, critic, n_iters=50000)

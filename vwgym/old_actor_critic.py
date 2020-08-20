#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:10:52 2020

@author: archie
"""

import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from datetime import datetime
from vwgym.init_utils import make_env, take_action
import math
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env, input_shape = make_env(3, 1)
# channels, h, w = input_shape
state_size = input_shape #(channels * h * w)
torch.manual_seed(518123)
env = env[0]
action_size = env.action_space.n
lr = 0.0001
steps = 0

class Actor(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Actor, self).__init__()
        channels, height, width = state_size

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
    def __init__(self, state_size):
        super(Critic, self).__init__()
        channels, height, width = state_size

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
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + (gamma * R * masks[step])
        returns.insert(0, R)
    return returns

def select_action(action_probs, steps):

    dist = Categorical(action_probs)
    ep_threshold = 10.01 + (1.0 - 0.01) * math.exp(-1. * steps/50000.0)

    if np.random.uniform(low=0.3, high=1.0, size=None) > ep_threshold:
        action = action_probs.max(1)[1]
    else:
        action = dist.sample()
        # action = torch.tensor([random.randrange(4)], device='cuda', dtype=torch.long)
        # print('Not ArgMax..', ep_threshold, steps)

    logp = dist.log_prob(action)
    entropy = dist.entropy()

    return action.cpu().numpy(), logp, entropy

def trainIters(actor, critic, n_iters):
    global steps

    dtm = datetime.now()
    log_dir = 'ac_logs_{}'.format('_'.join([str(dtm.date()), str(dtm.time())]))
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
        # env.reset()

        for i in count():
            # env.render()
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
            dist, value = actor(state), critic(state)
            # action = dist.sample()
            action, log_prob, etr = select_action(dist, steps)
            next_state, reward, done, ep_info = env.step(action.item())

            # log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += etr.mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state
            steps += 1
            if done:
                # print('Iteration: {}, Score: {}'.format(i, ))
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
            torch.save(actor, f'model/actor_{i_episode}.pt')
            torch.save(critic, f'model/critic_{i_episode}.pt')

    torch.save(actor, f'model/actor.pt')
    torch.save(critic, f'model/critic.pt')

if __name__ == '__main__':
    if os.path.exists('model/actor.pt'):
        # actor = Actor(state_size, action_size).to(device)
        # actor.load_state_dict('model/actor.pt')
        actor = torch.load('model/actor.pt')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pt'):
        # critic = Critic(state_size, action_size).to(device)
        # critic.load_state_dict('model/critic.pt')
        critic = torch.load('model/critic.pt')

        print('Critic Model loaded')
    else:
        critic = Critic(state_size).to(device)
    trainIters(actor, critic, n_iters=50000)

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

from vwgym.init_utils import make_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env, input_shape = make_env(3, 1)
channels, h, w = input_shape
state_size = channels * h * w

env = env[0]
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        # self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(32, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        # output = F.relu(self.linear2(output))
        output = self.linear3(output)
        d = F.softmax(output, dim=-1)
        distribution = Categorical(d)
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        # self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        # output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
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
            state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, ep_info = env.step(action.cpu().numpy()[0])

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                # print('Iteration: {}, Score: {}'.format(i, ))
                break


        next_state = torch.from_numpy(next_state.reshape(1, -1)).float().to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('model/critic.pkl'):
    #     critic = torch.load('model/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=100)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:08:44 2020

@author: archie
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss as huber
from vwgym.init_utils import *

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

class actor_critic(nn.Module):

    def __init__(self, input_shape, num_actions, device):
        super(actor_critic, self).__init__()

        self.device = device

        channels, height, width = input_shape
        self.d = (channels * height * width) + 4

        self.fc1 = nn.Linear(self.d, 1024)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.1)

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

        self.to(device)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        actions_prob = F.softmax(self.actor(x), dim=1)
        critic = self.critic(x)
        return actions_prob, critic

# Configuration parameters for the whole setup
seed = 518123
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 40000


env, input_shape = make_env(3, 1)
env = env[0]
env.rw_dirts = env.dirts
n_actions = env.action_space.n

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

ac = actor_critic(input_shape, n_actions, device)

optimizer = optim.Adam(ac.parameters(), lr=1e-4)
huber_loss = huber

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
steps = 0

dtm = datetime.now()
log_dir = 'ac_logs_{}'.format('_'.join([str(dtm.date()), str(dtm.time())]))
writer = SummaryWriter(log_dir=log_dir)

while True:  # Run until solved

    episode_reward = 0

    state = env.reset()
    print('reset')
    env.rw_dirts = env.dirts

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.

        state = torch.from_numpy(state).float()
        state = state.view(1, -1).to(device)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = ac(state)
        critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        ep_threshold = 0.01 + (1.0 - 0.01) * math.exp(-1. * steps/50000.0)
        sample = np.random.uniform(low=0.2, high=1.0, size=None)
        if sample > ep_threshold:
            action = torch.argmax(action_probs, dim=1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            print('Not ArgMax..', ep_threshold, sample, steps)

        action_probs_history.append(torch.log(action_probs[0, action]))

        # Apply the sampled action in our environment
        state, reward, done, ep_info = env.step(action.cpu().numpy()[0])
        rewards_history.append(reward)
        episode_reward += reward
        steps += 1

        if done:
            print('Episode No.:\t', episode_count)
            writer.add_scalars('episode/reward', {'reward': ep_info['ep_rewards']}, episode_count)
            writer.add_scalars('episode/length', {'length': ep_info['ep_len']}, episode_count)
            break

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    writer.add_scalars('episode/running_reward', {'avg_reward': running_reward}, episode_count)

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
    returns = torch.from_numpy(returns).float().to(device)

    # Calculating loss values to update our network
    history = zip(action_probs_history, critic_value_history, returns)
    actor_losses = []
    critic_losses = []
    for log_prob, value, ret in history:
        # At this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. We took an action with log probability
        # of `log_prob` and ended up recieving a total reward = `ret`.
        # The actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = ret - value
        actor_losses.append(log_prob * diff * -1)  # actor loss

        # The critic must be updated so that it predicts a better estimate of
        # the future rewards.
        critic_losses.append(
            huber_loss(torch.tensor(ret), value)
        )

    # Backpropagation
    loss_value = torch.stack(actor_losses).mean() + torch.stack(critic_losses).mean()
    writer.add_scalars('loss/total_loss',{'total_loss': loss_value},episode_count)
    writer.add_scalars('loss/actor_loss',{'actor_loss': torch.stack(actor_losses).sum()},episode_count)
    writer.add_scalars('loss/critic_loss',{'critic_loss': torch.stack(critic_losses).sum()},episode_count)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        # template = "\n running reward: {:.2f} at episode {}\n"
        # print(template.format(running_reward, episode_count))
        torch.save({'model': ac.state_dict(),
                    'optim': optimizer.state_dict()},
                   f'model/vwgym_ac_{episode_count}_ckpt.pt')

    if running_reward > 520:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

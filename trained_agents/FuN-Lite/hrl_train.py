#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:22:09 2020

@author: Krishnendu S. Kar
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from vwgym.init_utils import *
from fun_lite import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import numpy as np

import os, shutil
from itertools import count
import argparse

parser = argparse.ArgumentParser(description='FuN_Lite')
parser.add_argument('--seed', type=int, default=518123,
                    help='random seed')

parser.add_argument('--sparse', type=int, default=1,
                    help='less dirts or grid_size*2')

parser.add_argument('--grid_size', type=int, default=3,
                    help='grid_size')

term_args = parser.parse_args()

torch.manual_seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = 'cpu'

agent_args = {'lr': 1e-4,
        'max_steps': 1e7,
        'entropy_coef': 0.01,
        'gamma_w': 0.95,
        'gamma_m': 0.999,
        'alpha': 0.85,
        'eps': 0.001,
        'd': 256,
        'k': 16,
        'len_hist': 10,
        'grad_clip':5.0,
        'writer': True,
        'num_worker':1}

def train(args, t_args, device):
    env, input_shape = make_env(grid_size=t_args.grid_size,
                                sparse=t_args.sparse,
                                random_seed=t_args.seed)
    env = env[0]
    num_actions = env.action_space.n
    steps = 0

    log_dir = 'hrl_train_logs/'

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    f_net = FuNet(input_shape, args['d'],
                    args['len_hist'],
                    args['eps'],
                    args['k'],
                    num_actions,
                    args['num_worker'],
                    device)

    optimizer = optim.RMSprop(f_net.parameters(), lr=1e-3)
    goal_history, s_Mt_hist, ep_binary = f_net.agent_model_init()

    for ep_num in count():
        state = env.reset()
        env.rw_dirts = env.dirts

        goal_history = [g.detach() for g in goal_history]

        mini_db = {'log_probs':[],
                   'values_manager':[],
                   'values_worker':[],
                   'rewards':[],
                   'intrinsic_rewards':[],
                   'goal_errors':[],
                   'masks':[],
                   'entropy':0}

        for i in count():

            state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
            action_probs, v_Mt, v_Wt, goal_history, s_Mt_hist = f_net(state, goal_history, s_Mt_hist)
            a_t, log_p, etr = take_action(action_probs)
            next_state, reward, done, ep_info = env.step(a_t.item())

            ep = torch.FloatTensor([1.0 - done]).unsqueeze(-1).to(device)

            ep_binary.pop(0)
            ep_binary.append(ep)

            mini_db['entropy'] += etr.mean()

            mini_db['log_probs'].append(log_p.unsqueeze(-1))
            mini_db['values_manager'].append(v_Mt)
            mini_db['values_worker'].append(v_Wt)
            mini_db['rewards'].append(torch.tensor([[reward]], dtype=torch.float, device=device))
            mini_db['masks'].append(torch.tensor([[1-done]], dtype=torch.float, device=device))
            mini_db['intrinsic_rewards'].append(f_net.int_reward(goal_history, s_Mt_hist, ep_binary).unsqueeze(-1))
            mini_db['goal_errors'].append(f_net.del_g_theta(goal_history, s_Mt_hist, ep_binary))

            state = next_state
            steps += 1

            if done:
                writer.add_scalars('episode/reward', {'reward': ep_info['ep_rewards']}, ep_num)
                writer.add_scalars('episode/length', {'length': ep_info['ep_len']}, ep_num)
                break

        next_state = torch.from_numpy(next_state.reshape(1, -1)).float().to(device)
        _, v_Mtp1, v_Wtp1, _, _ = f_net(next_state, goal_history, s_Mt_hist)

        ret_m = compute_returns(v_Mtp1, mini_db['rewards'], mini_db['masks'], args['gamma_m'])
        ret_w = compute_returns(v_Wtp1, mini_db['rewards'], mini_db['masks'], args['gamma_w'])

        log_probs = torch.cat(mini_db['log_probs'])
        ret_m = torch.cat(ret_m).detach()
        ret_w = torch.cat(ret_w).detach()
        intrinsic_rewards = torch.cat(mini_db['intrinsic_rewards'])
        goal_errors = torch.cat(mini_db['goal_errors'])

        value_m = torch.cat(mini_db['values_manager'])
        value_w = torch.cat(mini_db['values_worker'])

        advantage_m = ret_m - value_m
        advantage_w = (ret_w + args['alpha'] * intrinsic_rewards) - value_w

        loss_manager = -1 * (goal_errors * advantage_m.detach()).mean()
        loss_worker = -1 * (log_probs * advantage_w.detach()).mean()

        value_m_loss = 0.5 * advantage_m.pow(2).mean()
        value_w_loss = 0.5 * advantage_w.pow(2).mean()

        loss = loss_worker + loss_manager + value_w_loss + value_m_loss - (args['entropy_coef'] * mini_db['entropy'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalars('loss/total_loss',{'total_loss': loss}, ep_num)
        writer.add_scalars('loss/manager_loss',{'manager_loss': loss_manager}, ep_num)
        writer.add_scalars('loss/worker_loss',{'worker_loss': loss_worker}, ep_num)
        writer.add_scalars('loss/worker_value_fn_loss',{'worker_value_func_loss': value_w_loss}, ep_num)
        writer.add_scalars('loss/manager_value_fn_loss',{'man_value_func_loss': value_m_loss}, ep_num)

        if ep_num % 1000 == 0:
            torch.save({'model': f_net.state_dict(),
                        'args':args,
                        'goal': goal_history,
                        'man_state': s_Mt_hist},
                       'saved_model/fnet_ckpt.pt')

if __name__ == '__main__':
    train(agent_args, term_args, device)

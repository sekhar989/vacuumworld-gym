#!/usr/bin/env python
# coding: utf-8

"""
Test Script

"""
import random
import numpy as np

from vwgym.fun_lite import FuNet
from vwgym.init_utils import take_action, take_step, make_env
import torch
from torch.distributions import Categorical
random.seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'


model_path = '/home/archie/Documents/RHUL/Term_01/CS5940-IA/HRL/vacuumworld-gym/vwgym/saved_model/vwgym_f_net_two_last_ckpt_632200.pt'

checkpoint = torch.load(model_path)

args = checkpoint['args']

env, input_shape = make_env(grid_size=args['grid_size'], num_env=1)

print(env[0].state())

num_actions = env[0].action_space.n
f_net = FuNet(input_shape, args['d'], args['len_hist'], args['eps'], args['k'], num_actions, 1, device)

f_net.load_state_dict(checkpoint['model'])

f_net

def select_action(action_probs):

    # dist = Categorical(action_probs)
    # print(action_probs)
    # action = dist.sample()
    action = action_probs.max(1)[1]

    return action.cpu().numpy()

with torch.no_grad():

    f_net.eval()
    predictions = []

    goal_history, s_Mt_hist, ep_binary = f_net.agent_model_init()
    x = np.array([e.reset() for e in env])

    for e in env:
        e.rw_dirts = e.dirts
        print(f'Dirts Present..:{e.rw_dirts}')

    x = torch.from_numpy(x).float().to(device)
    step = 0
    prev_action = []

    for __ in range(5000):
        action_probs, _, _, _, _  = f_net(x, goal_history, s_Mt_hist)
        a_t = select_action(action_probs)
        x, reward, done, ep_info = take_step(a_t, env, device)
        for ep_d in ep_info:
            if ep_d['ep_rewards'] is not None:
                print(f"reward = {round(ep_d['ep_rewards'], 2)} \t| ep_score = {round(600/ep_d['ep_len'], 2)}")
        predictions.append(a_t[0])
        if done:
            break

import pandas as pd

p = pd.DataFrame.from_dict({'actions': predictions, 'action_meanings':[env[0].action_meanings[i] for i in predictions]})
print(p['action_meanings'].value_counts().reset_index())
print(p['action_meanings'])


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 08-07-2020
"""

import torch

from vwgym.init_utils import *
from vwgym.fun_lite import *
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm
import os

torch.manual_seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'

args = {'lr': 1e-4,
        'steps': 51,
        'max_steps': 1e7,
        'env_reboot':5e5,
        'entropy_coef': 0.01,
        'gamma_w': 0.95,
        'gamma_m': 0.999,
        'alpha': 0.85,
        'eps': 0.001,
        'grid_size': 3,
        'd': 256,
        'k': 16,
        'len_hist': 10,
        'grad_clip':5.0,
        'writer': True,
        'num_worker': 16}


def train(args, device, reload=False):

    env, input_shape = make_env(args['grid_size'], args['num_worker'])
    num_actions = env[0].action_space.n
    if reload:
        model_path = 'saved_model/vwgym_f_net_last_ckpt.pt'
        checkpoint = torch.load(model_path)
        f_net = FuNet(input_shape, args['d'],
                    args['len_hist'],
                    args['eps'],
                    args['k'],
                    num_actions,
                    args['num_worker'],
                    device)
        f_net.load_state_dict(checkpoint['model'])
    else:
        f_net = FuNet(input_shape, args['d'],
                    args['len_hist'],
                    args['eps'],
                    args['k'],
                    num_actions,
                    args['num_worker'],
                    device)

    print('Model Summary...\n')
    print('-'*400)
    print(f_net)
    print('*'*400)
    optimizer = torch.optim.RMSprop(f_net.parameters(), lr=args['lr'])
    goal_history, s_Mt_hist, ep_binary = f_net.agent_model_init()

    x = np.array([e.reset() for e in env])
    x = torch.from_numpy(x).float().to(device)
    for e in env:
        e.rw_dirts = e.dirts
        print(f'Dirts Present..:{e.rw_dirts}')

    step = 0
    switch = 0.
    tr_ep = 0
    score = []


    if args['writer']:
        dtm = datetime.now()
        log_dir = 'train_logs_{}'.format('_'.join([str(dtm.date()), str(dtm.time())]))
        writer = SummaryWriter(log_dir=log_dir)

    while step < args['max_steps']:

        goal_history = [g.detach() for g in goal_history]

        db = mini_db(keys=[ 'ext_r_i', 'int_r_i',
                            'v_m', 'v_w',
                            'rt_w', 'rt_m',
                            'logp', 'entropy',
                            'goal_error', 'ep_indicator',
                            'adv_m', 'adv_w' ], space=args['steps'])

        # for __ in range(args['steps']):
        fin = False

        # while not fin:
        for __ in range(args['steps']):
            optimizer.zero_grad()

            action_probs, v_Mt, v_Wt, goal_history, s_Mt_hist = f_net(x, goal_history, s_Mt_hist)
            a_t, log_p, entropy = take_action(action_probs, step)

            x, reward, done, ep_info = take_step(a_t, env, device)
            tr_ep += args['num_worker']

            for ep_d in ep_info:
                if ep_d['ep_rewards'] is not None:
                    # print(f'Ep. {tr_ep} Completed .. | Steps {step} {datetime.now().time()}')
                    # print(f"reward = {round(ep_d['ep_rewards'], 2)} \t| ep_score = {round(600/ep_d['ep_len'], 2)}")
                    writer.add_scalars('episode/rewards', {'rewards': ep_d['ep_rewards']}, step)
                    writer.add_scalars('episode/length', {'length': ep_d['ep_len']}, step)

            ep = torch.FloatTensor(1.0 - done).unsqueeze(-1).to(device)

            ep_binary.pop(0)
            ep_binary.append(ep)

            db.update({'ext_r_i': torch.FloatTensor(reward).to(device).unsqueeze(-1),
                    'int_r_i': f_net.int_reward(goal_history, s_Mt_hist, ep_binary).unsqueeze(-1),
                    'v_m': v_Mt,
                    'v_w': v_Wt,
                    'logp': log_p.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    'goal_error': f_net.del_g_theta(goal_history, s_Mt_hist, ep_binary),
                    'ep_indicator': ep
                    })

            step += 1
            # x = x_next

            # if done[0]:
            #     fin = True

        with torch.no_grad():
            # x_next = torch.from_numpy(x_next).float().to(device)
            _, v_Mtp1, v_Wtp1, _, _ = f_net(x, goal_history, s_Mt_hist)
            v_Mtp1 = v_Mtp1.detach()
            v_Wtp1 = v_Wtp1.detach()

        loss, loss_summary = loss_function(db, v_Mtp1, v_Wtp1, args)

        if args['writer']:
            writer.add_scalars('loss/total_loss',{'total_loss': loss_summary['loss/total_fun_loss']},step)
            writer.add_scalars('loss/manager_loss',{'manager_loss': loss_summary['loss/manager']},step)
            writer.add_scalars('loss/worker_loss',{'worker_loss': loss_summary['loss/worker']},step)

            writer.add_scalars('loss/worker_value_fn_loss',{'worker_value_func_loss': loss_summary['loss/value_worker']},step)
            writer.add_scalars('loss/manager_value_fn_loss',{'man_value_func_loss': loss_summary['loss/value_manager']},step)

        if step % 1000 == 0:
            torch.save({
                'model': f_net.state_dict(),
                'args': args,
                # 'processor_mean': f_net.pre_.rms.mean,
                'optim': optimizer.state_dict()},
                f'saved_model/vwgym_f_net_last_ckpt_{step}.pt')

        loss.backward()
        optimizer.step()


    torch.save({
        'model': f_net.state_dict(),
        'args': args,
        'processor_mean': f_net.pre_.rms.mean,
        'optim': optimizer.state_dict()},
        'saved_model/vwgym_f_net.pt')


if __name__ == '__main__':
    if os.path.exists('saved_model/vwgym_f_net_last_ckpt.pt'):
        print('Model Reloaded..')
        train(args, device, reload=True)
    else:
        train(args, device)
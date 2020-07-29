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

torch.manual_seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'

args = {
            'lr': 25e-4,
            'steps': 400,
            'max_steps': 1e8,
            'env_reboot':5e5,
            'entropy_coef': 0.01,
            'gamma_w': 0.95,
            'gamma_m': 0.999,
            'alpha': 0.85,
            'eps': 0.1,
            'grid_size': 8,
            'd': 256,
            'k': 16,
            'len_hist': 10,
            'grad_clip':5.0,
            'writer': True
        }


def train(args, device):

    env, input_shape = make_env(args['grid_size'])
    # print(input_shape)
    num_actions = env.action_space.n

    f_net = FuNet(input_shape, args['d'], args['len_hist'], args['eps'], args['k'], num_actions, device)
    optimizer = torch.optim.Adam(f_net.parameters(), lr=args['lr'], eps=1e-5)
    goal_history, s_Mt_hist, ep_binary = f_net.agent_model_init()

    # if last_checkpoint:
    #     f_net.load

    prev_x = env.reset()

    # reset_history = [prev_x]
    x = torch.from_numpy(prev_x).float()
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

        for __ in tqdm(range(args['steps'])):

            action_probs, v_Mt, v_Wt, goal_history, s_Mt_hist = f_net(x, goal_history, s_Mt_hist)
            # print(action_probs)
            a_t, log_p, entropy = take_action(action_probs)
            # print('previous:\n', prev_x)
            x, reward, done, ep_info = env.step(prev_x, a_t[0])

            # print('Done\t:', done)
            # print('action:\t', env.action_meanings[a_t])
            # print('post:\n', x)
            # print('-'*50)
            if done:
                if step//args['env_reboot'] > switch:
                    env, _ = make_env(args['grid_size'])
                    x = env.reset()
                    switch = step//args['env_reboot']
                    print('Env Switch....')
                else:
                    x = env.reset()
                
                prev_x = x
                # reset_history.append(prev_x)
                score.append(600/ep_info['ep_len'])
                score  = score[-100:]
                if args['writer']:
                    writer.add_scalars('rewards/ep_rewards', {'rewards': ep_info['ep_rewards']}, step)
                    writer.add_scalars('rewards/avg_score', {'score': np.mean(score[-100:])}, step)
                    writer.add_scalars('action/action_summary', {'move': ep_info['move']}, step)
                    writer.add_scalars('action/action_summary', {'turn_left': ep_info['turn_left']}, step)
                    writer.add_scalars('action/action_summary', {'turn_right': ep_info['turn_right']}, step)
                    writer.add_scalars('action/action_summary', {'clean': ep_info['clean']}, step)
                    writer.add_scalars('action/action_summary', {'idle': ep_info['idle']}, step)

                tr_ep += 1
                print(f'Ep. {tr_ep} Completed')
                print(f"total steps = {step} \t| reward = {round(ep_info['ep_rewards'], 2)} \t| ep_score = {round(600/ep_info['ep_len'], 2)} \t| last 100 mean score = {round(np.mean(score[-100:]), 2)}")
                # f"> ep = {self.n_eps} |
            else:
                prev_x = x
            x = torch.from_numpy(x).float()
            ep = torch.FloatTensor([1.0 - done]).to(device)
            # print('EpisodeStatus:\t', ep)
            ep_binary.pop(0)
            ep_binary.append(ep.view(1, 1))

            # print('ext_r_i', torch.FloatTensor([reward]).to(device))
            # print('int_r_i', f_net.int_reward(goal_history, s_Mt_hist, ep_binary),)
            # print('v_m', v_Mt,)
            # print('v_w', v_Wt,)
            # print('logp', log_p)
            # print('entropy', entropy)
            # print('goal_error', f_net.del_g_theta(goal_history, s_Mt_hist, ep_binary),)
            # print('ep_indicator', ep)

            # print('---------------------------------------------------------------------')

            db.update({'ext_r_i': torch.FloatTensor([reward]).to(device).unsqueeze(0),
                    'int_r_i': f_net.int_reward(goal_history, s_Mt_hist, ep_binary).unsqueeze(0),
                    'v_m': v_Mt.unsqueeze(0),
                    'v_w': v_Wt.unsqueeze(0),
                    'logp': log_p.unsqueeze(0),
                    'entropy': entropy.unsqueeze(0),
                    'goal_error': f_net.del_g_theta(goal_history, s_Mt_hist, ep_binary),
                    'ep_indicator': ep.unsqueeze(0)
                    })

            step += 1
            # print(step)
            # ep_binary = [e.detach() for e in ep_binary]


        with torch.no_grad():
            _, v_Mtp1, v_Wtp1, _, _ = f_net(x, goal_history, s_Mt_hist)
            v_Mtp1 = v_Mtp1.detach()
            v_Wtp1 = v_Wtp1.detach()

        optimizer.zero_grad()
        loss, loss_summary = loss_function(db, v_Mtp1, v_Wtp1, args)

        # print('Sleeping.....')
        # time.sleep(600)
        
        if args['writer']:
            writer.add_scalars('loss/total_loss',{'total_loss': loss_summary['loss/total_fun_loss']},step)
            writer.add_scalars('loss/manager_loss',{'manager_loss': loss_summary['loss/manager']},step)
            writer.add_scalars('loss/worker_loss',{'worker_loss': loss_summary['loss/worker']},step)

        if step % 1e5 == 0:
            torch.save({
                'model': f_net.state_dict(),
                'args': args,
                'processor_mean': f_net.pre_.rms.mean,
                'optim': optimizer.state_dict()},
                f'saved_model/vwgym_f_net_last_checkpoint_02.pt')

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(f_net.parameters(), args['grad_clip'])
        optimizer.step()

    torch.save({
        'model': f_net.state_dict(),
        'args': args,
        'processor_mean': f_net.pre_.rms.mean,
        'optim': optimizer.state_dict()},
        f'saved_model/vwgym_f_net.pt')

    # for h, rh in enumerate(reset_history):
    #     print('run number..\t', h)
    #     print(rh)
    #     print('*'*50)

if __name__ == '__main__':
    train(args, device)
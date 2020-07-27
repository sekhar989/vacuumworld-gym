#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 08-07-2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.functional import cosine_similarity as dcos, normalize
from vwgym.pre_process import *
from vwgym.init_utils import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Percept(nn.Module):
    """
    f_percept(x_t) = z_t

    """
    def __init__(self, input_shape, d, device):
        super(Percept, self).__init__()
        
        # self.percept = nn.Sequential(
        #                                 nn.Linear((input_shape[-1]**2)*3, 128),
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, d),
        #                                 nn.ReLU()
                                    # )
        channels, height, width = input_shape
        
        # percept_linear_in = 16 * int((int((height - 3) / 3) - 2) / 2) * int((int((width - 3) / 3) - 2) / 2)
        h_out = int((height - (3 - 1) ))
        w_out = int((width - (3 - 1) ))

        self.percept = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = 16, kernel_size= (3, 3)),
            nn.Tanh(),
            Flatten(),
            nn.Linear(h_out * w_out * 16, d),
            nn.ReLU()
        )

    def forward(self, x):
        # x = x.view(-1)
        z = self.percept(x.unsqueeze(0))
        ## print('Inside Percept XXXXXXXXXXXXXXX:\n', z)
        return z.view(-1)


class Manager(nn.Module):
    """
    f_stateM(z_t) = s_t             ## Manager Internal State Representation
    goal_gen(s_t-1, s_t) = g_t      ## Manager Goal Generation
    f_valueM(g_t) = vM_t            ## Manager Value Function

    """
    def __init__(self, d, epsilon, device):
        super(Manager, self).__init__()

        self.d = d                  ## Hidden Internal Dimension
        self.d2 = d*2
        self.dd = 32
        self.epsilon = epsilon

        self.M_space = nn.Linear(self.d, self.d)
        self.M_relu = nn.ReLU()
        self.M_goals = nn.Linear(self.d2, self.d)
        self.M_value = nn.Linear(self.d, 1)
        # self.M_value = nn.Sequential(
        #                                 nn.Linear(self.d, self.dd),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.dd, 1)
        #                             )
        self.device = device

    def forward(self, z_precept, s_tm1):

        s_t = self.M_relu(self.M_space(z_precept))      ## Manager Internal State Representation
        ## print('w_shape:\t', s_tm1.shape, 'Ut_shape', s_t.shape)
        M_int_state = torch.cat((s_tm1, s_t))          ## Concat of previous state (s_{t-1} and s_{t})

        goal = self.M_relu(self.M_goals(M_int_state))   ## Goal generation using prev step and current state
        v_Mt = self.M_value(goal)                       ## Value function

        goal = normalize(goal.view(1, goal.shape[0]))
        s_t = s_t.detach()

        if (self.epsilon > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        return v_Mt, goal, s_t                         ## Manager Value, Manager Goal &  Manager state (s_{t})

    def goal_error(self, s_tp1, s_t, g_t, ep_indicator):
        """
        error = cos(s_{t+1} - s_{t}, g_{t})

        """
        # print(s_tp1)
        cosine_dist = dcos(s_tp1 - s_t, g_t)
        cosine_dist = ep_indicator * cosine_dist.unsqueeze(0)

        return cosine_dist

class Worker(nn.Module):
    """
    f_stateW(s_tm1, z_t) = W_u_t                        ## Worker Policy
    phi(goals) = w                                      ## Manager Goals sent to Worker
    torch.einsum("bk, bka -> ba", w, u_t)               ## Policy (U) X Goals (w)
    softmax(a)                                          ## Action Probabilities
    f_valueW(u_t) = vW_t                                ## Worker Value Function
    """
    def __init__(self, d, k, num_actions, device):
        super(Worker, self).__init__()


        self.num_actions = num_actions
        self.k = k
        self.d = d
        self.d2 = 128
        self.dd = 32
        self.knum2 = 20
        
        # self.f_stateW = nn.Linear(self.d, k*num_actions)
        self.f_stateW = nn.Sequential(
                                        nn.Linear(self.d, self.d2),
                                        nn.ReLU(),
                                        nn.Linear(self.d2, k*num_actions),
                                        nn.ReLU()
                                    )
        self.phi = nn.Linear(d, k, bias=False)
        self.f_valueW = nn.Sequential(
                                        nn.Linear(k*num_actions, self.knum2),
                                        nn.ReLU(),
                                        nn.Linear(self.knum2, 1)
                                    )
        self.W_relu = nn.ReLU()
        self.device = device


    def forward(self, z, goals):

        ## print(z)
        W_u_t = self.W_relu(self.f_stateW(z))
        v_Wt = self.f_valueW(W_u_t)
        
        g_t = torch.stack(goals).detach().sum(dim=0)
        # g_t = goals.detachz`()
        ## print(g_t, g_t.shape)

        w = self.phi(g_t)                                 ## phi(goals) = w ---- Manager Goals sent to Worker
        ## print(w.shape)
        W_u_t = W_u_t.view(self.k, self.num_actions)
        ## print(W_u_t.shape)
        
        ## print('w_shape:\t', w.shape, 'Ut_shape', W_u_t.shape)
        ## print(w)

        ## print(W_u_t)
        # a_t = torch.einsum("bk, bka -> ba", w, W_u_t).softmax(dim=-1)
        ## print(a_t)
        a_t = nn.functional.softmax(torch.mm(w, W_u_t), dim=1)
        
        return v_Wt, a_t


    def intrinsic_reward(self, len_hist, goal_history, manager_states, ep_indicator):

        t = len_hist-1
        r_i = torch.zeros(1, 1).to(self.device)
        ep_hist = torch.ones(1, 1).to(self.device)
        ## print(ep_indicator)
        
        for i in range(1, t+1):
            #print(i, t, t-i)  
            r_i_t = dcos(manager_states[t] - manager_states[t - i], goal_history[t - i])#.unsqueeze(-1)
            #print(r_i_t, ep_hist)
            r_i += (ep_hist * r_i_t)
            ep_hist = ep_hist * ep_indicator[t - i]
            ## print('post_update', ep_hist)

        r_i = r_i.detach().view(-1)
        #print('Final intrinsic_reward:\t', r_i)
        return r_i / len_hist


class FuNet(nn.Module):
    """

    """
    def __init__(self, input_shape, d, len_hist, epsilon, k, num_actions, device):
        super(FuNet, self).__init__()
        
        self.input_shape = input_shape
        
        self.d = d
        self.hist_lim = len_hist
        self.epsilon = epsilon
        self.device = device
        
        self.k = k
        self.num_actions = num_actions

        self.pre_ = Preprocessor(self.input_shape, self.device, True)
        self.f_percept = Percept(self.input_shape, self.d, self.device)
        self.manager = Manager(self.d, self.epsilon, self.device)
        self.worker = Worker(self.d, self.k, self.num_actions, self.device)

        self.to(device)
        self.apply(weight_init)
        

    def forward(self, x, goal_history, s_Mt_hist):
        
        x = self.pre_(x)
        z = self.f_percept(x)
        ## print('XXXXXXXXXXXXXXX\t', z)
        
        s_tM1 = s_Mt_hist[-1].view(-1)               ## Passing the immediate previous manager state

        v_Mt, g_t, s_Mt = self.manager(z, s_tM1)
        
        goal_history.append(g_t)
        s_Mt_hist.append(s_Mt.unsqueeze(0).detach())     ## No grad

        if len(goal_history) >= self.hist_lim:
            goal_history.pop(0)
            s_Mt_hist.pop(0)

        v_Wt, a_t = self.worker(z, goal_history)
        
        return a_t, v_Mt, v_Wt, goal_history, s_Mt_hist


    def agent_model_init(self):

        goal_history = [init_hidden(self.d, device=self.device, grad=True, rand=True) for _ in range(self.hist_lim)]
        s_Mt_hist = [init_hidden(self.d, device=self.device) for _ in range(self.hist_lim)]
        ep_indicators = [torch.ones(1, 1).to(self.device) for _ in range(self.hist_lim)]

        return goal_history, s_Mt_hist, ep_indicators

    def int_reward(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.worker.intrinsic_reward(self.hist_lim, goal_hist, s_Mt_hist, ep_indicator)

    def del_g_theta(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.manager.goal_error(s_Mt_hist[-1], s_Mt_hist[0], goal_hist[0], ep_indicator[0])


def loss_function(db, vMtp1, vWtp1, args):

    rt_m = vMtp1
    rt_w = vWtp1

    db.placeholder()  # Fill ret_m, ret_w with empty vals
    ## print('External Rewards:\n:', db.ext_r_i)
    ## print('ep_indicator:\n:', db.ep_indicator)
    ## print('Return Manager:\n', db.rt_m)
    ## print('Return Worker:\n', db.rt_m)

    
    for i in reversed(range(args['steps'])):
        ret_m = db.ext_r_i[i] + (args['gamma_m'] * rt_m * db.ep_indicator[i])
        ret_w = db.ext_r_i[i] + (args['gamma_w'] * rt_w * db.ep_indicator[i])
        # print(i, ret_m, ret_w)
        db.rt_m[i] = ret_m
        db.rt_w[i] = ret_w
        ## print()

        ## print('Return Manager:\n', db.rt_m)
        ## print('Return Worker:\n', db.rt_w)
    # Optionally, normalize the returns
    db.normalize(['rt_w', 'rt_m'])

    rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, \
        goal_errors = db.stack(['int_r_i', 
                                'v_m', 
                                'v_w', 
                                'rt_w', 
                                'rt_m',
                                'logp', 
                                'entropy', 
                                'goal_error'])

    # Calculate advantages, size B x T
    #print(rewards_intrinsic)
    #print(ret_m.shape)
    #print(ret_w.shape)
    advantage_w = ret_w + (args['alpha'] * rewards_intrinsic) - value_w
    advantage_m = ret_m - value_m

    #print('Adv Manager:\n', advantage_m)
    #print('Adv Worker:\n', advantage_w)
    # print('entropy\n', entropy)
    # print('logprobs\n', logps)
    # print('goal errors\n', goal_errors)

    loss_worker = (logps * advantage_w.detach()).mean()
    loss_manager = (goal_errors * advantage_m.detach()).mean()

    # print('loss_manager\n', loss_manager)
    # print('loss_worker\n', loss_worker)

    # Update the critics into the right direction
    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    # print('value_m_loss', value_m_loss)
    # print('value_w_loss', value_w_loss)

    entropy = entropy.mean()

    # print('entropy\n', entropy)

    loss = - loss_worker - loss_manager + value_w_loss + value_m_loss - (args['entropy_coef'] * entropy)

    # print('loss\n', loss)

    return loss, {'loss/total_fun_loss': loss.item(),
                  'loss/worker': loss_worker.item(),
                  'loss/manager': loss_manager.item(),
                  'loss/value_worker': value_w_loss.item(),
                  'loss/value_manager': value_m_loss.item(),
                  'worker/entropy': entropy.item(),
                  'worker/advantage': advantage_w.mean().item(),
                  'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
                  'manager/cosines': goal_errors.mean().item(),
                  'manager/advantage': advantage_m.mean().item()}
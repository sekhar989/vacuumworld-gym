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

        channels, height, width = input_shape
        input_dim = (channels * height * width) + 4

        self.percept = nn.Linear(in_features=input_dim, out_features=d)
        self.percept.weight.data.normal_(0, 0.1)

    def forward(self, x):

        z = F.leaky_relu(self.percept(x))
        return z


class Manager(nn.Module):
    """
    f_stateM(z_t) = s_t             ## Manager Internal State Representation
    goal_gen(s_t-1, s_t) = g_t      ## Manager Goal Generation
    f_valueM(g_t) = vM_t            ## Manager Value Function

    """
    def __init__(self, d, epsilon, device, len_hist):
        super(Manager, self).__init__()

        self.d = d                  ## Hidden Internal Dimension

        self.epsilon = epsilon
        self.hist_lim = len_hist

        self.M_space = nn.Linear(self.d, self.d)
        self.M_space.weight.data.normal_(0, 0.1)

        self.M_goals = nn.Linear(self.d, self.d)
        self.M_goals.weight.data.normal_(0, 0.1)

        self.fc1 = nn.Linear(self.d, 1024)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.1)

        self.M_value = nn.Linear(512, 1)
        self.M_value.weight.data.normal_(0, 0.1)

        self.device = device

    def forward(self, z_precept):

        s_t = F.leaky_relu(self.M_space(z_precept))      ## Manager Internal State Representation

        goal = F.leaky_relu(self.M_goals(s_t))   ## Goal generation using prev step and current state

        x = F.leaky_relu(self.fc1(goal))
        x = F.leaky_relu(self.fc2(x))
        v_Mt = self.M_value(x)                       ## Value function

        goal = normalize(goal)
        s_t = s_t.detach()

        if (self.epsilon > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        return v_Mt, goal, s_t                         ## Manager Value, Manager Goal &  Manager state (s_{t})

    def goal_error(self, s_t, g_t, ep_indicator):
        """
        error = cos(s_{t+history} - s_{t}, g_{t})

        """

        t = self.hist_lim
        ep_indicator = torch.stack(ep_indicator[t: t + self.hist_lim - 1]).prod(dim=0)
        cosine_dist = dcos(s_t[t + self.hist_lim] - s_t[t], g_t[t])
        cosine_dist = ep_indicator * cosine_dist.unsqueeze(-1)

        return cosine_dist


class Worker(nn.Module):
    """
    f_stateW(s_tm1, z_t) = W_u_t                        ## Worker Policy
    phi(goals) = w                                      ## Manager Goals sent to Worker
    torch.einsum("bk, bka -> ba", w, u_t)               ## Policy (U) X Goals (w)
    softmax(a)                                          ## Action Probabilities
    f_valueW(u_t) = vW_t                                ## Worker Value Function
    """
    def __init__(self, d, k, num_actions, device, len_hist, num_workers):
        super(Worker, self).__init__()


        self.num_actions = num_actions
        self.k = k
        self.d = d
        self.b = num_workers

        self.hist_lim = len_hist


        self.f_stateW = nn.Linear(self.d, k*num_actions)
        self.f_stateW.weight.data.normal_(0, 0.1)

        self.phi = nn.Linear(d, k, bias=False)
        self.phi.weight.data.normal_(0, 0.1)

        self.wfc1 = nn.Linear(k*num_actions, 1024)
        self.wfc1.weight.data.normal_(0, 0.1)

        self.wfc2 = nn.Linear(1024, 512)
        self.wfc2.weight.data.normal_(0, 0.1)

        self.W_value = nn.Linear(512, 1)
        self.W_value.weight.data.normal_(0, 0.1)

        self.device = device


    def forward(self, z, goals):

        W_u_t = F.leaky_relu(self.f_stateW(z))

        g_t = torch.stack(goals).detach().sum(dim=0)

        w = self.phi(g_t)                                 ## phi(goals) = w ---- Manager Goals sent to Worker

        x = F.leaky_relu(self.wfc1(W_u_t))
        x = F.leaky_relu(self.wfc2(x))
        v_Wt = self.W_value(x)

        W_u_t = W_u_t.view(W_u_t.shape[0], self.k, self.num_actions)
        a_t = F.softmax((torch.einsum("bk, bka -> ba", w, W_u_t)), dim=1)

        return v_Wt, a_t#, w


    def intrinsic_reward(self, goal_history, manager_states, ep_indicator):

        t = self.hist_lim
        r_i = torch.zeros(self.b, 1).to(self.device)
        ep_hist = torch.ones(self.b, 1).to(self.device)

        for i in range(1, t+1):
            r_i_t = dcos(manager_states[t] - manager_states[t - i], goal_history[t - i]).unsqueeze(-1)
            r_i += (ep_hist * r_i_t)
            ep_hist = ep_hist * ep_indicator[t - i]

        r_i = r_i.detach().view(-1)
        return r_i / self.hist_lim


class FuNet(nn.Module):
    """

    """
    def __init__(self, input_shape, d, len_hist, epsilon, k, num_actions, num_workers, device):
        super(FuNet, self).__init__()

        self.input_shape = input_shape

        self.d = d
        self.hist_lim = len_hist
        self.epsilon = epsilon
        self.device = device

        self.k = k
        self.num_actions = num_actions
        self.b = num_workers

        self.f_percept = Percept(self.input_shape, self.d, self.device)
        self.manager = Manager(self.d, self.epsilon, self.device, self.hist_lim)
        self.worker = Worker(self.d, self.k, self.num_actions, self.device, self.hist_lim, self.b)

        self.to(device)


    def forward(self, x, goal_history, s_Mt_hist):

        z = self.f_percept(x)
        v_Mt, g_t, s_Mt = self.manager(z)


        if len(goal_history) >= self.hist_lim * 2 + 1:
            goal_history.pop(0)
            s_Mt_hist.pop(0)

        goal_history.append(g_t)

        s_Mt_hist.append(s_Mt.detach())     ## No grad

        # v_Wt, a_t, phi_g = self.worker(z, goal_history[:self.hist_lim + 1])
        v_Wt, a_t = self.worker(z, goal_history[:self.hist_lim + 1])

        return a_t, v_Mt, v_Wt, goal_history, s_Mt_hist#, phi_g


    def agent_model_init(self):

        goal_history = [init_hidden(self.b, self.d, device=self.device, grad=True) for _ in range(self.hist_lim*2 + 1)]
        s_Mt_hist = [init_hidden(self.b, self.d, device=self.device) for _ in range(self.hist_lim*2 + 1)]
        ep_indicators = [torch.ones(self.b, 1).to(self.device) for _ in range(self.hist_lim*2 + 1)]

        return goal_history, s_Mt_hist, ep_indicators

    def int_reward(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.worker.intrinsic_reward(goal_hist, s_Mt_hist, ep_indicator)

    def del_g_theta(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.manager.goal_error(s_Mt_hist, goal_hist, ep_indicator)


def loss_function(db, vMtp1, vWtp1, args):

    rt_m = vMtp1
    rt_w = vWtp1

    db.placeholder()  # Fill ret_m, ret_w with empty vals

    for i in reversed(range(args['steps'])):
        ret_m = db.ext_r_i[i] + (args['gamma_m'] * rt_m * db.ep_indicator[i])
        ret_w = db.ext_r_i[i] + (args['gamma_w'] * rt_w * db.ep_indicator[i])

        db.rt_m[i] = ret_m
        db.rt_w[i] = ret_w

    # Optionally, normalize the returns
    # db.normalize(['rt_w', 'rt_m'])

    ext_r_i, rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, \
        goal_errors = db.stack(['ext_r_i',
                                'int_r_i',
                                'v_m',
                                'v_w',
                                'rt_w',
                                'rt_m',
                                'logp',
                                'entropy',
                                'goal_error'])

    # Calculate advantages, size B x T
    advantage_w = (ret_w + args['alpha'] * rewards_intrinsic) - value_w
    advantage_m = ret_m - value_m

    loss_worker = (logps * advantage_w.detach()).mean()

    loss_manager = (goal_errors * advantage_m.detach()).mean()

    # Update the critics into the right direction
    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    entropy = entropy.mean()

    loss = - loss_worker - loss_manager + value_w_loss + value_m_loss - (args['entropy_coef'] * entropy)


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

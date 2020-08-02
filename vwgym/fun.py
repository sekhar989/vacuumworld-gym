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


class DilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0

    def forward(self, state, hidden):
        """At each time step only the corresponding part of the state is updated
        and the output is pooled across the previous c out- puts."""
        d_idx = self.dilation_idx
        hx, cx = hidden

        hx[:, d_idx], cx[:, d_idx] = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(detached_hx.shape[0], self.hidden_size, self.radius-1)
        detached_hx = detached_hx.sum(-1)

        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """Because we do not want to have gradients flowing through all
        parameters but only at the dilation index, this function creates a
        'negated' version of dilated_index, everything EXCEPT these indices."""
        masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx = masked_idx.nonzero()
        masked_idx = masked_idx - 1
        return masked_idx

    @property
    def dilation_idx(self):
        """Keep track at which dilation we currently we are."""
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx



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

        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2)

        self.max_pool_1 = nn.MaxPool2d(2)

        self.fc_1 = nn.Linear(in_features=128, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=256)
        self.percept = nn.Linear(in_features=256, out_features=256)

        self.flat = nn.modules.Flatten()

        # self.percept = nn.Sequential(
        #     nn.Conv2d(in_channels = channels, out_channels = 256, kernel_size= (3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3)
        #     nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size= (3, 3)),
        #     # Flatten(),
        #     nn.Linear(h_out * w_out * 16, d),
        #     # nn.ReLU()
        # )

    def forward(self, x):
        
        # x = x.view(-1)
        z = F.relu(self.conv_1(x))
        z = self.max_pool_1(z)

        z = F.relu(self.conv_2(z))
        z = self.max_pool_1(z)

        z = self.flat(z)

        z = F.relu(self.fc_1((z)))
        z = F.relu(self.fc_2(z))
        z = F.relu(self.percept(z))
        # z = self.(x.unsqueeze(0))

        # t = self.conv1(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)

        # # conv 2
        # t = self.conv2(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)

        # # fc1
        # t = t.reshape(-1, 12*4*4)
        # t = self.fc1(t)
        # t = F.relu(t)

        # # fc2
        # t = self.fc2(t)
        # t = F.relu(t)
        ## print('Inside Percept XXXXXXXXXXXXXXX:\n', z)
        # print(z.unsqueeze(0).shape)
        return z


class Manager(nn.Module):
    """
    f_stateM(z_t) = s_t             ## Manager Internal State Representation
    goal_gen(s_t-1, s_t) = g_t      ## Manager Goal Generation
    f_valueM(g_t) = vM_t            ## Manager Value Function

    """
    def __init__(self, d, r, epsilon, device, len_hist):
        super(Manager, self).__init__()

        self.d = d                  ## Hidden Internal Dimension
        self.d2 = d*2
        self.dd = 32
        self.epsilon = epsilon
        self.hist_lim = len_hist
        self.r  = r

        self.M_space = nn.Linear(self.d, self.d)
        self.Mrnn = DilatedLSTM(self.d, self.d, self.r)
        # self.M_relu = nn.ReLU()
        # self.M_tanh = nn.Tanh()
        # self.M_goals = nn.Sequential(
        #                                 nn.Linear(self.d, self.d),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.d, self.d)
        #                             )

        self.M_value = nn.Linear(self.d, 1)
        # self.M_value = nn.Sequential(
        #                                 nn.Linear(self.d, self.dd),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.dd, 1)
        #                             )
        self.device = device

    def forward(self, z_precept, hidden, ep_indicator):

        s_t = F.relu(self.M_space(z_precept))      ## Manager Internal State Representation
        ## print('w_shape:\t', s_tm1.shape, 'Ut_shape', s_t.shape)
        # M_int_state = torch.cat((s_tm1, s_t))          ## Concat of previous state (s_{t-1} and s_{t})
        # print(s_t.shape, [h.shape for h in hidden], ep_indicator)
        hidden = (ep_indicator * hidden[0], ep_indicator * hidden[1])
        
        goal_hat, hidden = self.Mrnn(s_t, hidden)

        # goal = self.M_relu(self.M_goals(s_t))   ## Goal generation using prev step and current state
        v_Mt = self.M_value(goal_hat)                       ## Value function

        goal = normalize(goal_hat)
        #print('Normalized Goal Hat Shape:\t', goal.shape)
        
        # goal = normalize(goal.view(1, goal.shape[0]))
        s_t = s_t.detach()

        if (self.epsilon > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        return v_Mt, goal, s_t, hidden                         ## Manager Value, Manager Goal &  Manager state (s_{t})

    def goal_error(self, s_t, g_t, ep_indicator):
        """
        error = cos(s_{t+history} - s_{t}, g_{t})

        """
        # print(len(s_t), len(g_t))

        t = self.hist_lim

        ep_indicator = torch.stack(ep_indicator[t: t + self.hist_lim - 1]).prod(dim=0)
        cosine_dist = dcos(s_t[t + self.hist_lim] - s_t[t], g_t[t])
        cosine_dist = ep_indicator * cosine_dist.unsqueeze(0)

        return cosine_dist

        # cosine_dist = dcos(s_tp1 - s_t, g_t)
        # cosine_dist = ep_indicator * cosine_dist.unsqueeze(0)

        # return cosine_dist

class Worker(nn.Module):
    """
    f_stateW(s_tm1, z_t) = W_u_t                        ## Worker Policy
    phi(goals) = w                                      ## Manager Goals sent to Worker
    torch.einsum("bk, bka -> ba", w, u_t)               ## Policy (U) X Goals (w)
    softmax(a)                                          ## Action Probabilities
    f_valueW(u_t) = vW_t                                ## Worker Value Function
    """
    def __init__(self, b, d, k, num_actions, device, len_hist):
        super(Worker, self).__init__()


        self.num_actions = num_actions
        self.b = b
        self.k = k
        self.d = d
        self.d2 = 128
        self.dd = 32
        self.knum2 = 20
        self.hist_lim = len_hist
        
        # self.f_stateW = nn.Linear(self.d, k*num_actions)
        # self.f_stateW = nn.Sequential(
        #                                 nn.Linear(self.d, self.d2),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.d2, self.dd),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.dd, k*num_actions),
        #                                 nn.ReLU()
        #                             )
        
        self.Wrnn = nn.LSTMCell(d, k * self.num_actions)

        self.phi = nn.Linear(d, k, bias=False)
        self.f_valueW = nn.Sequential(
                                        nn.Linear(k*num_actions, self.knum2),
                                        nn.ReLU(),
                                        nn.Linear(self.knum2, 1)
                                    )
        self.W_relu = nn.ReLU()
        self.device = device


    def forward(self, z, goals, hidden, ep_indicator):

        ## print(z)
        # W_u_t = self.W_relu(self.f_stateW(z))
        hidden = (ep_indicator * hidden[0], ep_indicator * hidden[1])
        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)
        
        g_t = torch.stack(goals).detach().sum(dim=0)
        # g_t = goals.detachz`()
        ## print(g_t, g_t.shape)

        w = self.phi(g_t)                                 ## phi(goals) = w ---- Manager Goals sent to Worker
        ## print(w.shape)
        # W_u_t = W_u_t.view(self.k, self.num_actions)

        ## print(W_u_t.shape)
        v_Wt = self.f_valueW(u)
        
        ## print(w)
        u = u.reshape(u.shape[0], self.k, self.num_actions)
        # print('w_shape:\t', w.shape, 'Ut_shape', u.shape)
        ## print(W_u_t)
        a_t = torch.einsum("bk, bka -> ba", w, u).softmax(dim=-1)
        ## print(a_t)
        # a_t = nn.functional.softmax(torch.mm(w, u), dim=1)
        
        return v_Wt, a_t, hidden


    def intrinsic_reward(self, goal_history, manager_states, ep_indicator):
        # print(manager_states)
        t = self.hist_lim
        r_i = torch.zeros(self.b, 1).to(self.device)
        ep_hist = torch.ones(self.b, 1).to(self.device)
        ## print(ep_indicator)
        
        for i in range(1, t+1):
            # print(i, t, t-i)  
            # print(manager_states[t] - manager_states[t - i], goal_history[t - i])
            r_i_t = dcos(manager_states[t] - manager_states[t - i], goal_history[t - i]).unsqueeze(-1)
            # print('loss', r_i_t)
            r_i += (ep_hist * r_i_t)
            ep_hist = ep_hist * ep_indicator[t - i]
            ## print('post_update', ep_hist)

        r_i = r_i.detach().view(-1)
        #print('Final intrinsic_reward:\t', r_i)
        return r_i / self.hist_lim


class FuNet(nn.Module):
    """

    """
    def __init__(self, input_shape, d, len_hist, epsilon, k, num_actions, dilation, num_workers, device):
        super(FuNet, self).__init__()
        
        self.input_shape = input_shape
        
        self.d = d
        self.hist_lim = len_hist
        self.epsilon = epsilon
        self.device = device
        
        self.k = k
        self.num_actions = num_actions

        self.r = dilation
        self.b = num_workers

        self.pre_ = Preprocessor(self.input_shape, self.device, True)
        self.f_percept = Percept(self.input_shape, self.d, self.device)
        self.manager = Manager(self.d, self.r, self.epsilon, self.device, self.hist_lim)
        self.worker = Worker(self.b, self.d, self.k, self.num_actions, self.device, self.hist_lim)

        self.hidden_m = init_hidden(self.b, self.r * self.d, device=device, grad=True)
        self.hidden_w = init_hidden(self.b, self.k * self.num_actions,
                                    device=device, grad=True)

        self.to(device)
        self.apply(weight_init)
        

    def forward(self, x, goal_history, s_Mt_hist, ep_indicator, save=True):
        
        # x = normalize_input(x)
        z = self.f_percept(x)
        ## print('XXXXXXXXXXXXXXX\t', z)
        
        # s_tM1 = s_Mt_hist[-1].view(-1)               ## Passing the immediate previous manager state

        v_Mt, g_t, s_Mt, hst_m = self.manager(z, self.hidden_m, ep_indicator)

        # print(s_Mt, '\n', s_Mt.shape)
        
        goal_history.append(g_t)
        s_Mt_hist.append(s_Mt.detach())     ## No grad

        if len(goal_history) >= self.hist_lim * 2 + 1:
            goal_history.pop(0)
            s_Mt_hist.pop(0)

        v_Wt, a_t, hst_w = self.worker(z, goal_history[:self.hist_lim + 1], self.hidden_w, ep_indicator)
        
        if save:
            # Optional, dont do this for the next_v
            self.hidden_m = hst_m
            self.hidden_w = hst_w

        return a_t, v_Mt, v_Wt, goal_history, s_Mt_hist


    def agent_model_init(self):

        template = torch.zeros(self.b, self.d)
        goal_history = [torch.zeros_like(template).to(self.device) for _ in range(2*self.hist_lim+1)]
        s_Mt_hist = [torch.zeros_like(template).to(self.device) for _ in range(2*self.hist_lim+1)]
        ep_indicators = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.hist_lim+1)]

        return goal_history, s_Mt_hist, ep_indicators

    def int_reward(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.worker.intrinsic_reward(goal_hist, s_Mt_hist, ep_indicator)

    def del_g_theta(self, goal_hist, s_Mt_hist, ep_indicator):
        return self.manager.goal_error(s_Mt_hist, goal_hist, ep_indicator)


    def repackage_hidden(self):
        def repackage_rnn(x):
            return [item.detach() for item in x]

        self.hidden_w = repackage_rnn(self.hidden_w)
        self.hidden_m = repackage_rnn(self.hidden_m)


def loss_function(db, vMtp1, vWtp1, args):

    rt_m = vMtp1
    rt_w = vWtp1

    db.placeholder()  # Fill ret_m, ret_w with empty vals
    # print('External Rewards:\n:', db.ext_r_i)
    # print('ep_indicator:\n:', db.ep_indicator)
    # print('Return Manager:\n', db.rt_m)
    # print('Return Worker:\n', db.rt_m)

    
    for i in reversed(range(args['steps'])):
        ret_m = db.ext_r_i[i] + (args['gamma_m'] * rt_m * db.ep_indicator[i])
        ret_w = db.ext_r_i[i] + (args['gamma_w'] * rt_w * db.ep_indicator[i])
        # print(i, ret_m, ret_w)
        db.rt_m[i] = ret_m
        db.rt_w[i] = ret_w
        # print()

        # print('Return Manager:\n', db.rt_m)
        # print('Return Worker:\n', db.rt_w)
    # Optionally, normalize the returns

    # print(ret_m, ret_w)
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
    # print(rewards_intrinsic)
    # print(ret_m.shape)
    # print(ret_w.shape)
    advantage_w = ret_w + (args['alpha'] * rewards_intrinsic) - value_w
    advantage_m = ret_m - value_m

    # print('Adv Manager:\n', advantage_m)
    # print('Adv Worker:\n', advantage_w)
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

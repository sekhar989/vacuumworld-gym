#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Test Script

"""
import random
import copy
import numpy as np
from vwgym import VacuumWorld, Vectorise, StepWrapper
from vwgym.fun_lite import *
from torch.distributions import Categorical

random.seed(518123)

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'


# In[2]:


model_path = '/home/archie/Documents/RHUL/Term_01/CS5940-IA/HRL/vacuumworld-gym/vwgym/vwgym_fnet_1800_ckpt.pt'

env, input_shape = make_env(grid_size=3, num_env=1, vectorize=True)
env = env[0]
env.state()


# In[3]:


checkpoint = torch.load(model_path)


# In[4]:


num_actions = env.action_space.n
f_net = FuNet(input_shape, 256,
              10,
              0.001,
              16,
              num_actions,
              1, device)


# In[5]:


f_net.load_state_dict(checkpoint['model'])


# In[6]:


f_net


# In[7]:


goal_history = checkpoint['goal']
s_Mt_hist = checkpoint['man_state']


# In[8]:


def take_action(action_probs):

#     dist = Categorical(action_probs)
    # print(action_probs)
#     action = dist.sample()
    action = action_probs.max(1)[1]
    
    return action.cpu().numpy()


# In[9]:


with torch.no_grad():

    f_net.eval()
    predictions = []
    
#     goal_history, s_Mt_hist, ep_binary = f_net.agent_model_init()
    
    x = env.reset()
    env.rw_dirts = env.dirts
    print(f'Dirts Present..:{env.rw_dirts}')
        
    x = torch.from_numpy(x).float()
    x = x.view(1, -1).to(device)
    
    step = 0
    prev_action = []
    
    for __ in range(5000):
        action_probs, v_Mt, v_Wt, goal_history, s_Mt_hist  = f_net(x, goal_history, s_Mt_hist)
        a_t = take_action(action_probs)
        x, reward, done, ep_info = env.step(a_t[0])
        print(x, reward, done)
        
        x = torch.from_numpy(x).float()
        x = x.view(1, -1).to(device)

        predictions.append(a_t[0])
        if done:
            break


# In[10]:


import pandas as pd

p = pd.DataFrame.from_dict({'actions': predictions, 'action_meanings':[env.action_meanings[i] for i in predictions]})
p['action_meanings'].value_counts().reset_index()


# In[ ]:





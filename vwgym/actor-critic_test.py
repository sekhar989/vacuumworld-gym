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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from vwgym.old_actor_critic import *
from torch.distributions import Categorical
from vwgym.init_utils import *

if torch.cuda.is_available():
    print('GPU Available:\t', True)
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

else:
    device = 'cpu'


# In[2]:


model_path = 'model/actor_9190.pt'
checkpoint = torch.load(model_path)

env, input_shape = make_env(grid_size=3, num_env=1, vectorize=True)
env = env[0]
env.state()


# In[3]:


num_actions = env.action_space.n
ac = torch.load(model_path)


# In[4]:


ac


# In[5]:


def take_action(action_probs):

#     dist = Categorical(action_probs)
    # print(action_probs)
#     action = dist.sample()
    action = action_probs.max(1)[1]
    
    return action.cpu().numpy()


# In[6]:


with torch.no_grad():
    ac.eval()
    predictions = []

    x = env.reset()
    env.rw_dirts = env.dirts
    print(f'Dirts Present..:{env.rw_dirts}')
        
    x = torch.from_numpy(x).float()
    x = x.view(1, -1).to(device)
    
    step = 0
    prev_action = []
    
    for __ in range(5000):
        action_probs = ac(x)
#         print(action_probs)
        a_t = take_action(action_probs)
        x, reward, done, ep_info = env.step(a_t[0])
        print(x, reward, done)
        
        x = torch.from_numpy(x).float()
        x = x.view(1, -1).to(device)

        predictions.append(a_t[0])
        if done:
            break


# In[7]:


import pandas as pd


# In[8]:


p = pd.DataFrame.from_dict({'actions': predictions, 'action_meanings':[env.action_meanings[i] for i in predictions]})
p['action_meanings'].value_counts().reset_index()


# In[9]:


# print(len(predictions))
# predictions.append(4)
# print(len(predictions))
# print(env_v.reset())


# In[10]:


# ac_1 = iter(predictions)
# ac_2 = iter(predictions)

# nstate = gu.episode(env_n, lambda _: next(ac_1), max_length=len(predictions))[0]
# vstate = gu.episode(env_v, lambda _: next(ac_2), max_length=len(predictions))[0]

# vis = np.concatenate([vstate[:,None,i] for i in [0,1,2]], axis=3)

# J.images(vis, scale=40, on_interact=list(zip(np.array(env.action_meanings)[predictions],nstate)));


# In[11]:


# torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))

# a, b = torch.rand(1, 8), torch.rand(8, 5)

# nn.functional.softmax(torch.mm(a, b))

# import time

# time.sleep()


# In[ ]:





# In[ ]:





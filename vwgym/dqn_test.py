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
from vwgym.dqn import *
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


model_path = 'saved_model/vwgym_dqn_450_ckpt.pt'
checkpoint = torch.load(model_path)

env, input_shape = make_env(grid_size=3, num_env=1, vectorize=True)
env = env[0]
env.state()


# In[3]:


n_actions = env.action_space.n
dq_net = dqn(input_shape, n_actions, device)
dq_net.load_state_dict(checkpoint['model'])


# In[4]:


dq_net


# In[5]:


with torch.no_grad():
    dq_net.eval()
    predictions = []

    x = env.reset()
    env.rw_dirts = env.dirts
    print(f'Dirts Present..:{env.rw_dirts}')
        
    x = torch.from_numpy(x.reshape(1, -1)).float().to(device)
    step = 0
    prev_action = []
    
    for __ in range(5000):
        action = dq_net(x)
        a_t = action.max(1)[1].view(1, 1) #take_action(action_probs, 10000, greedy=True)
        x, reward, done, ep_info = env.step(a_t.item())#take_step(a_t, env, device)
        print(x, reward, done)
        x = torch.from_numpy(x.reshape(1, -1)).float().to(device)
#         for ep_d in ep_info:
#             if ep_d['ep_rewards'] is not None:
#                 print(f"reward = {round(ep_d['ep_rewards'], 2)} \t| ep_score = {round(600/ep_d['ep_len'], 2)}")
        predictions.append(a_t.item())        
        if done:
            break


# In[6]:


import pandas as pd


# In[7]:


p = pd.DataFrame.from_dict({'actions': predictions, 'action_meanings':[env.action_meanings[i] for i in predictions]})
p['action_meanings'].value_counts().reset_index()


# In[8]:


# print(len(predictions))
# predictions.append(4)
# print(len(predictions))
# print(env_v.reset())


# In[9]:


# ac_1 = iter(predictions)
# ac_2 = iter(predictions)

# nstate = gu.episode(env_n, lambda _: next(ac_1), max_length=len(predictions))[0]
# vstate = gu.episode(env_v, lambda _: next(ac_2), max_length=len(predictions))[0]

# vis = np.concatenate([vstate[:,None,i] for i in [0,1,2]], axis=3)

# J.images(vis, scale=40, on_interact=list(zip(np.array(env.action_meanings)[predictions],nstate)));


# In[10]:


# torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))

# a, b = torch.rand(1, 8), torch.rand(8, 5)

# nn.functional.softmax(torch.mm(a, b))

# import time

# time.sleep()


# In[ ]:





# In[ ]:





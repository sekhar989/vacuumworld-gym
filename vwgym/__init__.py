#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-06-2020 17:44:31

    vacuumworld-gym is a wrapper around the Vacuum World environment that follows the OpenAI-Gym conventions for single agent environments.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import copy
from pprint import pprint

import gym
import numpy as np

import vacuumworld

from vacuumworld.vwenvironment import GridAmbient, GridEnvironment
from vacuumworld import vw, vwc, saveload
from vacuumworld.vwc import action, direction

import vacuumworld.vwaction as vwaction
import vacuumworld.vwagent
import itertools

class VacuumWorld(gym.Env):
    """ 
        An OpenAI gym wrapper around the Vacuum World Environment. See OpenAI gym API details: https://gym.openai.com/
        Defined for a single (white) agent. Fully observable.
    """
    
    def __init__(self, dimension, grid=None):
        """

        Args:
            dimension (int): grid dimension
            grid ([type], optional): a vacuum world grid to will be loaded. By default a random grid will be loaded.
        """
        if grid is None:
            grid = vw.random_grid(dimension, 1, 0, 0, 0, dimension, dimension)
        self.__initial_grid = copy.deepcopy(grid)
        self.env = GridEnvironment(GridAmbient(copy.deepcopy(self.__initial_grid), {'white':None}))
 
        self.actions = [action.move(), action.clean(), action.turn(direction.left), action.turn(direction.right), action.idle()]
        self.action_meanings = ['move', 'clean', 'turn_left', 'turn_right', 'idle']

        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = None # TODO...

    @property
    def dimension(self):
        return self.env.ambient.grid.dim

    def step(self, action):
        assert action in self.action_space
        #execute action
        action = self.actions[action]
        a = vwaction._action_factories[type(action).__name__](*action)
        if a is not None: #idle action
            a.__actor__ = next(iter(self.env.ambient.agents.keys()))
            self.env.physics.execute(self.env, [a])

        return self.state(), 0., False, None
    
    def state(self):
        return copy.deepcopy(self.env.ambient.grid)
   
    def reset(self):
        self.env = GridEnvironment(GridAmbient(copy.deepcopy(self.__initial_grid), {'white':None}))
        return self.state()

class VacuumWorldPO(VacuumWorld):
    """ 
        Partially Observable version of the VacuumWorld environment, observations are as seen by the agent (2 x 3 or 3 x 2).
    """

    def state(self):
        agent = next(iter(self.env.ambient.agents.values()))
        return self.env.processes[0].get_perception(self.env.ambient.grid, agent)

class Vectorise(gym.ObservationWrapper):
    """ 
        Vectorised version of the VacuumWorld environment. Converts the grid into a CHW [0-255] image. 

            Channel[0]: Agent Position
            Channel[1]: Agent Orientation
            Channel[2]: Dirt
    """

    dirt_table = {vwc.colour.green:128, vwc.colour.orange:255}
    orientation_table = {vwc.orientation.north:64,vwc.orientation.east:128,vwc.orientation.south:192,vwc.orientation.west:255}
    
    def __init__(self, env):
        super(Vectorise,self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, env.unwrapped.dimension, env.unwrapped.dimension), dtype=np.uint8)
            
    def observation(self, grid):

        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        c = next(iter(grid._get_agents().keys()))
        obs[0, c[1], c[0]] = 255 #mask for the current agent
        
        for c,a in grid._get_agents().items(): #all agents
            #print(c,a)
            obs[1, c[1], c[0]] = Vectorise.orientation_table[a.orientation] 
            
        for c,d in grid._get_dirts().items(): #all dirts
            #print(c,d)
            obs[2, c[1], c[0]] = Vectorise.dirt_table[d.colour]

        return obs


class StepWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(StepWrapper, self).__init__(env)
        self.ep_rewards = 0
        self.ep_len = 0
        self.time_penalty = 0
        self.ep_action_summary = []


    def _get_agent_loc(self, agent_locs):
        x, y = [list(i) for i in agent_locs]
        return [i for i in zip(x, y)]

    def _get_dirt_loc(self, dirt_locs):
        x, y = [list(i) for i in dirt_locs]
        return [i for i in zip(x, y)]

    def reward(self, obs, action):

        self.ep_len += 1
        self.time_penalty += 1

        agent_loc = self._get_agent_loc(np.where(obs[0,:, :]==255))[0]
        gr_dirt_loc = self._get_dirt_loc(np.where(obs[2,:, :]==128))
        or_dirt_loc = self._get_dirt_loc(np.where(obs[2,:, :]==255))
        # print(agent_loc, 'agent loc')
        # print(gr_dirt_loc, 'green dirt loc')
        # print(or_dirt_loc, 'orange dirt loc')
        dirt_locs = list(itertools.chain.from_iterable([or_dirt_loc, gr_dirt_loc]))

        action = self.action_meanings[action]
        done = False
        dead = False
        blank_hit = 0

        if self.ep_len >= 5000:
            done = True
            dead = True
            print('Time Up...')
        elif len(dirt_locs) == 0:
            print('Grid Cleaned !! :)')
            done = True

        
        # if agent_loc in dirt_locs and action == 'clean':
        # # if action == 'clean':
        #     reward = 10 #- len(dirt_locs) * (self.ep_rewards/self.ep_len)
        #     self.ep_rewards += reward
        #     # self.time_penalty = 0
        #     return reward, done
        # else:
        #     reward = -1
        #     self.ep_rewards += reward
        #     return reward, done

            # if self.ep_len < 500:
            #     reward = -1 - self.time_penalty/10 #* (2**(len(dirt_locs))) #- len(dirt_locs) * (self.ep_rewards/self.ep_len)
            #     self.ep_rewards += reward
            #     return reward, done
            # else:
            #     print('dead...')
            #     done = True
            #     reward = -100
            #     self.ep_rewards += reward
            #     return reward, done

        #####################################
        # print(action, agent_loc, dirt_locs)
        #####################################

        if dead:
            reward = -20
            self.ep_rewards += reward
            return reward, done

        elif agent_loc in dirt_locs and action == 'clean':
            reward = 20 #- round(len(dirt_locs) * (self.ep_rewards/self.ep_len), 3)
            blank_hit = 0
            if len(dirt_locs) == 0:
                print('Grid Cleaned !! :)')
                done = True
                reward += 5
            self.ep_rewards += reward 
            return reward, done
            
        elif agent_loc in dirt_locs and (action != 'clean' or action == 'idle'):
            reward = -5 #- round(len(dirt_locs) * (self.ep_rewards/self.ep_len), 3)
            self.ep_rewards += reward 
            return reward, done

        elif agent_loc not in dirt_locs and (action == 'clean' or action == 'idle'):
            reward = -2 #- round(len(dirt_locs) * (self.ep_rewards/self.ep_len), 3)
            self.ep_rewards += reward 
            return reward, done

        elif len(dirt_locs) > 0 and action == 'idle':
            reward = -5 #- round(len(dirt_locs) * (self.ep_rewards/self.ep_len), 3)
            self.ep_rewards += reward 
            return reward, done

        elif len(dirt_locs) > 0 and (self.ep_action_summary[-8:].count('turn_right') == 8 or \
                self.ep_action_summary[-8:].count('turn_left') == 8) or \
                self.ep_action_summary[-20:].count('move') == 20:
            # reward = -1 * len(dirt_locs) * int(np.sqrt(self.ep_len))
            reward = -2 #* int(np.sqrt(self.ep_len)/2)
            # print('stuck penalty', reward)
            self.ep_rewards += reward
            return reward, done

        else:
            # blank_hit += 1

            # if blank_hit == 10:
            #     reward = -5
            #     self.ep_rewards += reward
            #     blank_hit = 0
            # else:
            reward = 0
            self.ep_rewards += reward
            return reward, done
        
    def step(self, prev_state, action):
        state, reward, done, _x = self.env.step(action)
        reward, done = self.reward(prev_state, action)
        self.ep_action_summary.append(self.env.action_meanings[action])
        if done:
            ep_info = {'ep_rewards': self.ep_rewards, 'ep_len': self.ep_len}
            for a in self.env.action_meanings:
                ep_info[a] = self.ep_action_summary.count(a)
            print(ep_info)
            self.ep_rewards = 0
            self.ep_len = 0
            self.ep_action_summary = []
        else:
            ep_info = {'ep_rewards': None, 'ep_len': None}
            for a in self.env.action_meanings:
                ep_info[a] = None

        return state, reward, done, ep_info


if __name__ == "__main__":
    env = Vectorise(VacuumWorld(3))
    state = env.reset()
    print(state)
    for i in range(10):
        state, reward, done, *_ = env.step(0)
        print(state)
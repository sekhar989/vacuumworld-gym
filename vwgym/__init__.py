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

class VacuumWorld(gym.Env):
    
    def __init__(self, dimension, grid=None):
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

    def state(self):
        agent = next(iter(self.env.ambient.agents.values()))
        return self.env.processes[0].get_perception(self.env.ambient.grid, agent)

class Vectorise(gym.ObservationWrapper):

    dirt_table = {vwc.colour.green:128, vwc.colour.orange:255}
    orientation_table = {vwc.orientation.north:64,vwc.orientation.east:128,vwc.orientation.south:192,vwc.orientation.west:255}
    
    def __init__(self, env):
        super(Vectorise,self).__init__(env)

        print(env.unwrapped.dimension)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, env.unwrapped.dimension, env.unwrapped.dimension), dtype=np.uint8)
            
    def observation(self, grid):

        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        c = next(iter(grid._get_agents().keys()))
        obs[0, c[0], c[1]] = 255 #mask for the current agent
        
        for c,a in grid._get_agents().items(): #all agents
            #print(c,a)
            obs[1, c[0],c[1]] = Vectorise.orientation_table[a.orientation] 
            
        for c,d in grid._get_dirts().items(): #all dirts
            #print(c,d)
            obs[2, c[0], c[1]] = Vectorise.dirt_table[d.colour]

        return obs


if __name__ == "__main__":
    env = Vectorise(VacuumWorld(3))
    state = env.reset()
    print(state)
    for i in range(10):
        state, reward, done, *_ = env.step(0)
        print(state)
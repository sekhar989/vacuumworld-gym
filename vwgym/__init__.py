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
import random

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
            grid = vw.random_grid(dimension, 1, 0, 0, 0, 2, 1)
        self.__initial_grid = copy.deepcopy(grid)
        self.env = GridEnvironment(GridAmbient(copy.deepcopy(self.__initial_grid), {'white':None}))
        ## Remove Idle action
        self.actions = [action.move(), action.clean(), action.turn(direction.left), action.turn(direction.right)]#, action.idle()]
        self.action_meanings = ['move', 'clean', 'turn_left', 'turn_right']#, 'idle']

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

    dirt_table = {vwc.colour.green:255, vwc.colour.orange:255}
    # orientation_table = {vwc.orientation.north:64,
    #                      vwc.orientation.east:128,
    #                      vwc.orientation.south:192,
    #                      vwc.orientation.west:255}
    orientation_table = {vwc.orientation.north: 0,
                          vwc.orientation.east: 1,
                          vwc.orientation.south: 2,
                          vwc.orientation.west: 3}

    def __init__(self, env):
        super(Vectorise,self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, env.unwrapped.dimension, env.unwrapped.dimension), dtype=np.uint8)
        self.dirts = None

    def observation(self, grid):

        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        # obs_01 = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        ao_oh = np.zeros(4)
        self.dirts = 0

        c = next(iter(grid._get_agents().keys()))
        # for i in c:
        #     agent_loc.append(i)

        obs[0, c[1], c[0]] = 255 #mask for the current agent

        for c,a in grid._get_agents().items(): #all agents
            ao_oh[Vectorise.orientation_table[a.orientation]] = 255
        #     #print(c,a)
            # obs[1, c[1], c[0]] = Vectorise.orientation_table[a.orientation]
            # agent_loc.append(Vectorise.orientation_table[a.orientation])

        for c,d in grid._get_dirts().items(): #all dirts
            #print(c,d)
            self.dirts += 1
            obs[1, c[1], c[0]] = Vectorise.dirt_table[d.colour]
            # obs_01[0, c[1], c[0]] = Vectorise.dirt_table[d.colour]

        obs = np.append(ao_oh, obs)
        # obs.reshape(1, -1)/255.0
        # obs_01 = np.concatenate((np.array(agent_loc, dtype=self.observation_space.dtype), obs_01.reshape(-1)))

        return obs/255.0


class StepWrapper(gym.Wrapper):

    def __init__(self, env):
        super(StepWrapper, self).__init__(env)
        self.ep_rewards = 0
        self.ep_len = 0
        self.time_penalty = 0
        self.ep_action_summary = []
        self.rw_dirts = 0


    def _get_agent_loc(self):
        z = self.env.state()
        return [(c[0], c[1]) for c, a in z._get_agents().items()][0]

    def _get_dirt_loc(self):
        z = self.env.state()
        return [(c[0], c[1]) for c, d in z._get_dirts().items()]

    def reward(self, action):

        self.time_penalty += 1
        # print(self.time_penalty)

        # agent = self._get_agent_loc()
        # dirts = self._get_dirt_loc()

        if self.env.dirts < self.rw_dirts:
            # print('Dirt Cleaned...', self.env.dirts, self.rw_dirts)
            self.rw_dirts = self.env.dirts
            # self.time_penalty = int(0.25*self.time_penalty)
            # if self.env.dirts == 0:
            #     return 500, False
            return 100, False

        elif self.env.dirts == 0:
            print('\nGrid Cleaned !!\n') ## End the episode after all are cleaned i.e. cycle after
            self.time_penalty = 0
            return 0, True

        elif self.time_penalty > 50:
            # print(self.time_penalty)
            print(f'\nTime Up !! :(, Dirts Cleaned:  {3 - self.env.dirts}\n')
            self.time_penalty = 0
            return 0, True

        # elif agent not in dirts and action == 1:
        #     # print('Agent not in dirt cleaning..')
        #     return -5, False

        # elif agent in dirts and action != 1:
        #     # print('Agent in dirt not cleaning..')
        #     return -10, False

        else:
            return -1, False



    def step(self, action):
        state, reward, done, _x = self.env.step(action)
        reward, done = self.reward(action)
        # print(reward, done, self.env.action_meanings[action])
        self.ep_rewards += reward
        self.ep_len += 1

        self.ep_action_summary.append(self.env.action_meanings[action])
        if done:
            ep_info = {'ep_rewards': self.ep_rewards, 'ep_len': self.ep_len}
            for a in self.env.action_meanings:
                ep_info[a] = self.ep_action_summary.count(a)
            print(ep_info)
            print('Episode Rewards:\t', self.ep_rewards, '\n', '-'*40)
            self.ep_rewards = 0
            self.ep_len = 0
            self.ep_action_summary = []
        else:
            ep_info = {'ep_rewards': None, 'ep_len': None}
            for a in self.env.action_meanings:
                ep_info[a] = None

        return state, reward, done, ep_info


if __name__ == "__main__":
    random.seed(123)
    env = StepWrapper(Vectorise(VacuumWorld(5)))
    state = env.reset()
    env.rw_dirts = env.dirts
    print(env.action_space.n)

    for i in range(500):
        state, reward, done, ep_info = env.step(np.random.choice(range(4)))
        # print(state, reward, done, ep_info)
        if done: break

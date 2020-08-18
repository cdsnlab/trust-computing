#* general
import numpy as np
import pandas as pd
import os

#* for RL 
import gym
from gym import spaces
from gym.utils import seeding


N_DISCRETE_ACTIONS = 2

class trustEnv(gym.Env):
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 
        
    """
    def __init__(self, data):
        super(trustEnv, self).__init__()
        self.df = data
        #* creates two actions: increase t_threshold OR decrease t_threshold
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        #* creates all possible spaces 
        self.get_space_matrix(self.df)
        

    def step(self, a):
        #* inputs an action to current state & gets a reward to this action

        return observed_state, reward

    def reset(self):
        #* theres no need for our program to reset.. i think
        print("reseting")

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("rendering")
        
    def get_space_matrix(self, df):
        #* returns all available spaces
        colname = list(df.columns)
        print(colname)
        #! requires min max of each category
        mmax = list(df.max(axis=0))
        print(mmax)
        mmin = list(df.min(axis=0))
        print(mmin)
        #self.observation_space = spaces.Dict({""})

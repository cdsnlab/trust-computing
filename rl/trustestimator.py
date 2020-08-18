
#* general
import numpy as np
import pandas as pd
import os
#* for NN models
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
#* for RL 
import gym
from gym import spaces
from gym.utils import seedings
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory



class trustestimator(gym.Env):
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 

        # TODO 3) define rewards
        # TODO 4) define reset => after N 
        
    """
    def __init__(self):
        self.observed_state = None
        self.action_space = self.generate_as(4)


    def generate_as(self, integer_len):
        #* generate and return all possible action space
        assert integer_len > 1
        return list(x for i in range(integer_len))

    def step(self, a):
        #* inputs an action to current state & gets a reward to this action

        return observed_state, reward

    def make_action(self, self.action_space, policy):
        #* choose an action from the action_space
        if policy =="r":
            # pick a random choice
        elif policy == "eg":
            # pick based on q value
        return action

    

    def reset(self):
        print("reseting")

#* read dataset
os.chdir('/home/spencer/data_0_.txt')


te = trustestimator(gym.Env):
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
while(True):
    # TODO read context-tuple

    ra = te.make_action()
    te.step(ra)


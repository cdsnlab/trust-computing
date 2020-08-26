#* general
import numpy as np
import pandas as pd
import os

#* for RL 
import gym
from gym import spaces
from gym.utils import seeding
from collections import defaultdict
#* draw now
import matplotlib.pyplot as plt



N_DISCRETE_ACTIONS = 2

class trustEnv(gym.Env):
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 
        
    """
    def __init__(self):
        super(trustEnv, self).__init__()
        self.load_dataset('../sampledata/additional_features_with_predictions.csv') # load dataset
        #* creates 3 actions: increase t_threshold OR decrease t_threshold
        self.action_space = ["u", "d", "s"] #* 0, 1, 2
        self.n_actions = len(self.action_space)
        self.cases = defaultdict(lambda:[1, 1, 1]) #* total number of cases, correct cases, wrong cases
        self.state = 80
        self.next_car_index = 1

    def load_dataset(self, filename):
        #* read dataset
        print("[INFO] Reading file...")
        #data = pd.read_csv('../sampledata/data_6_.txt', sep='\t', header=0)
        self.data = pd.read_csv(filename, header=0)
        print("[INFO] File loaded")
       

    def step(self, action):
        #* inputs an action to current state & gets a reward to this action
        DELAY = 10
        delta = 5
        #! 당연하게도 0에 수렴하겠지... 왜냐하면 올려도 threshold보다 크면 그만이고, 줄여도 threshold보다 크면 그만이니깐...
        #! 어케 따라다니게 만들지?

        e_trust_val = int(self.data['I_trust_val'][self.next_car_index]*100)
        self.cases["p"][0] +=1
        #* adjust tthreshold
        if action == 0: #up
            if self.state < 100:
                self.state+=delta
        elif action == 1: #down
            if self.state > 0:
                self.state-=delta
        else:
            self.state-=0
        print("[INFO] action: {}, tthreshold: {}, estimated trust value: {}, ".format(action, self.state, e_trust_val))

        #* check if trust value is higher or lower than tthreshold.

        if e_trust_val > self.state: 
            self.cases["p"][1] +=1
            result=1
        else:
            self.cases["p"][2] +=1
            result=0
        #! 이전 trust값과 이전 feedback으로 계산해야지 밥팅아 
        print(self.next_car_index, DELAY)
        if self.next_car_index > DELAY:
            self.cases['g'][0]+=1
            feedback = int(self.data['actual_status'][self.next_car_index-DELAY])
            if result == feedback:
                reward = 1
                self.cases["g"][1]+=1
            else: #* 구라친 경우
                print("[DEBUG] FFFFFFFFFFFFFFFFFFFFFFFF")
                reward = -100
                self.cases["g"][2]+=1
        else:
            reward = 0

        
        print(self.cases['p'])
        print("Perceived Accuracy: {}".format(self.cases["p"][1]/self.cases["p"][0]))
        print(self.cases['g'])
        print("GT Accuracy: {}".format(self.cases["g"][1]/self.cases["g"][0]))

        self.next_car_index+=1

        if self.next_car_index == 99999:
            self.drawgraph()
        return reward, self.state
    
    def drawgraph(self):
        

    def reset(self):
        #* theres no need for our program to reset.. i think
        print("reseting")

    def render(self, mode='human', close=False):
        # Render the environment to the screen. Don't think we need this though
        print("rendering")
        
    # def get_space_matrix(self, df):
    #     #* returns all available spaces
    #     colname = list(df.columns)
    #     print(colname)
    #     #! requires min max of each category 
    #     #! typically from 0 ~ 100 should be enough.
    #     mmax = list(df.max(axis=0))
    #     print(mmax)
    #     mmin = list(df.min(axis=0))
    #     print(mmin)
    #     #self.observation_space = spaces.Dict({""})
        

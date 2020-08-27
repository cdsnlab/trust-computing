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
        self.state = 80.0
        self.next_car_index = 1
        self.result_values = defaultdict(lambda: [1, 1, 1, 1])

    def load_dataset(self, filename):
        #* read dataset
        print("[INFO] Reading file...")
        #data = pd.read_csv('../sampledata/data_6_.txt', sep='\t', header=0)
        self.data = pd.read_csv(filename, header=0)
        print("[INFO] File loaded")
    
    def get_car(self):
        #print(self.next_car_index)
        e_trust_val = int(self.data['I_trust_val'][self.next_car_index]*100)
        self.result_values[self.next_car_index][0] = e_trust_val
        self.result_values[self.next_car_index][1] = self.state # dynamic threshold 값.
        self.result_values[self.next_car_index][2] = self.cases["p"][1]/self.cases["p"][0]
        self.cases['p'][0]+=1
        if e_trust_val > self.state:
            self.cases["p"][1]+=1
        else:
            self.cases["p"][2]+=1
        self.next_car_index+=1
        return e_trust_val

    def step(self, action):
        #* inputs an action to current state & gets a reward to this action

        #! 문제점 1: 이전 dynamic threshold에 해당하는 action이 잘 못 됨을 적용시켜야되는데, 현재는 지금 state와 action에 q값 조정함. 
        DELAY = 10
        delta = 5

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
        #print(self.next_car_index, DELAY)
        if self.next_car_index > DELAY:
            self.cases['g'][0]+=1
            feedback = int(self.data['actual_status'][self.next_car_index-DELAY])
            if result == feedback:
                reward = 1
                self.cases["g"][1]+=1
            else: #* 구라친 경우
                #print("[DEBUG] FFFFFFFFFFFFFFFFFFFFFFFF")
                reward = -1
                self.cases["g"][2]+=1
        else:
            reward = 0

        # print(self.cases['p'])
        # print("Perceived Accuracy: {}".format(self.cases["p"][1]/self.cases["p"][0]))
        # print(self.cases['g'])
        # print("GT Accuracy: {}".format(self.cases["g"][1]/self.cases["g"][0]))

        self.write_to_variable(self.next_car_index, e_trust_val)        

        if self.next_car_index == 99999:
            self.drawgraph()

        #self.next_car_index+=1

        return reward, self.state
    

    def step2(self, action, car_id, perceived_btrust):
        #* inputs an action to current state & gets a reward to this action

        delta = 1
        #! should we limit the min max of that threshold?
        if action == 0: #up
            if self.state < 100:
                self.state+=delta
        elif action == 1: #down
            if self.state > 0:
                self.state-=delta
        else:
            self.state-=0
        
        feedback = int(self.data['actual_status'][car_id])
        #print(car_id, self.next_car_index)
        self.cases['g'][0]+=1
        if perceived_btrust == feedback:
            self.cases["g"][1]+=1
            reward =1
        else:
            self.cases["g"][2]+=1
            reward =-1
        
        
        # #* check if trust value is higher or lower than tthreshold.
        # if e_trust_val > self.state: 
        #     self.cases["p"][1] +=1
        #     result=1
        # else:
        #     self.cases["p"][2] +=1
        #     result=0
        
        # print(self.cases['p'])
        # print("Perceived Accuracy: {}".format(self.cases["p"][1]/self.cases["p"][0]))
        # print(self.cases['g'])
        # print("GT Accuracy: {}".format(self.cases["g"][1]/self.cases["g"][0]))

        self.write_to_variable(self.next_car_index)        

        # if self.next_car_index == 99999:
        #     self.drawgraph()

        #self.next_car_index+=1

        return reward, self.state


    
    def write_to_variable(self, nci):
        
        #self.result_values[nci][0] = self.state
        self.result_values[nci][3] = self.cases["g"][1]/self.cases["g"][0]

        #print("len:", len(self.result_values))
        #for i in range (2):
        #    print(self.result_values[nci][i])


    def drawgraph(self): #! Y축1: accuracy, Y축2: dyanmic threshold 변화, 실제 trust값.
        xvalues = range(0, len(self.result_values))
        ytrust = []
        ythr = []
        yacc= []
        yaccgt= []

        for i in range (len(self.result_values)):
            ytrust.append(self.result_values[i][0])
            ythr.append(self.result_values[i][1])
            yacc.append(self.result_values[i][2]*100.0)
            yaccgt.append(self.result_values[i][3]*100.0)
        

        plt.plot(xvalues, ytrust, 'r', label="estimated trust value") # 
        plt.plot(xvalues, ythr, 'b', label="dynamic threshold")
        plt.plot(xvalues, yacc, 'g', label="perceived accuracy")
        plt.plot(xvalues, yaccgt, 'y', label="actual accuracy")
        plt.legend()
        plt.grid()
        plt.show()

    def get_accuracy(self):
        print(self.result_values[len(self.result_values)-1][2])


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
        

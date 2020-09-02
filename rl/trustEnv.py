#* general
import numpy as np
import pandas as pd
import os
import json

#* for RL 
import gym
from gym import spaces
from gym.utils import seeding
from collections import defaultdict
#* draw now
import matplotlib.pyplot as plt
from pymongo import MongoClient

# def load_dataset(filename):
#         #* read dataset
print("[INFO] Reading file...")
#data = pd.read_csv('../sampledata/data_6_.txt', sep='\t', header=0)
data = pd.read_csv('../sampledata/additional_features_with_predictions.csv', header=0)
print("[INFO] File loaded")

class trustEnv:
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 
        
    """
    def __init__(self, thrvalue, deltavalue, rewvalue):
        super(trustEnv, self).__init__()
        #load_dataset('../sampledata/additional_features_with_predictions.csv') # load dataset
        self.connect()
        #* creates 3 actions: increase t_threshold OR decrease t_threshold
        self.action_space = ["u", "d", "s"] #* 0, 1, 2
        self.n_actions = len(self.action_space)
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, TN, FN, FP
        self.delta = deltavalue
        self.state = thrvalue
        self.reward_value = rewvalue
        self.cur_decision ={}
        self.next_car_index = 1
        self.result_values = defaultdict(lambda: [0, 0, 0, 0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy

        self.client = None

    def connect(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['trustdb']
        self.coll = self.db['q_learning']


    
    
    def get_car(self):
        e_trust_val = int(data['I_trust_val'][self.next_car_index]*100)

        if e_trust_val > self.state:
            self.cur_decision[self.next_car_index]=1
        else:
            self.cur_decision[self.next_car_index]=0
        
        self.result_values[self.next_car_index][0] = e_trust_val
        self.result_values[self.next_car_index][1] = self.state # dynamic threshold 값.

        return e_trust_val

    def step(self, action):
        #* inputs an action to current state & gets a reward to this action

        #! 문제점 1: 이전 dynamic threshold에 해당하는 action이 잘 못 됨을 적용시켜야되는데, 현재는 지금 state와 action에 q값 조정함. 
        DELAY = 10
        delta = 5

        e_trust_val = int(data['I_trust_val'][self.next_car_index]*100)
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
            feedback = int(data['actual_status'][self.next_car_index-DELAY])
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


        if self.next_car_index == 99999:
            self.drawgraph()

        #self.next_car_index+=1

        return reward, self.state
    

    def step2(self, action, car_id, perceived_btrust): 
        #* inputs an action to current state & gets a reward to this action
        #! should we limit the min max of that threshold? 
        if action == 0: #up
            if self.state < 100:
                self.state+=self.delta
        elif action == 1: #down
            if self.state > 0:
                self.state-=self.delta
        else:
            self.state-=0
        
        ###* Reward주는 방법 1) TP, TN, FN, FP 구별해서 주기
        if self.cur_decision[car_id] == 1 and data['actual_status'][car_id] == 1: 
            reward = self.reward_value
        elif self.cur_decision[car_id]==1 and data['actual_status'][car_id] == 0:
            reward = -(self.reward_value)
        elif self.cur_decision[car_id]==0 and data['actual_status'][car_id] == 1:
            reward = -(self.reward_value)
        else:
            reward = self.reward_value

        ###* Reward주는 방법 2) 이전보다 accuracy가 늘어나면 주기
        # if self.result_values[car_id][3] > self.result_values[car_id-1][3]: #* 이전보다 accuracy가 높아지면 +, 아니면 -
        #     reward = self.reward_value
        # else:
        #     reward = -(self.reward_value)
        # print("id {}, prev {}, cur {}, rew {}".format(car_id, self.result_values[car_id-1][3], self.result_values[car_id][3], reward))

        return reward, self.state


    def drawgraph(self): 
        xvalues = range(0, len(self.result_values))
        ytrust = []
        ythr = []
        yacc= []
        yaccgt= []

        for i in range (len(self.result_values)):
            ytrust.append(self.result_values[i][0])
            ythr.append(self.result_values[i][1])
            # yacc.append(self.result_values[i][2]*100.0)
            yaccgt.append(self.result_values[i][3]*100.0)
        
        plt.plot(xvalues, ytrust, 'r', label="estimated trust value") # 
        plt.plot(xvalues, ythr, 'b', label="DTT")
        #plt.plot(xvalues, yacc, 'g', label="estimated accuracy")
        plt.plot(xvalues, yaccgt, 'y', label="actual accuracy")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(name)

    def gt_accuracy(self, nci): #* gets gt accuracy regardless of time
        if self.cur_decision[nci] == 1 and data['actual_status'][nci] == 1: 
            self.cases["gt"][0]+=1
        elif self.cur_decision[nci]==1 and data['actual_status'][nci] == 0:
            self.cases["gt"][1]+=1
        elif self.cur_decision[nci]==0 and data['actual_status'][nci] == 1:
            self.cases["gt"][2]+=1
        else:
            self.cases["gt"][3]+=1

        #* Threshold 가 낮으면 그냥 trust하겠다는 의민데, 그렇게하면 
        self.result_values[nci][3] = (self.cases["gt"][0] + self.cases["gt"][3])/(self.cases["gt"][0] + self.cases["gt"][1] + self.cases["gt"][2] + self.cases["gt"][3]) #* TT + FF / all cases


    def savetomongo(self,name):
        ytrust, ythr, yacc, yaccgt = [], [], [], []
        for i in range (len(self.result_values)):
            ytrust.append(self.result_values[i][0])
            ythr.append(self.result_values[i][1])
            # yacc.append(self.result_values[i][2]*100.0)
            yaccgt.append(self.result_values[i][3]*100.0)
        row = {"id": str(name), "v_d": name.v_d, "v_lr": name.v_lr, "v_df": name.v_df, "v_eps": name.v_eps, "v_fd": name.v_fd, "v_s": name.v_s, "v_i": name.v_i, "yvalue": yaccgt}
        #print(row)
        self.coll.insert_one(row)

    def savetojson(self, name, filename):
        ytrust, ythr, yacc, yaccgt = [], [], [], []
        for i in range (len(self.result_values)):
            ytrust.append(self.result_values[i][0])
            ythr.append(self.result_values[i][1])
            # yacc.append(self.result_values[i][2]*100.0)
            yaccgt.append(self.result_values[i][3]*100.0)
        #print(yaccgt)
        if os.path.exists(filename): # if it exists
            if os.stat(filename).st_size!=0: # if it is not empty
                with open(filename) as json_file:
                    json_decoded = json.load(json_file)
                    #print(json_decoded)

                json_decoded[str(name)] = yaccgt
                with open(filename, 'w') as json_file:
                    json.dump(json_decoded, json_file)
            else: # if content is empty
                json_decoded = {}
                json_decoded[str(name)] = yaccgt
                with open(filename, 'w') as json_file:
                    json.dump(json_decoded, json_file)
        else:
            os.mknod(filename)
            json_decoded = {}
            json_decoded[str(name)] = yaccgt
            with open(filename, 'w') as json_file:
                json.dump(json_decoded, json_file)
        
        self.reset()

    def reset(self):
        
        print("[INFO] reseting...")
        self.result_values = defaultdict(lambda: [0, 0, 0, 0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy

    def render(self, mode='human', close=False):
        # Render the environment to the screen. Don't think we need this though
        print("rendering")
        
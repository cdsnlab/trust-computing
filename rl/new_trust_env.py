#* general
import numpy as np
import pandas as pd
import os
import json

#* for RL 
#import gym
#from gym import spaces
#from gym.utils import seeding
from collections import defaultdict
#* draw now
import matplotlib.pyplot as plt
from pymongo import MongoClient

print("[INFO] Reading file...")
#data = pd.read_csv('../sampledata/data_6_.txt', sep='\t', header=0)
data = pd.read_csv('../sampledata/local_data_ce_db.csv', header=0)
print("[INFO] File loaded")

class trustEnv:
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 
        
    """
    def __init__(self, thrvalue, deltavalue, bdelta, rewvalue, beta):
        super(trustEnv, self).__init__()
        
        #* creates 3 actions: increase t_threshold OR decrease t_threshold
        #* general parameters
        # self.action_space = ["uu", "ud", "us", "du", "dd", "ds", "su", "sd", "ss"] #* {state, beta} x {up, down, stay}
        self.action_space = ["u", "d", "s"]
        self.n_actions = len(self.action_space)
        self.delta = deltavalue
        self.bdelta = bdelta
        self.dtt = thrvalue
        self.beta = beta #* added beta!
        self.state = None
        self.reward_value = rewvalue
        #* epoch parameters
        # self.average_iteration_accgt =[] # keeps track of accuracy for same configuration during AVERAGEITERATION iteration to make an average. (근데 average해도 되는거야?)
        # self.cumulative_iteration_reward =[]
        
        self.average_gt=defaultdict(list)
        self.average_state= defaultdict(list)
        self.epoch_accuracy = defaultdict(list)
        #* step parameters
        self.cur_decision ={}
        self.next_car_index = 1
        # self.cumulative_reward =0 
        self.cumulative_state=0
        self.cumulative_gt =0
        self.result_values = defaultdict(lambda: [0, 0, 0, 0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN

        self.client = None
        #self.connect()

    def connect(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['trustdb']
        # self.acccoll = self.db['accuracy'] #* contains accuracy
        # self.rewcoll = self.db['rewards'] #* contains rewards 
        # self.accrewcollection = self.db['a5acc_3actions'] #20201019 meeting data
        self.accrewcollection = self.db['new_tv_data']

    
    
    def get_car(self):
        
        tv_id = int(data['indirect_tv'][self.next_car_index]*100)
        tv_d = int(data['direct_tv'][self.next_car_index]*100)
        # print(tv_d)
        tv = (self.beta * tv_id + (1-self.beta) * tv_d) #! if beta close to 1 -> indirect evidence 
        #! i don't think that beta should lean towards 0 at the end of the epoch
        # print("ID: {}, tid {}, td {}, tv {}, beta {}, dtt {}".format(self.next_car_index, tv_id, tv_d, tv, self.beta, self.dtt))
        # print("NCI {}".format(self.next_car_index))
        if tv > self.dtt:
            self.cur_decision[self.next_car_index]=0 #! if benign, it should be over dtt
        else: 
            self.cur_decision[self.next_car_index]=1  #! if malicious, it should be under dtt
        
        self.result_values[self.next_car_index][0] = tv
        self.result_values[self.next_car_index][1] = self.dtt # dynamic threshold 값.
        self.cumulative_gt+=tv
        # print(self.result_values)
        #return self.next_car_index
    
    # def step2(self, action, car_id): 
    #     #* inputs an action to current state & gets a reward to this action
    #     #! we should also consider changing betas values too. 
    #     if action == 0: #uu
    #         if self.dtt + self.delta < 100:
    #             self.dtt+=self.delta
    #         if self.beta + self.bdelta < 1:
    #             self.beta += self.bdelta
    #     elif action == 1: #ud
    #         if self.dtt + self.delta < 100:
    #             self.dtt+=self.delta
    #         if self.beta - self.bdelta > 0:
    #             self.beta += -(self.bdelta)
    #     elif action == 2: # us
    #         if self.dtt + self.delta < 100:
    #             self.dtt+=self.delta
    #     elif action == 3: # du
    #         if self.dtt - (self.delta) > 0:
    #             self.dtt-=self.delta
    #         if self.beta + self.bdelta < 1:
    #             self.beta += self.bdelta
    #     elif action == 4: # dd
    #         if self.dtt - (self.delta) > 0:
    #             self.dtt-=self.delta
    #         if self.beta - self.bdelta > 0:
    #             self.beta += -(self.bdelta)
    #     elif action == 5: # ds
    #         if self.dtt - (self.delta) > 0:
    #             self.dtt-=self.delta
    #     elif action == 6: # su
    #         if self.beta + self.bdelta < 1:
    #             self.beta += self.bdelta
    #     elif action == 7: # sd
    #         if self.beta - self.bdelta > 0:
    #             self.beta += -(self.bdelta)
    #     else: # ss
    #         self.dtt-=0
    #         self.beta-=0
        
    #     ###* Reward주는 방법 1) TP, TN, FN, FP 구별해서 주기
    #     # print(self.cur_decision[car_id], data['status'][car_id])
    #     if self.cur_decision[car_id] == 1 and data['status'][car_id] == 1: 
    #         reward = self.reward_value
    #     elif self.cur_decision[car_id]==1 and data['status'][car_id] == 0:
    #         reward = -(self.reward_value) #! penalty
    #     elif self.cur_decision[car_id]==0 and data['status'][car_id] == 1:
    #         reward = -(self.reward_value)*2 #! less penalty for upper
    #     else:
    #         reward = self.reward_value
    #     # self.cumulative_reward += reward
    #     self.cumulative_state += self.dtt
        
    #     textbeta = format(self.beta, ".2f")
    #     self.state = (textbeta, self.dtt)
    #     #! it should be self.beta * self.dtt
    #     return reward, self.state


    def step3(self, action, car_id): 
        #* inputs an action to current state & gets a reward to this action
        #! we should also consider changing betas values too. 
        if action == 0: #u
            if self.dtt + self.delta < 100:
                self.dtt+=self.delta

        elif action == 1: # du
            if self.dtt - (self.delta) > 0:
                self.dtt-=self.delta

        else: # ss
            self.dtt-=0
        
        ###* Reward주는 방법 1) TP, TN, FN, FP 구별해서 주기
        # print(self.cur_decision[car_id], data['status'][car_id])
        if self.cur_decision[car_id] == 1 and data['behavior'][car_id] == 1: 
            reward = self.reward_value
        elif self.cur_decision[car_id]==1 and data['behavior'][car_id] == 0:
            reward = -(self.reward_value) #! penalty
        elif self.cur_decision[car_id]==0 and data['behavior'][car_id] == 1:
            reward = -(self.reward_value)*2 #! less penalty for upper
        else:
            reward = self.reward_value
        # self.cumulative_reward += reward
        self.cumulative_state += self.dtt
        
        # textbeta = format(self.beta, ".2f")
        # self.state = (textbeta, self.dtt)
        #! it should be self.beta * self.dtt
        return reward, self.dtt #self.state



    def gt_accuracy(self, nci): #* gets gt accuracy regardless of time
        if self.cur_decision[nci] == 1 and data['behavior'][nci] == 1: #TP
            self.cases["gt"][0]+=1
        elif self.cur_decision[nci]==1 and data['behavior'][nci] == 0: #!FP
            self.cases["gt"][1]+=1
        elif self.cur_decision[nci]==0 and data['behavior'][nci] == 1: #FN 
            self.cases["gt"][2]+=1
        else:  #!TN
            self.cases["gt"][3]+=1

        #* 매번 호출 할 때마다 TT + FF / all case 업데이트.
        self.result_values[nci][3] = (self.cases["gt"][0] + self.cases["gt"][3])/(self.cases["gt"][0] + self.cases["gt"][1] + self.cases["gt"][2] + self.cases["gt"][3]) #* TT + FF / all cases
        # print(self.cases)
        # print(self.result_values[nci][3])


    def append_accuracy(self, run_counts, iteration):
        self.epoch_accuracy[run_counts].append(self.result_values[iteration][3]*100.0) #* save the last value of the iteration.
        self.average_state[run_counts].append(self.cumulative_state / iteration)
        self.average_gt[run_counts].append(self.cumulative_gt / iteration)
        # print(self.epoch_accuracy)

    def save_avg_accuracy(self, run_counts, name): #! iterate and make average of the iterations.
        # print(len(self.epoch_accuracy[0]))
        final_acc, final_dtt, final_gt = [], [], []
        for j in range(len(self.epoch_accuracy[0])):
            temp ={0:0, 1:0, 2:0} #acc, dtt, gt

            for i in range(run_counts):
                # print("i {} j {}".format(i,j))
                temp[0]+=self.epoch_accuracy[i][j]  
                temp[1]+=self.average_state[i][j] 
                temp[2]+=self.average_gt[i][j] 
                # print(self.epoch_accuracy[i][j])
            # print("j: {}, avg {}".format(j, (temp / run_counts)))
            final_acc.append(temp[0]/run_counts)
            final_dtt.append(temp[1]/run_counts)
            final_gt.append(temp[2]/run_counts)
        print(final_acc[-1])
        row = {"id": str(name), "v_d": name.v_d, "v_lr": name.v_lr, "v_df": name.v_df, "v_eps": name.v_eps, "v_fd": name.v_fd, "v_s": name.v_s, "v_i": name.v_i, "yvalue": final_acc, "avg_dtt": final_dtt, "avg_gt": final_gt}
        self.accrewcollection.insert_one(row)

    def reset(self): #* per iteration reset
        self.result_values = defaultdict(lambda: [0, 0, 0, 0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, TN, FN, FP
        self.cur_decision ={}
        # self.cumulative_reward = 0
        self.cumulative_state = 0
        self.cumulative_gt = 0
        self.next_car_index = 1

    def resetepoch(self): #* per epoch reset
        # self.average_iteration_accgt = []
        # self.cumulative_iteration_reward = []
        self.epoch_accuracy =defaultdict(list)
        self.average_state =defaultdict(list)
        self.average_gt =defaultdict(list)

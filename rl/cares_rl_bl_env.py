#* general
import numpy as np
import pandas as pd


#* for RL 
#import gym
#from gym import spaces
#from gym.utils import seeding
from collections import defaultdict
#* draw now
import matplotlib.pyplot as plt
from pymongo import MongoClient

class trustEnv:
    """ trust threshold getter
        the goal of this program is train our system to set dynamic trust threshold 
        
    """
    def __init__(self, thrvalue, deltavalue, rewvalue, beta, filename):
        super(trustEnv, self).__init__()
        self.originals = [thrvalue, deltavalue, rewvalue]
        #* creates 3 actions: increase t_threshold OR decrease t_threshold
        #* general parameters
        print("[INFO] Reading file...")
        self.data = pd.read_csv('../sampledata/'+filename, header=0)
        print("[INFO] File loaded")

        # self.action_space = ["uu", "ud", "us", "du", "dd", "ds", "su", "sd", "ss"] #* {state, beta} x {up, down, stay}
        self.action_space = ["u", "d", "s"]
        self.n_actions = len(self.action_space)
        self.delta = deltavalue
        # self.bdelta = bdelta
        self.dtt = thrvalue
        self.beta = beta #* added beta!
        self.state = None
        self.reward_value = rewvalue
        #* epoch parameters
        self.previousFP=0.0
        self.previousFN=0.0
    
        self.average_gt=defaultdict(list)
        self.average_state= defaultdict(list)
        self.accuracy = defaultdict(list)
        self.precision = defaultdict(list)
        self.recall = defaultdict(list)
        self.cum_reward = defaultdict(list)
        #* step parameters
        self.cur_decision ={}
        self.next_car_index = 1
        self.cumulative_reward =0 
        self.cumulative_state=0
        self.cumulative_gt =0
        self.result_values = defaultdict(lambda: [0, 0, 0, 0,0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        self.tempcases = defaultdict(lambda:[0,0,0,0]) #* temporary cases for 100 vehicles. 
        self.client = None
        self.connect()

    def connect(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['trustdb']
        self.accrewcollection = self.db['cares']

    def get_car(self):
        
        tv_id = int(self.data['indirect_tv'][self.next_car_index]*100)
        tv_d = int(self.data['direct_tv'][self.next_car_index]*100)

        tv = (self.beta * tv_id + (1-self.beta) * tv_d) #! if beta close to 1 -> indirect evidence 
        
        if tv > self.dtt:
            self.cur_decision[self.next_car_index]=0 #! if benign, it should be over dtt
        else: 
            self.cur_decision[self.next_car_index]=1  #! if malicious, it should be under dtt
        
        self.result_values[self.next_car_index][0] = tv
        self.result_values[self.next_car_index][1] = self.dtt # dynamic threshold 값.
        self.cumulative_gt+=tv

    def step3(self, action, car_id): 
        #* inputs an action to current state & gets a reward to this action
        reward = 0
        if action == 0: 
            if self.dtt + self.delta < 100: 
                self.dtt+=self.delta
            else:
                #! penalty for hitting the ceiling!
                reward -= self.reward_value

        elif action == 1: # du
            if self.dtt - (self.delta) > 0:
                self.dtt-=self.delta
            else:
                #! penalty for hitting the ceiling!
                reward -= self.reward_value

        else: # ss
            self.dtt-=0
        
        ###* Reward주는 방법 
        ###* 방법1) 여기서 gt_accuracy에서 구한 값의 (TP + TN) / (TP+TN+FP+FN) 로 계산해서 reward값 선정. 
        
        # temperaryaccuracy = ( self.tempcases['gt'][0] + self.tempcases['gt'][3] ) / ( self.tempcases['gt'][0] + self.tempcases['gt'][1] + self.tempcases['gt'][2] + self.tempcases['gt'][3]) 
        # if temperaryaccuracy > self.previousaccuracy: 
        #     reward+=self.reward_value
        # else:
        #     reward-=self.reward_value
        ###* 방법2) 구체적으로 FP나 FN의 갯수가 증가되면 negative reward를 주는 방식? 
        if self.tempcases['gt'][1] > self.previousFP or self.tempcases['gt'][2] > self.previousFN:
            reward -= self.reward_value *2
                
        self.previousFP = self.tempcases['gt'][1]
        self.previousFN = self.tempcases['gt'][2]

        self.cumulative_reward += reward
        self.cumulative_state += self.dtt
        
        self.tempcases = defaultdict(lambda:[0, 0, 0, 0]) #! tempcases를 초기화 해야됨.  
        return reward, self.dtt #self.state

    def gt_evaluate(self, nci): #* gets gt accuracy regardless of time
        if self.cur_decision[nci] == 1 and self.data['status'][nci] == 1: #TP
            self.cases["gt"][0]+=1
            self.tempcases['gt'][0]+=1
        elif self.cur_decision[nci]==1 and self.data['status'][nci] == 0: #!FP
            self.cases["gt"][1]+=1
            self.tempcases['gt'][1]+=1
        elif self.cur_decision[nci]==0 and self.data['status'][nci] == 1: #FN 
            self.cases["gt"][2]+=1
            self.tempcases['gt'][2]+=1
        else: #!TN
            self.cases["gt"][3]+=1
            self.tempcases['gt'][3]+=1

        
        #* precision
        if self.cases["gt"][0] + self.cases["gt"][1] == 0:
            self.result_values[nci][2]=100
        else:
            self.result_values[nci][2] = (self.cases["gt"][0])/(self.cases["gt"][0] + self.cases["gt"][1]) *100 
        #* accuracy
        if (self.cases["gt"][0] + self.cases["gt"][1] + self. cases["gt"][2] + self.cases["gt"][3]) ==0:
            self.result_values[nci][3] =100
        else:
            self.result_values[nci][3] = (self.cases["gt"][0] + self.cases["gt"][3])/(self.cases["gt"][0] + self.cases["gt"][1] + self.cases["gt"][2] + self.cases["gt"][3]) *100 
        #* recall
        if (self.cases["gt"][0] + self.cases["gt"][2]) == 0:
            self.result_values[nci][4]=100
        else:
            self.result_values[nci][4] = (self.cases["gt"][0])/(self.cases["gt"][0] + self.cases["gt"][2]) *100
        # print(nci, self.result_values[nci])
        # print(nci, self.cases)
    def append_accuracy(self, run_counts, step):
        self.precision[run_counts].append(self.result_values[step][2])
        self.accuracy[run_counts].append(self.result_values[step][3]) 
        self.recall[run_counts].append(self.result_values[step][4])
        self.average_state[run_counts].append(self.cumulative_state / step)
        self.average_gt[run_counts].append(self.cumulative_gt / step)
        self.cum_reward[run_counts].append(self.cumulative_reward)

    def save_avg_accuracy(self, run_counts, name): #! iterate and make average of the iterations.
        # print(len(self.accuracy[0]))
        final_acc, final_dtt, final_gt, final_rew, final_precision, final_recall= [], [], [], [], [], []
        # print(len(self.accuracy[0]))
        for j in range(len(self.accuracy[0])):
            temp ={0:0, 1:0, 2:0, 3:0, 4:0, 5:0} #acc, dtt, gt, rew

            for i in range(run_counts):
                # print("i {} j {}".format(i,j))
                temp[0]+=self.accuracy[i][j]  
                temp[1]+=self.average_state[i][j] * 100
                temp[2]+=self.average_gt[i][j] 
                temp[3]+=self.cum_reward[i][j]
                temp[4]+=self.precision[i][j] 
                temp[5]+=self.recall[i][j] 
            # print(temp)
            final_acc.append(temp[0]/run_counts)
            final_dtt.append(temp[1]/run_counts)
            final_gt.append(temp[2]/run_counts)
            final_rew.append(temp[3]/run_counts)
            final_precision.append(temp[4]/run_counts)
            final_recall.append(temp[5]/run_counts)
        
        print("Accuracy: ", final_acc[-1])
        print("Precision: ", final_precision[-1])
        print("Recall: ", final_recall[-1])
        row = {"id": str(name), 'v_mvp': name.v_mvp, 'v_mbp': name.v_mbp, 'v_oap': name.v_oap, 'v_interval':name.v_interval, "v_d": name.v_d, "v_lr": name.v_lr, "v_df": name.v_df, "v_eps": name.v_eps, "v_fd": name.v_fd, "v_s": name.v_s, "v_i": name.v_i, "accuracy": final_acc, "avg_dtt": final_dtt, "avg_gt": final_gt, "cum_rew": final_rew, 'precision': final_precision, 'recall': final_recall}
        # self.accrewcollection.insert_one(row)

    def reset(self): #* per iteration reset
        self.result_values = defaultdict(lambda: [0, 0, 0, 0,0]) #* actual trust value, dynamic threshold value, estimated accuracy, actual accuracy
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        self.cur_decision ={}
        self.cumulative_reward = 0
        self.cumulative_state = 0
        self.cumulative_gt = 0
        self.next_car_index = 1
        self.dtt = self.originals[0]

    def resetepoch(self): #* per epoch reset

        self.accuracy =defaultdict(list)
        self.average_state =defaultdict(list)
        self.average_gt =defaultdict(list)

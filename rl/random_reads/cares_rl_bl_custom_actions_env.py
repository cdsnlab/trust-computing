""" 
trust environment    
"""
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
import statistics

class trustEnv:

    def __init__(self, output, rewvalue, filename):
        super(trustEnv, self).__init__()
        self.originals = [output.v_i, output.v_d, rewvalue]
        #* creates 3 actions: increase t_threshold OR decrease t_threshold
        #* general parameters
        print("[INFO] Reading file...")
        self.data = pd.read_csv('../../sampledata/'+filename, header=0)
        print("[INFO] File loaded")
        self.action_space = None
        self.create_action_space( [-15, -10, -5, 0, 5, 10, 15], [-15, -10, -5, 0, 5, 10, 15], output.v_d) #* beta_d, dtt_d
        print(self.action_space)
        self.n_actions = len(self.action_space)

        self.delta = output.v_d
        self.dtt = output.v_i
        self.beta = output.v_i #* added beta!
        self.state = None
        self.reward_value = rewvalue
        self.PPV_THR = output.v_ppvnpvthr
        self.NPV_THR = output.v_ppvnpvthr
    
        #* epoch parameters
        self.average_reward = defaultdict(list)
        self.average_dtt= defaultdict(list)
        self.step_accuracy=defaultdict(list)
        self.cum_accuracy = defaultdict(list)
        self.precision = defaultdict(list)
        self.recall = defaultdict(list)
        self.f1score = defaultdict(list)
        self.result_values = defaultdict(lambda: [0, 0, 0, 0, 0, 0,0]) #* actual trust value, dtt, precision, accuracy, recall, f1
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        self.tempcases = defaultdict(lambda:[0,0,0,0]) #* temporary cases for 100 vehicles. 
        self.betarecord = defaultdict(list)

        #* step parameters
        self.cur_decision ={}
        self.step_reward = 0
        self.step_dtt=0
        self.cumulative_gt =0

        #* step parameters
        self.cumulative_gt =0

        self.client = None
        self.connect()
    
    def create_action_space(self, deltas, betas, gap): #! creates merged list of actions {deltas, betas}
        merged_list = [(p1*gap, p2*gap) for idx1, p1 in enumerate(deltas) for idx2, p2 in enumerate(betas)]

        self.action_space = merged_list  
    def connect(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['trustdb']
        self.accrewcollection = self.db['cares_rl_bl_custom_manual']

    def make_decision(self, samplelist):        
        for index, sid in enumerate (samplelist):
            tv_id = int(self.data['indirect_tv'][sid]*100)
            tv_d = int(self.data['direct_tv'][sid]*100)

            tv = ((self.beta/100) * tv_id + (1-(self.beta/100)) * tv_d) #! if beta close to 1 -> indirect evidence 
        
            if tv > self.dtt: 
                self.cur_decision[sid]=0 #! if benign, it should be over dtt
            else: 
                self.cur_decision[sid]=1  #! if malicious, it should be under dtt
        
    def step(self, action): 
        #* inputs an action to current state & gets a reward to this action
        reward = 0
        PPV =0
        NPV=0
        action_beta = int(self.action_space[action][0]) #* beta change delta
        action_dtt = int(self.action_space[action][1]) #* dtt change delta

        self.beta+=action_beta
        self.dtt+=action_dtt
        
        ###* Reward주는 방법 
        ###* 방법1) 여기서 gt_accuracy에서 구한 값의 (TP + TN) / (TP+TN+FP+FN) 로 계산해서 reward값 선정. 
        
        # temperaryaccuracy = ( self.tempcases['gt'][0] + self.tempcases['gt'][3] ) / ( self.tempcases['gt'][0] + self.tempcases['gt'][1] + self.tempcases['gt'][2] + self.tempcases['gt'][3]) 
        # if temperaryaccuracy > self.previousaccuracy: 
        #     reward+=self.reward_value
        # else:
        #     reward-=self.reward_value
        ###* 방법2) 구체적으로 FP나 FN의 갯수가 이전에 비해서 증가되면 negative reward를 주는 방식? 
        # if self.tempcases['gt'][1] > self.previousFP or self.tempcases['gt'][2] > self.previousFN:
        #     reward -= self.reward_value * (self.tempcases['gt'][1]+self.tempcases['gt'][2]) /100
        # else:
        #     reward += self.reward_value
                
        # self.previousFP = self.tempcases['gt'][1]
        # self.previousFN = self.tempcases['gt'][2]
        ###* 방법3) FN가 가장 낮춰야 할 목표. FP는 그다음, ceiling/floor 치는 것은 그다음 중요도.
        # if self.tempcases['gt'][2] > self.previousFN:
        #     # reward -= self.reward_value * 10        
        #     pass
        # else:
        #     reward += self.reward_value *10
        # if self.tempcases['gt'][1] > self.previousFP:
        #     # reward -= self.reward_value * 10
        #     pass
        # else:
        #     reward += self.reward_value *10
        # print(self.next_car_index, self.tempcases['gt'], self.previousFN, self.previousFP, self.dtt, reward )

        # self.previousFP = self.tempcases['gt'][1]
        # self.previousFN = self.tempcases['gt'][2]        

        ###* 방법4) FP, FN 변화량에 따른 방법: #! 이거 리워드 shaping이 너무 어렵네..
        # if self.tempcases['gt'][2] - self.ttempcases['gt'][2] >= 0: #! FN가 
        #     reward -= self.reward_value*8
        # else:
        #     reward += self.reward_value*4
        # if self.tempcases['gt'][1] - self.ttempcases['gt'][1] >= 0: #! FP는 
        #     reward -= self.reward_value*6
        # else:
        #     reward += self.reward_value*2

        # self.ttempcases['gt'] = self.tempcases['gt']
        # self.tempcases = defaultdict(lambda:[0, 0, 0, 0]) #! tempcases를 초기화 해야됨.  


        ###* 방법5) 이전 accuracy보다 증가되는지, 감소되는지로만 판단?

        ###* 방법6) PPV, NPV 매커니즘 그대로 활용해볼 것. 1이되면 가장 accurate하게 걸러내는 것! 0.95이하로 되면 올리기.
        # PPV_THR, NPV_THR = 0.95, 0.95
        # if (self.tempcases['gt'][0] + self.tempcases['gt'][1]) == 0:
        #     PPV = 0
        # else:
        #     PPV = self.tempcases['gt'][0] / (self.tempcases['gt'][0] + self.tempcases['gt'][1])
        # if (self.tempcases['gt'][3] + self.tempcases['gt'][2]) == 0:
        #     NPV = 0
        # else:
        #     NPV = self.tempcases['gt'][3] / (self.tempcases['gt'][3] + self.tempcases['gt'][2])

        # if (PPV > self.PPV_THR and NPV > self.NPV_THR):
        #     reward += self.reward_value*2
        # elif(NPV < self.NPV_THR):
        #     reward -= self.reward_value
        # elif(PPV < self.PPV_THR):
        #     reward -= self.reward_value

        ###* 방법7: FP/all, FN/all이면 reward아니면 ?
        FPR = self.tempcases['gt'][1] / (self.tempcases['gt'][0] + self.tempcases['gt'][1] + self.tempcases['gt'][2] + self.tempcases['gt'][3])
        FNR = self.tempcases['gt'][2] / (self.tempcases['gt'][0] + self.tempcases['gt'][1] + self.tempcases['gt'][2] + self.tempcases['gt'][3])
        if FPR < 0.05:
            reward +=self.reward_value
        else :
            reward -=self.reward_value
        if FNR < 0.05:
            reward +=self.reward_value
        else :
            reward -=self.reward_value
        # print("FPR: {}, FNR: {}, Rew: {}, DTT: {}".format(FPR, FNR, reward, self.dtt))

        self.step_reward=reward
        self.step_dtt = self.dtt
        self.state = (self.beta, self.dtt)
        
        return reward, self.state


    def evaluate(self, intnumb, samplelist): 
        for index, sid in enumerate (samplelist):
            if self.cur_decision[sid] == 1 and self.data['status'][sid] == 1: #TP
                self.cases["gt"][0]+=1
                self.tempcases['gt'][0]+=1
            elif self.cur_decision[sid]==1 and self.data['status'][sid] == 0: #!FP
                self.cases["gt"][1]+=1
                self.tempcases['gt'][1]+=1
            elif self.cur_decision[sid]==0 and self.data['status'][sid] == 1: #FN 
                self.cases["gt"][2]+=1
                self.tempcases['gt'][2]+=1
            else: #!TN
                self.cases["gt"][3]+=1
                self.tempcases['gt'][3]+=1
            # print(intnumb, self.dtt, self.cur_decision[sid], self.data['status'][sid])
        # print(self.cases)
        #* precision
        if self.cases["gt"][0] + self.cases["gt"][1] == 0:
            self.result_values[intnumb][2]=0
        else:
            self.result_values[intnumb][2] = (self.cases["gt"][0])/(self.cases["gt"][0] + self.cases["gt"][1]) *100 
        #* accuracy
        if (self.cases["gt"][0] + self.cases["gt"][1] + self. cases["gt"][2] + self.cases["gt"][3]) ==0:
            self.result_values[intnumb][3] =0
        else:
            self.result_values[intnumb][3] = (self.cases["gt"][0] + self.cases["gt"][3])/(self.cases["gt"][0] + self.cases["gt"][1] + self.cases["gt"][2] + self.cases["gt"][3]) *100 
        #* recall
        if (self.cases["gt"][0] + self.cases["gt"][2]) == 0:
            self.result_values[intnumb][4]=0
        else:
            self.result_values[intnumb][4] = (self.cases["gt"][0])/(self.cases["gt"][0] + self.cases["gt"][2]) *100
        #* f1 score
        if self.result_values[intnumb][2]+self.result_values[intnumb][4] == 0:
            self.result_values[intnumb][5]=0
        else:
            self.result_values[intnumb][5]= (2*self.result_values[intnumb][2]*self.result_values[intnumb][4]) / (self.result_values[intnumb][2] + self.result_values[intnumb][4])
        self.result_values[intnumb][6] = self.beta
        #* beta values
        self.cur_decision ={}

    def step_records(self, run_counts, step):
        self.precision[run_counts].append(self.result_values[step][2])
        self.cum_accuracy[run_counts].append(self.result_values[step][3]) 
        self.recall[run_counts].append(self.result_values[step][4])
        self.f1score[run_counts].append(self.result_values[step][5])
        self.betarecord[run_counts].append(self.result_values[step][6])
        self.average_dtt[run_counts].append(self.step_dtt)
        self.average_reward[run_counts].append(self.step_reward )
        self.step_accuracy[run_counts].append((self.tempcases["gt"][0] + self.tempcases["gt"][3])/(self.tempcases["gt"][0] + self.tempcases["gt"][1] + self.tempcases["gt"][2] + self.tempcases["gt"][3]) *100) 
        # print(self.step_accuracy)
        self.tempcases = defaultdict(lambda:[0, 0, 0, 0]) 

        # print(self.average_reward)

    def save_avg_accuracy(self, run_counts, name): #! iterate and make average of the iterations.
        # print(len(self.cum_accuracy[0]))
        final_cum_acc, final_dtt, final_gt, final_rew, final_precision, final_recall, final_acc_error, final_f1, final_avg_acc, final_beta= [], [], [], [], [], [], [], [], [], []
        # print(len(self.cum_accuracy[0]))
        # print(self.cum_accuracy[:][-1])
        for j in range(len(self.cum_accuracy[0])):
            temp ={0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0} #acc, dtt, gt, rew
            errors=[]
            for i in range(run_counts):
                # print("i {} j {}".format(i,j))
                temp[0]+=self.cum_accuracy[i][j]  
                temp[1]+=self.average_dtt[i][j] 
                temp[3]+=self.average_reward[i][j]
                temp[4]+=self.precision[i][j] 
                temp[5]+=self.recall[i][j] 
                temp[6]+=self.f1score[i][j]
                temp[7]+=self.betarecord[i][j]
                temp[8]+=self.step_accuracy[i][j]
                errors.append(self.cum_accuracy[i][j])
            
            final_acc_error.append(statistics.stdev(errors))
            final_cum_acc.append(temp[0]/run_counts)
            final_dtt.append(temp[1]/run_counts)
            final_rew.append(temp[3]/run_counts)
            final_precision.append(temp[4]/run_counts)
            final_recall.append(temp[5]/run_counts)
            final_f1.append(temp[6]/run_counts)
            final_beta.append(temp[7]/ run_counts)
            final_avg_acc.append(temp[8]/run_counts)


        print("Accuracy: ", final_cum_acc[-1])
        print("Precision: ", final_precision[-1])
        print("Recall: ", final_recall[-1])
        print("F1 score: ", final_f1[-1])

        row = {"id": str(name), 'v_mvp': name.v_mvp, 'v_mbp': name.v_mbp, 'v_oap': name.v_oap,  "v_d": name.v_d, "v_lr": name.v_lr, "v_df": name.v_df, "v_eps": name.v_eps, "v_fd": name.v_fd, "v_s": name.v_s, "v_i": name.v_i, "cum_accuracy": final_cum_acc, "step_accuracy": final_avg_acc, "avg_dtt": final_dtt, "cum_rew": final_rew, 'precision': final_precision, 'f1score': final_f1, 'recall': final_recall,"error":final_acc_error, 'beta_changes':final_beta, 'v_ppvnpvthr': name.v_ppvnpvthr}
        self.accrewcollection.insert_one(row)

    def reset(self): #* per iteration reset
        self.result_values = defaultdict(lambda: [0, 0, 0, 0, 0, 0,0]) #* actual trust value, dtt, precision, accuracy, recall, f1
        self.cases = defaultdict(lambda:[0, 0, 0, 0]) #* TP, FP, FN, TN
        self.cur_decision ={}
        self.step_dtt = 0
        self.step_reward = 0
        self.dtt = self.originals[0]

#* general
'''
instead of repeating the traffic pattern as EPOCHs, lets use the number of vehicles as EPOCHs (x-axis).
'''
import numpy as np
import pandas as pd
import os, time
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import queue
from itertools import product, starmap
from collections import namedtuple
from tqdm import tqdm
#* for RL 
from trustEnv import trustEnv

import faulthandler
faulthandler.enable()

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))
    
class QLearningAgent():
    def __init__ (self, actions, lr, df, eps):    
        self.actions = actions
        self.learning_rate = lr
        self.discount_factor = df
        self.epsilon = eps
        self.q_table = defaultdict(lambda:[0, 0, 0]) #* this depends on the number of actions the system can make.
    
    def learn (self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)
        # print(self.q_table)


    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    #* v_d: delta, v_lr: learning_rate, v_df: discount_factor, v_eps: epsilon, v_fd: feedback_delay, v_s: 
    for output in named_product(v_d=[1,2,3,4,5,6,7,8,9,10], v_lr=[0.01], v_df=[0.1], v_eps=[0.1], v_fd=[1], v_s=[30000], v_i=[50]):
    # for output in named_product(v_d=[4,5,6], v_lr=[0.01, 0.1, 0.5], v_df=[0.01, 0.1, 0.5], v_eps=[0.1, 0.5], v_fd=[1, 5, 10], v_s=[30000], v_i=[10, 50, 90]):
    
        env = trustEnv(output.v_i, output.v_d, 1)
        agent = QLearningAgent(list(range(env.n_actions)), output.v_lr, output.v_df, output.v_eps)
        
        DELAY = output.v_fd
        STEPS = output.v_s
        evaluation_q = queue.Queue(DELAY)
        env.state = output.v_i
        state = env.state
        run_counts = 10

        env.connect()
        time.sleep(1)

        for i in range(run_counts): #* RUN THIS xxx times each and make an average.

            while True: 

                if env.next_car_index <= STEPS:
                    car_id = env.next_car_index
                    evaluation_q.put(car_id)
                    env.get_car()

                if env.next_car_index >= DELAY : #! 사실상 이때까지 step을 하지 않음으로 feedback이 반영되지 않음.
                    # car_id, perceived_btrust = evaluation_q.get() #! 딱히 perceived_btrust 가 필요 없음 .. 
                    car_id=evaluation_q.get()
                    # take action and proceed one step in the environment
                    env.gt_accuracy(car_id) 
                    action = agent.get_action(str(state))
                    
                    reward, next_state = env.step2(action, car_id)

                    # with sample <s,a,r,s'>, agent learns new q function
                    agent.learn(str(state), action, reward, str(next_state))

                    state = next_state
                    env.append_accuracy(i, car_id) #* adds accuracy to the list
                    
                # this is the end of one simulation
                if env.next_car_index == (STEPS+DELAY)-1:
                    #print("{} finished ".format(output))
                    env.reset()
                    break
                env.next_car_index+=1
        #* this is the end of run_counts, you average everything and save it.
        print("{} finished ".format(output))
        env.save_avg_accuracy(run_counts, output)
        env.client.close()
        # env.resetepoch() #어짜피 새로 object를 만들어 버려서 reset할 필요도 없겠는데.
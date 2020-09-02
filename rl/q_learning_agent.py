#* general
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import queue
from itertools import product, starmap
from collections import namedtuple
from tqdm import tqdm

#* for RL 
from trustEnv import trustEnv


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


    
class QLearningAgent():
    def __init__ (self, actions, lr, df, eps):    
        self.actions = actions
        self.learning_rate = lr
        self.discount_factor = df
        self.epsilon = eps
        self.q_table = defaultdict(lambda:[0, 0, 0])

    def learn (self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

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

    #* lets leave this one till later.
    v_number_of_actions = [2, 4, 6] #* up down, upx2 downx2, upx4 downx4
    
    filename = "result/iteration.json"
    for output in tqdm(named_product(v_d=[1,3,5], v_lr=[0.01,0.1,0.5, 0.9], v_df=[0.1, 0.5, 0.9], v_eps=[0.1, 0.5, 0.9], v_fd=[100, 200, 500], v_s=[1000, 10000, 50000], v_i=[10, 50, 90])):
        #! 이렇게 돌리니깐 JSON파일을 읽고, 내용 추가하고, 저장하고를 반복하다보니깐 너무 시간이 오래걸림.

        env = trustEnv(output.v_i, output.v_d, 1)
        agent = QLearningAgent(list(range(env.n_actions)), output.v_lr, output.v_df, output.v_eps)
        
        DELAY = output.v_fd
        STEPS = output.v_s
        evaluation_q = queue.Queue(DELAY)
        env.state = output.v_i
        state = env.state
    
        while True:
            
            #if not evaluation_q.full():
            if env.next_car_index <= STEPS:
                car_id = env.next_car_index
                car_trust_val = env.get_car()
                if car_trust_val > state:
                    evaluation_q.put((car_id, 1))
                else:
                    evaluation_q.put((car_id, 0))
            
            if env.next_car_index >= DELAY :
                car_id, perceived_btrust = evaluation_q.get()
                # take action and proceed one step in the environment
                env.gt_accuracy(car_id) 

                action = agent.get_action(str(state))

                reward, next_state = env.step2(action, car_id, perceived_btrust)

                # with sample <s,a,r,s'>, agent learns new q function
                agent.learn(str(state), action, reward, str(next_state))

                state = next_state
            
            #print(env.next_car_index)
            # if episode ends, then break
            if env.next_car_index == (STEPS+DELAY)-1:
                #env.drawgraph()
                env.savetojson(output, filename) #! save this results to the file. READ_JSON APPEND SAVE REPEAT
                break

            env.next_car_index+=1
        print("finished: {}".format(output))

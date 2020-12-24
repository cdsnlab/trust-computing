#* general
'''
only rewards at 100 vehicles..............?
'''
import numpy as np
import time
from collections import defaultdict
import random
import queue
from itertools import product, starmap
from collections import namedtuple
#* for RL 
from cares_rl_bl_env import trustEnv

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
        self.q_table = defaultdict(lambda:[0, 0, 0, 0, 0, 0, 0, 0, 0]) #* this depends on the number of actions the system can make.
        # self.q_table = defaultdict(lambda:[0, 0, 0])
    
    def learn (self, state, action, reward, next_state):
        # print(state, action)
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)
        # print(self.q_table)

    def decayed_eps(self, current_step, max_step):
        p_init = self.epsilon
        p_end = 0.05
        r = max((max_step-current_step)/max_step, 0)
        # print((p_init-p_end)*r + p_end)
        self.epsilon=(p_init-p_end)*r + p_end

    def print_qtable(self): 
        print(self.q_table)

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
    for output in named_product(v_d=[3], v_bd=[0.5], v_lr=[0.1], v_df=[0.1], v_eps=[0.5], v_fd=[1], v_s=[11000], v_i=[50], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.2, 0.3, 0.4, 0.5], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3], v_interval=[100]):
    # for output in named_product(v_d=[1], v_bd=[0.5], v_lr=[0.1], v_df=[0.1], v_eps=[0.1], v_fd=[1], v_s=[11000], v_i=[50], v_mvp=[0.2], v_mbp=[0.5], v_oap=[0.2, 0.4], v_interval=[100]):
        filename = "rl_df_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
        env = trustEnv(output.v_i, output.v_d, 1, output.v_bd, filename)
        agent = QLearningAgent(list(range(env.n_actions)), output.v_lr, output.v_df, output.v_eps)
        
        DELAY = output.v_fd
        STEPS = output.v_s
        INTERVAL = output.v_interval
        evaluation_q = queue.Queue(DELAY)
        #env.dtt = output.v_i
        env.state = None
        state = env.state
        run_counts = 100

        # time.sleep(0.1)

        for i in range(run_counts): #* RUN THIS xxx times each and make an average.

            while True: 
                
                if env.next_car_index <= STEPS: 
                    # print("inq {}".format(env.next_car_index))
                    #car_id = env.next_car_index
                    evaluation_q.put(env.next_car_index)
                    env.get_car()

                if env.next_car_index >= DELAY : #! delay 이후에 실제 event에 대한 verification을 함.
                    # print("eval {}".format(env.next_car_index))
                    car_id=evaluation_q.get()
                    # take action and proceed one step in the environment
                    env.gt_evaluate(car_id) #! 정확도 기록.
                
                if env.next_car_index % INTERVAL == 0: #! 무조건 100일때마다 action 취하고, reward 받는걸로
                    # print("step {}".format(env.next_car_index))
                    agent.decayed_eps(env.next_car_index, STEPS)

                    # print(env.next_car_index)
                    action = agent.get_action(state)                    
                    reward, next_state = env.step2(action, env.next_car_index)
                    # with sample <s,a,r,s'>, agent learns new q function
                    agent.learn(state, action, reward, next_state)

                    state = next_state
                # if env.next_car_index % (100-DELAY) ==0:
                    env.append_accuracy(i, env.next_car_index-DELAY) #* adds accuracy to the list
                    
                # this is the end of one simulation
                if env.next_car_index == (STEPS+DELAY)-1:
                    print("run count: {} finished ".format(i))
                    # print("cases {}".format(env.cases))
                    print("dtt {}".format(env.dtt))
                    print("reward {}".format(env.cumulative_reward))
                    print("beta {}".format(env.beta))
                    print("Accuracy: {}".format(env.accuracy[i][-1]))
                    env.reset()
                    agent.q_table = defaultdict(lambda:[0, 0, 0, 0, 0, 0, 0, 0, 0])
                    # print("[INFO] Finished {}th ".format(i))
                    break
                env.next_car_index+=1
            # agent.print_qtable()
        #* this is the end of run_counts, you average everything and save it.
        print("{} finished ".format(output))
        # slacknoti("finished {}".format(output), "s")
        env.save_avg_accuracy(run_counts, output)


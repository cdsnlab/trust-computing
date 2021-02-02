#* general
import random
import numpy as np
import time
from collections import defaultdict
import random
import queue
from itertools import product, starmap
from collections import namedtuple
import statistics
from slack_noti import slacknoti

#* for RL 
from cares_rl_sb_env import trustEnv

import faulthandler
faulthandler.enable()

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))
    
class QLearningAgent():
    # def __init__ (self, actions, lr, df, eps):   
    def __init__ (self, actions, output):    
        self.actions = actions
        self.learning_rate = output.v_lr
        self.discount_factor = output.v_df
        self.p_init=output.v_eps
        self.epsilon = output.v_eps
        self.q_table = defaultdict(lambda:[0, 0, 0])
    
    def learn (self, state, action, reward, next_state):
        # print(state, action)
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)
        # print(self.q_table)

    def decayed_eps(self, current_step, max_step): 
        p_end = 0.05
        r = max(((max_step)-current_step)/(max_step), 0) #!
        self.epsilon=(self.p_init-p_end)*r + p_end
        # print(current_step, self.epsilon)

    def print_qtable(self): 
        print(self.q_table)

    def get_action(self, state, action_list):
        #* block out unavailable choices.
        notDone = True
        while notDone:
            
            if np.random.rand() < self.epsilon:
                # take random action
                action = np.random.choice(self.actions)
            else:
                # take action according to the q function table
                state_action = self.q_table[state]
                action = self.arg_max(state_action)
            if action_list[action] + state < 100 and action_list[action] + state > 0:
                notDone = False
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
    
    def get_sample(self, max_value):
        sample_index = random.sample(range(max_value), 100)
        return sample_index
        

if __name__ == "__main__":
    # for output in named_product(v_d=[5, 10, 15], v_lr=[0.1], v_df=[0.1], v_eps=[0.1], v_fd=[1], v_s=[12000], v_i=[10, 50, 90], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.3, 0.5, 0.7, 0.9], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3], v_ppvnpvthr= [0.5]):
    for output in named_product(v_d=[10], v_lr=[0.1], v_df=[0.1], v_eps=[0.1], v_fd=[1], v_s=[12000], v_i=[10], v_mvp=[0.3], v_mbp=[0.9], v_oap=[0.1], v_ppvnpvthr= [0.5]): 
    # for output in named_product(v_d=[5], v_lr=[0.1], v_df=[0.1], v_eps=[0.5], v_fd=[1], v_s=[12000], v_i=[10], v_mvp=[0.1, 0.2, 0.3, 0.4], v_mbp=[0.1, 0.3, 0.5, 0.7, 0.9], v_oap=[0.1, 0.15, 0.2, 0.25, 0.3], v_ppvnpvthr= [0.1, 0.5, 0.9]): 

    # for output in named_product(v_d=[5], v_lr=[0.1], v_df=[0.1], v_eps=[0.1], v_fd=[1], v_s=[12000], v_i=[90], v_mvp=[0.3], v_mbp=[0.9], v_oap=[0.3], v_ppvnpvthr= [0.5]): 
        
        filename = "cares_df_0_"+str(output.v_mbp)+"mbp"+str(output.v_oap)+"oap"+str(output.v_mvp)+"mvp.csv"
        env = trustEnv(output,1, 0.5, filename)
        agent = QLearningAgent(list(range(env.n_actions)), output)
        
        DELAY = output.v_fd
        STEPS = output.v_s
        #env.dtt = output.v_i
        state = env.dtt
        run_counts = 100
        starttime, endtime = [], []

        for i in range(run_counts): 
            starttime.append(time.time())

            interaction_number=1

            while True:                
                samplelist = agent.get_sample(STEPS)
                
                env.make_decision(samplelist)
                
                env.evaluate(interaction_number, samplelist)
                
                agent.decayed_eps(interaction_number, STEPS/100)

                action = agent.get_action(state, list(env.action_space))
                reward, next_state = env.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                env.step_records(i, interaction_number)

                if interaction_number == (STEPS)/100:
                    print("run count: {} finished ".format(i))
                    print("dtt: {}".format(env.dtt))
                    # print("rew: {}".format(env.cumulative_reward))
                    print("Accuracy: {}".format(env.cum_accuracy[i][-1]))
                    print("Precision: {}".format(env.precision[i][-1]))
                    env.reset()
                    agent.q_table = defaultdict(lambda:[0, 0, 0])
                    agent.epsilon = 0.5
                    endtime.append(time.time())

                    # print("[INFO] Finished {}th ".format(i))
                    break
                
                interaction_number+=1
                
        
        avgsimtime=0
        errors=[]   
        for r in range(run_counts):
            # print(r)
            avgsimtime+=endtime[r]-starttime[r]
            errors.append(endtime[r]-starttime[r])
        print("avgsimtime: ",avgsimtime/run_counts)
        print(statistics.stdev(errors))

        print("{} finished ".format(output))        
        env.save_avg_accuracy(run_counts, output)

slacknoti("CARES-S SB finished", 's')

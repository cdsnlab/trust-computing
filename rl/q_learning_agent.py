#* general
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random

import matplotlib.pyplot as plt
import queue

#* for NN models
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
#* for RL 
from trustEnv import trustEnv
# import gym
# from gym import spaces
# from gym.utils import seeding
# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory
    
class QLearningAgent:
    def __init__ (self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
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
    env = trustEnv()
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    
    DELAY = 20
    evaluation_q = queue.Queue(DELAY)
    state = env.state
    while True:
        
        #env.get_accuracy()       
        if not evaluation_q.full() :
            car_id = env.next_car_index
            car_trust_val = env.get_car()
            if car_trust_val > state:
                evaluation_q.put((car_id, 1))
            else:
                evaluation_q.put((car_id, 0))

        if evaluation_q.full():
            car_id, perceived_btrust = evaluation_q.get()
            # take action and proceed one step in the environment
            action = agent.get_action(str(state))

            reward, next_state = env.step2(action, car_id, perceived_btrust)

            # with sample <s,a,r,s'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state


        # if episode ends, then break
        if env.next_car_index == 99999:
            env.drawgraph()
            break
        #print(agent.q_table)



#policy = EpsGreedyQPolicy()
#memory = SequentialMemory(limit=50000, window_length=1)

#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
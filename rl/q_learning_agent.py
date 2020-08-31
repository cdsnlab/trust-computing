#* general
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import queue
import argparse

#* for RL 
from trustEnv import trustEnv

    
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

    argparser = argparse.ArgumentParser(
        description="welcome")
    argparser.add_argument(
        '--lr',
        metavar='lr',
        type = float, 
        default= 0.01,
        help='learning rate'
    )
    argparser.add_argument(
        '--df',
        metavar='df',
        type = float, 
        default= 0.9,
        help='discount factor'
    )
    argparser.add_argument(
        '--eps',
        metavar='eps',
        type = float, 
        default= 0.1,
        help='epsilon'
    )
    argparser.add_argument(
        '--fd',
        metavar='fd',
        type = int, 
        default= 80,
        help='feedback delay'
    )
    argparser.add_argument(
        '--d',
        metavar='d',
        type = int, 
        default= 1,
        help='delta value'
    )
    argparser.add_argument(
        '--r',
        metavar='r',
        type = int, 
        default= 1,
        help='reward'
    )
    argparser.add_argument(
        '--s',
        metavar='s',
        type = int, 
        default= 999,
        help='steps'
    )
    argparser.add_argument(
        '--thr',
        metavar='thr',
        type = int, 
        default= 60,
        help='initial threshold'
    )
    args = argparser.parse_args()


    env = trustEnv(args.thr, args.d, args.r)
    agent = QLearningAgent(list(range(env.n_actions)), args.lr, args.df, args.eps)
    
    DELAY = args.fd
    STEPS = args.s
    evaluation_q = queue.Queue(DELAY)
    state = env.state
    while True:
        
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
        if env.next_car_index == STEPS:
            env.drawgraph()
            break

        env.gt_accuracy(env.next_car_index) #* writes down gt accuracy
        env.next_car_index+=1



#policy = EpsGreedyQPolicy()
#memory = SequentialMemory(limit=50000, window_length=1)

#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
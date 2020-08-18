#* general
import numpy as np
import pandas as pd
import os
#* for NN models
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
#* for RL 
from trustEnv import trustEnv
import gym
from gym import spaces
from gym.utils import seeding
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
    


#* read dataset
print("[INFO] Reading file...")
data = pd.read_csv('../sampledata/data_6_.txt', sep='\t', header=0)
print("[INFO] File loaded")

te = trustEnv(data)



#policy = EpsGreedyQPolicy()
#memory = SequentialMemory(limit=50000, window_length=1)

# TODO 
#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
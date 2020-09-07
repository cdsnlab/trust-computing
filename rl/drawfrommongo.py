import numpy as np
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product, starmap
from collections import namedtuple
from pymongo import MongoClient

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    #coll = db['q_learning']
    coll = db['acc_rew']

    return coll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


def reconstruct(d, lr, df, eps, fd, s, i): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i):
        allcombinations.append(str(output))
        #print(output)
    return allcombinations

def getxvalue(key): #parse for v_s

    elements = key.split(',')
    for i in elements:
        if 'v_s' in i:
            temp=i.split('=')
            return temp[1]
            
    
data_load_state = st.text('Loading data...')
coll = connect()
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct multiselect options, match it with 
d =   st.multiselect("Delta", [1, 3, 5], default=[1])
lr =  st.multiselect("Learning rate", [0.01,0.1,0.5, 0.9], default=[0.01])
df =  st.multiselect("Discount factor", [0.1, 0.5, 0.9], default=[0.1])
eps = st.multiselect("Epsilon", [0.1, 0.5, 0.9], default=[0.1])
fd =  st.multiselect("Feedback delay",[5, 10, 50, 100, 200, 500], default=[5])
s =   st.multiselect("Total number of steps", [1000, 10000, 50000], default=[1000])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[90])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.
allcombination = reconstruct(d, lr, df, eps, fd, s, i)

fig_accuracy = go.FigureWidget()
fig_cumulative_reward = go.FigureWidget()
for k in allcombination:
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    mydoc=list(coll.find(myquery, {"_id":0, "yvalue": 1}))
    yvalue = mydoc[0]['yvalue']
    mydoc2=list(coll.find(myquery, {"_id":0, "ycrew": 1}))
    ycrew = mydoc2[0]['ycrew']
    x = np.arange(int(xvalue))
    fig_accuracy.add_trace(go.Scatter(x=x, y=yvalue, name=k))
    fig_cumulative_reward.add_trace(go.Scatter(x=x, y=ycrew, name=k))


st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_cumulative_reward, use_container_width=True)
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
    #coll = db['acc_rew']
    coll = db['testalpha5']

    return coll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


def reconstruct(d, lr, df, eps, fd, s, i, epoch): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, epoch=epoch):
        allcombinations.append(str(output))
        #print(output)
    return allcombinations

def getxvalue(key): #parse for epoch
    elements = key.split(',')
    for i in elements:
        #print(i)
        if "epoch" in i:
            temp=i.split('=')
            #print(temp[1][:-1])
            return temp[1][:-1]
            
    
data_load_state = st.text('Loading data...')
coll = connect()
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct multiselect options, match it with 
d =   st.multiselect("Delta", [1, 2, 3, 5], default=[2])
lr =  st.multiselect("Learning rate", [0.01,0.1,0.5, 0.9], default=[0.1])
df =  st.multiselect("Discount factor", [0.1, 0.5, 0.9], default=[0.1])
eps = st.multiselect("Epsilon", [0.1, 0.5, 0.9], default=[0.1])
fd =  st.multiselect("Feedback delay",[1, 5, 10, 50, 100, 500], default=[5])
s =   st.multiselect("Total number of steps", [500, 1000, 2000, 10000, 20000, 50000], default=[20000])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[90])
epoch = st.multiselect("epoch", [1000, 5000, 10000], default=[1000])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.
allcombination = reconstruct(d, lr, df, eps, fd, s, i, epoch)
fig_accuracy = go.FigureWidget(
    layout=go.Layout(yaxis=dict(range=[0, 100], tickvals=list(range(0,100,10))))
    )
# fig_cumulative_reward = go.FigureWidget()
# fig_average_state_vs_gt = go.FigureWidget()

for k in allcombination:
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    mydoc=list(coll.find(myquery, {"_id":0, "yvalue": 1}))
    print(mydoc)
    yvalue = mydoc[0]['yvalue']
    # mydoc2=list(coll.find(myquery, {"_id":0, "ycrew": 1}))
    # ycrew = mydoc2[0]['ycrew']
    # mydoc3=list(coll.find(myquery, {"_id":0, "yastate": 1, "yagt": 1}))
    # yastate = mydoc3[0]['yastate']
    # yagt = mydoc3[0]['yagt']
    x = np.arange(int(xvalue))
    fig_accuracy.add_trace(go.Scatter(x=x, y=yvalue))
    # fig_accuracy.add_layout(layout)
    # fig_accuracy.xlim(0,100)
    # fig_accuracy.ylim(0,100)
    # fig_cumulative_reward.add_trace(go.Scatter(x=x, y=ycrew, name=k))
    # fig_average_state_vs_gt.add_trace(go.Scatter(x=x, y=yastate, name=k))
    # fig_average_state_vs_gt.add_trace(go.Scatter(x=x, y=yagt, name=k))


st.plotly_chart(fig_accuracy, use_container_width=True)
#st.plotly_chart(fig_cumulative_reward, use_container_width=True)
# st.plotly_chart(fig_average_state_vs_gt, use_container_width=True)

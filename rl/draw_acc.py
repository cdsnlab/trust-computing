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
    coll = db['a5acc']

    return coll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, lr, df, eps, fd, s, i): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')

            return temp[1]
            
    
data_load_state = st.text('Loading data...')
coll = connect()
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct multiselect options, match it with 
d =   st.multiselect("Delta", [1, 2, 3], default=[2])
lr =  st.multiselect("Learning rate", [0.01,0.1,0.5, 0.9], default=[0.1])
df =  st.multiselect("Discount factor", [0.1, 0.5, 0.9], default=[0.1])
eps = st.multiselect("Epsilon", [0.1, 0.5, 0.9], default=[0.1])
fd =  st.multiselect("Feedback delay",[1, 5, 10, 50, 100, 500], default=[5])
s =   st.multiselect("Total number of steps", [50000], default=[50000])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[90])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, lr, df, eps, fd, s, i)
fig_accuracy = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="# vehicles looked at"),yaxis=dict(title="Accuracy (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )

for k in allcombination:
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    mydoc=list(coll.find(myquery, {"_id":0, "yvalue": 1}))
    yvalue = mydoc[0]['yvalue']
    x = np.arange(int(xvalue))
    fig_accuracy.add_trace(go.Scatter(x=x, y=yvalue))

st.plotly_chart(fig_accuracy, use_container_width=True)

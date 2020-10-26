import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from itertools import product, starmap
from collections import namedtuple
from pymongo import MongoClient

import faulthandler
faulthandler.enable()

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']
    #coll = db['q_learning']
    #coll = db['acc_rew']
    coll = db['new_tv_data']

    return coll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, bd, lr, df, eps, fd, s, i): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_bd=bd, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i):
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
d =   st.multiselect("Delta", [1,2,3,4,5,6,7,8,9,10], default=[1])
bd =   st.multiselect("Beta Delta", [0.01, 0.1], default=[0.01])
lr =  st.multiselect("Learning rate", [0.01,0.1,0.5, 0.9], default=[0.01])
df =  st.multiselect("Discount factor", [0.1, 0.5, 0.9], default=[0.1])
eps = st.multiselect("Epsilon", [0.1, 0.5, 0.9], default=[0.1])
fd =  st.multiselect("Feedback delay",[1, 2, 5, 10, 50, 100, 500], default=[1])
s =   st.multiselect("Total number of steps", [20000], default=[20000])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i)
fig_accuracy = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="# vehicles looked at"),yaxis=dict(title="Accuracy (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_dtt = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="# vehicles looked at"),yaxis=dict(title="DTT & GT", range=[0, 100], tickvals=list(range(0,100,10))))
    )
for k in allcombination:
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    mydoc=list(coll.find(myquery, {"_id":0, "yvalue": 1, "avg_dtt":1, "avg_gt":1}))
    yvalue = mydoc[0]['yvalue']
    avg_dtt = mydoc[0]['avg_dtt']
    avg_gt = mydoc[0]['avg_gt']
    x = np.arange(int(xvalue))
    fig_accuracy.add_trace(go.Scatter(x=x, y=yvalue))
    fig_dtt.add_trace(go.Scatter(x=x, y=avg_dtt, name="Average DTT value"))
    fig_dtt.add_trace(go.Scatter(x=x, y=avg_gt, name="Average GT value"))

st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_dtt, use_container_width=True)

'''
This file draws the change in beta value over time.
input files: CARES_RL_BL 
output diagram: Learning accuracy with RL algorithms
'''
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

    cares_rl_sb = db['cares_rl_sb']
    cares_rl_bl = db['cares_rl_bl']

    return cares_rl_sb, cares_rl_bl

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_bd=bd, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_interval=interval):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')

            return temp[1]

    
data_load_state = st.text('Loading data...')
cares_rl_sb, cares_rl_bl = connect()
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct multiselect options, match it with 
d =   st.multiselect("Delta", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13], default=[1, 3, 5, 7, 9])
bd =  st.multiselect("Beta", [0.01, 0.99, 0.5], default=[0.5])
lr =  st.multiselect("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
df =  st.multiselect("Discount factor", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
eps = st.multiselect("Epsilon", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.5])
fd =  st.multiselect("Feedback delay",[1, 2, 5, 10, 20, 50, 100,200], default=[1])
s =   st.multiselect("Total number of steps", [59999], default=[59999])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])
mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
oap =   st.multiselect("Outside attack probability", [0.1, 0.15, 0.2, 0.25, 0.3], default=[0.2])
interval =   st.multiselect("Update interval", [10, 20, 50, 100], default=[100])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval)

# allcombination.append(rwcombo)
# dtmcombo = reconstruct_dtm()
fig_accuracy = go.FigureWidget(
    layout=go.Layout(title="Accuracy",xaxis=dict(title="Time (s)"),yaxis=dict(title="Accuracy (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_precision = go.FigureWidget(
    layout=go.Layout(title="Precision",xaxis=dict(title="Time (s)"),yaxis=dict(title="Precision (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_recall = go.FigureWidget(
    layout=go.Layout(title="Recall",xaxis=dict(title="Time (s)"),yaxis=dict(title="Recall (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_dtt = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="Time (s)"),yaxis=dict(title="DTT changes", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_cum_rew = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="Time (s)"), yaxis=dict(title="cumulative reward"))
    )
#! draw ours.
for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'v_interval':1, 'cum_rew':1, 'avg_dtt':1}))
    # print(mydocrlsb)
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']

    x = np.arange(int(xvalue)/mydocrlsb[0]['v_interval'])
        
    fig_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, name=d))
    fig_precision.add_trace(go.Scatter(x=x, y=rlsbprecision, name="CARES-RL-SB"))
    fig_recall.add_trace(go.Scatter(x=x, y=rlsbrecall, name="CARES-RL-SB"))


    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']

    fig_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, name="CARES-RL-BL"))
    fig_precision.add_trace(go.Scatter(x=x, y=rlblprecision, name="CARES-RL-BL"))
    fig_recall.add_trace(go.Scatter(x=x, y=rlblrecall, name="CARES-RL-BL"))

    # print(x)

    fig_dtt.add_trace(go.Scatter(x=x, y=rlsbdtt, name="Average DTT value"))
    # fig_dtt.add_trace(go.Scatter(x=x, y=avg_gt, name="Average GT value"))
    fig_cum_rew.add_trace(go.Scatter(x=x, y=rlsbrew, name="Cumulative rewards"))


st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)
st.plotly_chart(fig_cum_rew, use_container_width=True)
st.plotly_chart(fig_dtt, use_container_width=True)

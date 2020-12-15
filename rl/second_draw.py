'''
this prog uses all available dataset. 
runs 100 epochs and makes an average on each step it takes. 
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

    coll = db['partial']
    rtmcoll = db['rtm']
    dtmcoll = db['dtm']
    istmcoll = db['istm']
    return coll, rtmcoll, dtmcoll, istmcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_bd=bd, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_interval=interval):
        allcombinations.append(str(output))
    return allcombinations

def reconstruct_rtm(s,mvp, mbp, oap, interval, dynamicthr):
    allcombinations=[]
    for output in named_product(v_s=s, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_interval=interval, v_dynamic=dynamicthr):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')

            return temp[1]
            
    
data_load_state = st.text('Loading data...')
coll, rtmcoll, dtmcoll, istmcoll = connect()
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct multiselect options, match it with 
d =   st.multiselect("Delta", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13], default=[1])
bd =  st.multiselect("Beta", [0.01, 0.99, 0.5], default=[0.5])
lr =  st.multiselect("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
df =  st.multiselect("Discount factor", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
eps = st.multiselect("Epsilon", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
fd =  st.multiselect("Feedback delay",[1, 2, 5, 10, 20, 50, 100,200], default=[1])
s =   st.multiselect("Total number of steps", [11000], default=[11000])
i =   st.multiselect("Initial starting value", [50], default=[50])
mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
oap =   st.multiselect("Outside attack probability", [0.1, 0.2, 0.3, 0.4], default=[0.2, 0.4])
interval =   st.multiselect("Update interval", [10, 20, 50, 100], default=[100])
dynamicthr = st.multiselect("dynamic threshold", [0,1], default=[0])

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval)
rtmcombo = reconstruct_rtm(s, mvp, mbp, oap, interval, dynamicthr)

# allcombination.append(rtmcombo)
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
# fig_dtt = go.FigureWidget(
#     layout=go.Layout(xaxis=dict(title="Time (s)"),yaxis=dict(title="DTT & GT", range=[0, 100], tickvals=list(range(0,100,10))))
#     )
# fig_cum_rew = go.FigureWidget(
#     layout=go.Layout(xaxis=dict(title="Time (s)"), yaxis=dict(title="cumulative reward"))
#     )

for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    
    mydoc=list(coll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1}))
    # print(mydoc)
    accuracy = mydoc[0]['accuracy']
    precision = mydoc[0]['precision']
    recall = mydoc[0]['recall']
    # avg_dtt = mydoc[0]['avg_dtt']
    # avg_gt = mydoc[0]['avg_gt']
    # cum_rew = mydoc[0]['cum_rew']
    x = np.arange(int(xvalue)/100)
    # print(x)
    fig_accuracy.add_trace(go.Scatter(x=x, y=accuracy, name="OURS"))
    fig_precision.add_trace(go.Scatter(x=x, y=precision, name="OURS"))
    fig_recall.add_trace(go.Scatter(x=x, y=recall, name="OURS"))

    # fig_dtt.add_trace(go.Scatter(x=x, y=avg_dtt, name="Average DTT value"))
    # fig_dtt.add_trace(go.Scatter(x=x, y=avg_gt, name="Average GT value"))
    # fig_cum_rew.add_trace(go.Scatter(x=x, y=cum_rew, name="Cumulative rewards"))

for j in rtmcombo:
    print(j)
    xvalue = getxvalue(j)
    myquery = {'id': str(j)}
    mydoc = list(rtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    accuracy = mydoc[0]['accuracy']
    precision = mydoc[0]['precision']
    recall = mydoc[0]['recall']   

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   

    x = np.arange(int(xvalue)/100)
    fig_accuracy.add_trace(go.Scatter(x=x, y=accuracy, name='RTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=dtm_accuracy, name='DTM'))

    fig_precision.add_trace(go.Scatter(x=x, y=precision, name='RTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=dtm_precision, name='DTM'))

    fig_recall.add_trace(go.Scatter(x=x, y=recall, name='RTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=dtm_recall, name='DTM'))

st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)

# st.plotly_chart(fig_dtt, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)

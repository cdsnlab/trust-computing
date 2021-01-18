'''
This file draws the difference it makes in how much the threshold hold move each time.
input files: CARES_RL * 
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

    rtmdcoll = db['rtm_d']
    rtmcoll = db['rtm']

    return rtmdcoll, rtmcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct_rtm(i, s,mvp, mbp, oap):
    allcombinations=[]
    for output in named_product(v_i = i, v_s=s, v_mvp=mvp, v_mbp=mbp, v_oap=oap):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')

            return temp[1]
            
    
data_load_state = st.text('Loading data...')
rtmdcoll, rtmcoll = connect()
data_load_state.text('Loading data...done!')
st.title("RTM / RTM-D")
#reconstruct multiselect options, match it with 
s =   st.multiselect("Total number of steps", [59999], default=[59999])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])
mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
oap =   st.multiselect("Outside attack probability", [0.1, 0.15, 0.2, 0.25, 0.3], default=[0.2])

rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

# allcombination.append(rwcombo)
# rtmcombo = reconstruct_rtm()
fig_accuracy = go.FigureWidget(
    layout=go.Layout(title="Accuracy",xaxis=dict(title="Time (s)"),yaxis=dict(title="Accuracy (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_precision = go.FigureWidget(
    layout=go.Layout(title="Precision",xaxis=dict(title="Time (s)"),yaxis=dict(title="Precision (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_recall = go.FigureWidget(
    layout=go.Layout(title="Recall",xaxis=dict(title="Time (s)"),yaxis=dict(title="Recall (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_f1 = go.FigureWidget(
    layout=go.Layout(title="F1 Score",xaxis=dict(title="Time (s)"),yaxis=dict(title="F1 score (%)", range=[0, 100], tickvals=list(range(0,100,10))))
    )
fig_dtt = go.FigureWidget(
    layout=go.Layout(xaxis=dict(title="Time (s)"),yaxis=dict(title="DTT changes", range=[0, 100], tickvals=list(range(0,100,10))))
    )

for j in rwcombo:
    print(j)
    xvalue = getxvalue(j)

    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1, 'v_interval':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1, 'v_interval':1, 'dtt':1}))
    # print(mydoc_rtm)
    # print(mydoc_rtmd)
    x = np.arange(int(xvalue)/100)

    rtm_accuracy = mydoc_rtm[0]['accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   
    rtm_f1= mydoc_rtm[0]['f1score']
    rtm_dtt = mydoc_rtm[0]['dtt']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1 = mydoc_rtmd[0]['f1score']
    rtmd_dtt = mydoc_rtmd[0]['dtt']


    fig_accuracy.add_trace(go.Scatter(x=x, y=rtm_accuracy, name='RTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=rtmd_accuracy, name='RTMD'))

    fig_precision.add_trace(go.Scatter(x=x, y=rtm_precision, name='RTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=rtmd_precision, name='RTMD'))

    fig_recall.add_trace(go.Scatter(x=x, y=rtm_recall, name='RTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=rtmd_recall, name='RTMD'))

    fig_f1.add_trace(go.Scatter(x=x, y=rtm_f1, name="RTM"))
    fig_f1.add_trace(go.Scatter(x=x, y=rtmd_f1, name="RTMD"))

    fig_dtt.add_trace(go.Scatter(x=x, y=rtm_dtt, name="RTM"))
    fig_dtt.add_trace(go.Scatter(x=x, y=rtmd_dtt, name="RTMD"))

st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)
st.plotly_chart(fig_f1, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)
st.plotly_chart(fig_dtt, use_container_width=True)

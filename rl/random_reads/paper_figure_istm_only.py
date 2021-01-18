'''
This file draws ISTM.
input files: mongo * 
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

    istmdcoll = db['istm_d']
    istmcoll = db['istm']

    return istmdcoll, istmcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct_istm(i, s,mvp, mbp, oap):
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
istmdcoll, istmcoll = connect()
data_load_state.text('Loading data...done!')
st.title("istm / istm-D")
#reconstruct multiselect options, match it with 
s =   st.multiselect("Total number of steps", [59999], default=[59999])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])
mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
oap =   st.multiselect("Outside attack probability", [0.1, 0.15, 0.2, 0.25, 0.3], default=[0.2])

rwcombo = reconstruct_istm(i, s, mvp, mbp, oap)

# allcombination.append(rwcombo)
# istmcombo = reconstruct_istm()
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
    mydoc_istm = list(istmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1, 'v_interval':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1, 'v_interval':1, 'dtt':1}))
    # print(mydoc_istm)
    # print(mydoc_istmd)
    x = np.arange(int(xvalue)/100)

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']   
    istm_f1= mydoc_istm[0]['f1score']
    istm_dtt = mydoc_istm[0]['dtt']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1 = mydoc_istmd[0]['f1score']
    istmd_dtt = mydoc_istmd[0]['dtt']


    fig_accuracy.add_trace(go.Scatter(x=x, y=istm_accuracy, name='istm'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=istmd_accuracy, name='istmD'))

    fig_precision.add_trace(go.Scatter(x=x, y=istm_precision, name='istm'))
    fig_precision.add_trace(go.Scatter(x=x, y=istmd_precision, name='istmD'))

    fig_recall.add_trace(go.Scatter(x=x, y=istm_recall, name='istm'))
    fig_recall.add_trace(go.Scatter(x=x, y=istmd_recall, name='istmD'))

    fig_f1.add_trace(go.Scatter(x=x, y=istm_f1, name="istm"))
    fig_f1.add_trace(go.Scatter(x=x, y=istmd_f1, name="istmD"))

    fig_dtt.add_trace(go.Scatter(x=x, y=istm_dtt, name="istm"))
    fig_dtt.add_trace(go.Scatter(x=x, y=istmd_dtt, name="istmD"))

st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)
st.plotly_chart(fig_f1, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)
st.plotly_chart(fig_dtt, use_container_width=True)

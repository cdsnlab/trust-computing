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

    dtmdcoll = db['dtm_d']
    dtmcoll = db['dtm']

    return dtmdcoll, dtmcoll

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
dtmdcoll, dtmcoll = connect()
data_load_state.text('Loading data...done!')
st.title("DTM / DTM-D")
#reconstruct multiselect options, match it with 
s =   st.multiselect("Total number of steps", [59999], default=[59999])
i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])
mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
oap =   st.multiselect("Outside attack probability", [0.1, 0.15, 0.2, 0.25, 0.3], default=[0.2])

rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

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
    mydoc_dtm = list(dtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1,  'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1, 'f1score':1,  'dtt':1}))

    x = np.arange(int(xvalue)/100)

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1= mydoc_dtm[0]['f1score']
    dtm_dtt = mydoc_dtm[0]['dtt']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1 = mydoc_dtmd[0]['f1score']
    dtmd_dtt = mydoc_dtmd[0]['dtt']


    fig_accuracy.add_trace(go.Scatter(x=x, y=dtm_accuracy, name='DTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=dtmd_accuracy, name='DTMD'))

    fig_precision.add_trace(go.Scatter(x=x, y=dtm_precision, name='DTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=dtmd_precision, name='DTMD'))

    fig_recall.add_trace(go.Scatter(x=x, y=dtm_recall, name='DTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=dtmd_recall, name='DTMD'))

    fig_f1.add_trace(go.Scatter(x=x, y=dtm_f1, name="DTM"))
    fig_f1.add_trace(go.Scatter(x=x, y=dtmd_f1, name="DTMD"))

    fig_dtt.add_trace(go.Scatter(x=x, y=dtm_dtt, name="DTM"))
    fig_dtt.add_trace(go.Scatter(x=x, y=dtmd_dtt, name="DTMD"))

st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)
st.plotly_chart(fig_f1, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)
st.plotly_chart(fig_dtt, use_container_width=True)

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

    cares_rl_sb = db['cares_rl_sb']
    cares_rl_bl = db['cares_rl_bl']

    rtmcoll = db['rtm']
    
    dtmcoll = db['dtm']
    istmcoll = db['istm']
    rtmdcoll = db['rtm-d']
    dtmdcoll = db['dtm-d']
    istmdcoll = db['istm-d']
    return cares_rl_sb, cares_rl_bl, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_bd=bd, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_interval=interval):
        allcombinations.append(str(output))
    return allcombinations

def reconstruct_rtm(s,mvp, mbp, oap, interval):
    allcombinations=[]
    for output in named_product(v_s=s, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_interval=interval):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')

            return temp[1]
            
    
data_load_state = st.text('Loading data...')
cares_rl_sb, cares_rl_bl, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll = connect()
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

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap, interval)
rwcombo = reconstruct_rtm(s, mvp, mbp, oap, interval)

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
# fig_dtt = go.FigureWidget(
#     layout=go.Layout(xaxis=dict(title="Time (s)"),yaxis=dict(title="DTT & GT", range=[0, 100], tickvals=list(range(0,100,10))))
#     )
# fig_cum_rew = go.FigureWidget(
#     layout=go.Layout(xaxis=dict(title="Time (s)"), yaxis=dict(title="cumulative reward"))
#     )
#! draw ours.
for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1}))
    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1}))

    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']

    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']


    x = np.arange(int(xvalue)/100)
    # print(x)
    fig_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, name="CARES-RL-SB"))
    fig_precision.add_trace(go.Scatter(x=x, y=rlsbprecision, name="CARES-RL-SB"))
    fig_recall.add_trace(go.Scatter(x=x, y=rlsbrecall, name="CARES-RL-SB"))

    fig_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, name="CARES-RL-BL"))
    fig_precision.add_trace(go.Scatter(x=x, y=rlblprecision, name="CARES-RL-BL"))
    fig_recall.add_trace(go.Scatter(x=x, y=rlblrecall, name="CARES-RL-BL"))

    # fig_dtt.add_trace(go.Scatter(x=x, y=avg_dtt, name="Average DTT value"))
    # fig_dtt.add_trace(go.Scatter(x=x, y=avg_gt, name="Average GT value"))
    # fig_cum_rew.add_trace(go.Scatter(x=x, y=cum_rew, name="Cumulative rewards"))

#! draw RW
for j in rwcombo:
    print(j)
    xvalue = getxvalue(j)
    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_istm = list(istmcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {'_id':0, 'accuracy':1, 'precision':1, 'recall':1}))

    rtm_accuracy = mydoc_rtm[0]['accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']

    x = np.arange(int(xvalue)/100)
    fig_accuracy.add_trace(go.Scatter(x=x, y=rtm_accuracy, name='RTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=dtm_accuracy, name='DTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=istm_accuracy, name='ISTM'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=rtmd_accuracy, name='RTMD'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=dtmd_accuracy, name='DTMD'))
    fig_accuracy.add_trace(go.Scatter(x=x, y=istmd_accuracy, name='ISTMD'))


    fig_precision.add_trace(go.Scatter(x=x, y=rtm_precision, name='RTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=dtm_precision, name='DTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=istm_precision, name='ISTM'))
    fig_precision.add_trace(go.Scatter(x=x, y=rtmd_precision, name='RTMD'))
    fig_precision.add_trace(go.Scatter(x=x, y=dtmd_precision, name='DTMD'))
    fig_precision.add_trace(go.Scatter(x=x, y=istmd_precision, name='ISTMD'))

    fig_recall.add_trace(go.Scatter(x=x, y=rtm_recall, name='RTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=dtm_recall, name='DTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=istm_recall, name='ISTM'))
    fig_recall.add_trace(go.Scatter(x=x, y=rtmd_recall, name='RTMD'))
    fig_recall.add_trace(go.Scatter(x=x, y=dtmd_recall, name='DTMD'))
    fig_recall.add_trace(go.Scatter(x=x, y=istmd_recall, name='ISTMD'))


st.plotly_chart(fig_accuracy, use_container_width=True)
st.plotly_chart(fig_precision, use_container_width=True)
st.plotly_chart(fig_recall, use_container_width=True)

# st.plotly_chart(fig_dtt, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)

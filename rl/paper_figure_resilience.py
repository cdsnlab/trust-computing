'''
This file draws the accuracy of detecting malicious vehicles with different schemes.
input files: CARES_RL*, RTM*, DTM*, ISTM* 
output diagram: Detection accuracy, precision, recall, f1
BAR graph.
'''

import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
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
    rtmdcoll = db['rtm_d']
    dtmdcoll = db['dtm_d']
    istmdcoll = db['istm_d']
    return cares_rl_sb, cares_rl_bl, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_bd=bd, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap):
        allcombinations.append(str(output))
    return allcombinations

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
cares_rl_sb, cares_rl_bl, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll = connect()
data_load_state.text('Loading data...done!')

d = [5]
bd=[0.5]
lr=[0.1]
df=[0.1]
eps=[0.5]
fd=[1]
s=[59999]
i=[50]
mvp=[0.2]
mbp=[0.5]
oap=[0.1, 0.15, 0.2, 0.25, 0.3]
#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap)
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)
st.title("OAP resilience graph")

fig_acc_oap = make_subplots(
    cols=2,
    rows=1,
    subplot_titles=("Detection Accuracy(%)", "F1 Score")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_acc_oap.update_yaxes(showgrid=True, gridcolor="black", range=[0, 100], mirror=True, showline=True, linecolor='black')
# fig_acc_oap.update_yaxes(showgrid=True, gridcolor="black",title_text="F1 Score", range=[0, 100], secondary_y=True)
fig_acc_oap.update_xaxes(showgrid=True, gridcolor="black", mirror=True, showline=True, linecolor='black')
fig_acc_oap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',height=400, width=600)
# fig_acc_oap.update_layout(height=600, width=600)

#! draw oap
rlsbacc=[]
rlblacc=[]
rlsbf11=[]
rlblf11=[]
for k in allcombination:
    print(k)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    # print(mydocrlsb[0])
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsboap = mydocrlsb[0]['v_oap']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlbloap = mydocrlbl[0]['v_oap']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    rlsbacc.append(rlsbaccuracy[-1])
    rlblacc.append(rlblaccuracy[-1])
    rlsbf11.append(rlsbf1[-1])
    rlblf11.append(rlblf1[-1])

#! Line chart
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rlsbacc, name="RLSB", marker=dict(size=12,symbol="x")))
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rlblacc, name="RLBL", marker=dict(size=12,symbol="x")))
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlsbf11, name="RLSB-f1", marker=dict(size=12,symbol="circle")),
    col=2,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlblf11, name="RLBL-f1", marker=dict(size=12,symbol="circle")),
    col=2,
    row=1
)

rtmacc=[]
rtmdacc=[]
dtmacc=[]
dtmdacc=[]
istmacc=[]
istmdacc=[]

rtmf11=[]
rtmdf11=[]
dtmf11=[]
dtmdf11=[]
istmf11=[]
istmdf11=[]

# #! draw RW
for j in rwcombo:
    print(j)
    # xvalue = getxvalue(j)
    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istm = list(istmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))

    rtm_accuracy = mydoc_rtm[0]['accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   
    rtm_f1score = mydoc_rtm[0]['f1score']

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1score = mydoc_dtm[0]['f1score']

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']
    istm_f1score = mydoc_istm[0]['f1score']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']

    rtmacc.append(rtm_accuracy[-1])
    rtmdacc.append(rtmd_accuracy[-1])
    dtmacc.append(dtm_accuracy[-1])
    dtmdacc.append(dtmd_accuracy[-1])
    istmacc.append(istm_accuracy[-1])
    istmdacc.append(istmd_accuracy[-1])

    rtmf11.append(rtm_f1score[-1])
    rtmdf11.append(rtmd_f1score[-1])
    dtmf11.append(dtm_f1score[-1])
    dtmdf11.append(dtmd_f1score[-1])
    istmf11.append(istm_f1score[-1])
    istmdf11.append(istmd_f1score[-1])

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmacc, name="RTM", marker=dict(size=12,symbol="x")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmdacc, name="RTMD", marker=dict(size=12,symbol="x")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmacc, name="DTM", marker=dict(size=12,symbol="x")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmdacc, name="DTMD", marker=dict(size=12,symbol="x")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmacc, name="ISTM", marker=dict(size=12,symbol="x")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmdacc, name="ISTMD", marker=dict(size=12,symbol="x")),col=1, row=1)

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmf11, name="RTM-f1",marker=dict(size=12,symbol="circle")) ,col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmdf11, name="RTMD-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmf11, name="DTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmdf11, name="DTMD-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmf11, name="ISTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmdf11, name="ISTMD-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)



st.plotly_chart(fig_acc_oap, use_container_width=True)

mvp=[0.1, 0.2, 0.3, 0.4]
oap=[0.2]
allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap)
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

st.title("MVP resilience graph")

fig_acc_mvp = make_subplots(
    specs=[[{"secondary_y": True}]]
)
fig_acc_mvp.update_yaxes(showgrid=True, gridcolor="black",title_text="Malicious Vehicle Detection Accuracy (%)", range=[0, 100], secondary_y=False)
fig_acc_mvp.update_xaxes(showgrid=True, gridcolor="black")

fig_acc_mvp.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')

#! draw mvp
rlsbacc=[]
rlblacc=[]
for k in allcombination:
    print(k)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    # print(mydocrlsb[0])
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsboap = mydocrlsb[0]['v_oap']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlbloap = mydocrlbl[0]['v_oap']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    rlsbacc.append(rlsbaccuracy[-1])
    rlblacc.append(rlblaccuracy[-1])
#! Line chart
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rlsbacc, name="RLSB", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rlblacc, name="RLBL", marker=dict(size=12,symbol="x")))

rtmacc=[]
rtmdacc=[]
dtmacc=[]
dtmdacc=[]
istmacc=[]
istmdacc=[]
# #! draw RW
for j in rwcombo:
    print(j)
    # xvalue = getxvalue(j)
    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istm = list(istmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))

    rtm_accuracy = mydoc_rtm[0]['accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   
    rtm_f1score = mydoc_rtm[0]['f1score']

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1score = mydoc_dtm[0]['f1score']

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']
    istm_f1score = mydoc_istm[0]['f1score']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']

    rtmacc.append(rtm_accuracy[-1])
    rtmdacc.append(rtmd_accuracy[-1])
    dtmacc.append(dtm_accuracy[-1])
    dtmdacc.append(dtmd_accuracy[-1])
    istmacc.append(istm_accuracy[-1])
    istmdacc.append(istmd_accuracy[-1])

fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmacc, name="RTM", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmdacc, name="RTMD", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmacc, name="DTM", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmdacc, name="DTMD", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmacc, name="ISTM", marker=dict(size=12,symbol="x")))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmdacc, name="ISTMD", marker=dict(size=12,symbol="x")))

st.plotly_chart(fig_acc_mvp, use_container_width=True)


mbp=[0.1, 0.2, 0.3, 0.4, 0.5]
mvp=[0.2]

allcombination = reconstruct(d, bd, lr, df, eps, fd, s, i, mvp, mbp, oap)
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

st.title("MBP resilience graph")

fig_acc_mbp = make_subplots(
    cols=2,
    rows=1,
    subplot_titles=("Detection Accuracy(%)", "F1 Score")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_acc_mbp.update_yaxes(showgrid=True, gridcolor="black", range=[0, 100], mirror=True, showline=True, linecolor='black')
fig_acc_mbp.update_xaxes(showgrid=True, gridcolor="black", mirror=True, showline=True, linecolor='black')
fig_acc_mbp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',height=400, width=600)


#! draw mvp
rlsbacc=[]
rlblacc=[]
rlsbf11=[]
rlblf11=[]
for k in allcombination:
    print(k)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    # print(mydocrlsb[0])
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsboap = mydocrlsb[0]['v_oap']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1, 'v_oap':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlbloap = mydocrlbl[0]['v_oap']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    rlsbacc.append(rlsbaccuracy[-1])
    rlblacc.append(rlblaccuracy[-1])
    rlsbf11.append(rtm_f1score[-1])
    rlblf11.append(rtmd_f1score[-1])
    
#! Line chart
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rlsbacc, name="RLSB", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rlblacc, name="RLBL", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlsbf11, name="RLSB-f1",marker=dict(size=12,symbol="circle")),
    col=2,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlblf11, name="RLBL-f1",marker=dict(size=12,symbol="circle")),
    col=2,
    row=1
)


rtmacc=[]
rtmdacc=[]
dtmacc=[]
dtmdacc=[]
istmacc=[]
istmdacc=[]

rtmf11=[]
rtmdf11=[]
dtmf11=[]
dtmdf11=[]
istmf11=[]
istmdf11=[]

# #! draw RW
for j in rwcombo:
    print(j)
    # xvalue = getxvalue(j)
    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istm = list(istmcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))

    rtm_accuracy = mydoc_rtm[0]['accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   
    rtm_f1score = mydoc_rtm[0]['f1score']

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1score = mydoc_dtm[0]['f1score']

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']
    istm_f1score = mydoc_istm[0]['f1score']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']

    rtmacc.append(rtm_accuracy[-1])
    rtmdacc.append(rtmd_accuracy[-1])
    dtmacc.append(dtm_accuracy[-1])
    dtmdacc.append(dtmd_accuracy[-1])
    istmacc.append(istm_accuracy[-1])
    istmdacc.append(istmd_accuracy[-1])

    rtmf11.append(rtm_f1score[-1])
    rtmdf11.append(rtmd_f1score[-1])
    dtmf11.append(dtm_f1score[-1])
    dtmdf11.append(dtmd_f1score[-1])
    istmf11.append(istm_f1score[-1])
    istmdf11.append(istmd_f1score[-1])

fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmacc, name="RTM", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmdacc, name="RTMD", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmacc, name="DTM", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmdacc, name="DTMD", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmacc, name="ISTM", marker=dict(size=12,symbol="x")))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmdacc, name="ISTMD", marker=dict(size=12,symbol="x")))

# print(rtmf11)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmf11, name="RTM-f1",marker=dict(size=12,symbol="circle")) ,col=2,
    row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmdf11, name="RTMD-f1",marker=dict(size=12,symbol="circle")),col=2,
    row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmf11, name="DTM-f1",marker=dict(size=12,symbol="circle")),col=2,
    row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmdf11, name="DTMD-f1",marker=dict(size=12,symbol="circle")),col=2,
    row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmf11, name="ISTM-f1",marker=dict(size=12,symbol="circle")),col=2,
    row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmdf11, name="ISTMD-f1",marker=dict(size=12,symbol="circle")),col=2,
    row=1)
st.plotly_chart(fig_acc_mbp, use_container_width=True)
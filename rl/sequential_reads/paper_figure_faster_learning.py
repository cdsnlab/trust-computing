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

    cares_rl_sb = db['cares_rl_sb95']
    cares_rl_bl = db['cares_rl_bl95']

    rtmcoll = db['rtm95']
    dtmcoll = db['dtm95']
    istmcoll = db['istm95']
    rtmdcoll = db['rtm_d95']
    dtmdcoll = db['dtm_d95']
    istmdcoll = db['istm_d95']
    return cares_rl_sb, cares_rl_bl, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap):
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
lr=[0.1]
df=[0.1]
eps=[0.5]
fd=[1]
s=[59999]
i=[50]
mvp=[0.2]
mbp=[0.5]
oap=[0.2]
#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s

st.title("Detection Accuracy! ")

fig_learn_speed = make_subplots(
    cols=1,
    rows=1,
    subplot_titles=("Detection Accuracy(%)", "DTT changes")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_learn_speed.update_yaxes(showgrid=True, gridcolor="black", range=[0, 100], mirror=True, showline=True, linecolor='black')
# fig_learn_speed.update_yaxes(showgrid=True, gridcolor="black",title_text="F1 Score", range=[0, 100], secondary_y=True)
fig_learn_speed.update_xaxes(showgrid=True, title_text="Interaction number",gridcolor="black", mirror=True, showline=True, linecolor='black')
fig_learn_speed.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',height=400, width=1000)
# fig_learn_speed.update_layout(height=600, width=600)

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)
for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    fig_learn_speed.add_trace(
        go.Scatter(x=x, y=rlsbaccuracy, name="CARES_SB", mode='lines', line=dict(color='yellow', width=2, dash='solid')),
        col=1,
        row=1,        
        )
    fig_learn_speed.add_trace(
        go.Scatter(x=x, y=rlblaccuracy, name="CARES_BL", mode='lines', line=dict(color='yellow', width=2, dash='dash')),
        col=1,
        row=1,
    )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlsbdtt, name="CARES_SB_DTT", line=dict(color='black', width=2, dash='solid'),showlegend=False),
    #     col=1,
    #     row=2
    # )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlbldtt, name="CARES_BL_DTT", line=dict(color='black', width=2, dash='dash'),showlegend=False),
    #     col=1,
    #     row=2
    # )
# i=[50]
# allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)
# for k in allcombination:
#     print(k)
#     xvalue = getxvalue(k)
#     x = np.arange(int(xvalue)/100)

#     myquery = {"id": str(k)}
    
#     mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
#     rlsbaccuracy = mydocrlsb[0]['accuracy']
#     rlsbprecision = mydocrlsb[0]['precision']
#     rlsbrecall = mydocrlsb[0]['recall']
#     rlsbf1 = mydocrlsb[0]['f1score']
#     rlsbrew = mydocrlsb[0]['cum_rew']
#     rlsbdtt = mydocrlsb[0]['avg_dtt']

#     mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
#     rlblaccuracy = mydocrlbl[0]['accuracy']
#     rlblprecision = mydocrlbl[0]['precision']
#     rlblrecall = mydocrlbl[0]['recall']
#     rlblf1 = mydocrlbl[0]['f1score']
#     rlblrew = mydocrlbl[0]['cum_rew']
#     rlbldtt = mydocrlbl[0]['avg_dtt']
#     #mvp: 0.1, 0.2, 0.3, 0.4
#     #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
#     #oap: 0.1, 0.15, 0.2, 0.25, 0.3

#     fig_learn_speed.add_trace(
#         go.Scatter(x=x, y=rlsbaccuracy, name="CARES_SB", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
#         col=1,
#         row=1,        
#         )
#     fig_learn_speed.add_trace(
#         go.Scatter(x=x, y=rlblaccuracy, name="CARES_BL", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
#         col=1,
#         row=1,
#     )
#     # fig_learn_speed.add_trace(
#     #     go.Scatter(x=x, y=rlsbdtt, name="CARES_SB_DTT", line=dict(color='black', width=2, dash='solid'),showlegend=False),
#     #     col=1,
#     #     row=2
#     # )
#     # fig_learn_speed.add_trace(
#     #     go.Scatter(x=x, y=rlbldtt, name="CARES_BL_DTT", line=dict(color='black', width=2, dash='dash'),showlegend=False),
#     #     col=1,
#     #     row=2
#     # )
# i=[90]
# allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)
# for k in allcombination:
#     print(k)
#     xvalue = getxvalue(k)
#     x = np.arange(int(xvalue)/100)

#     myquery = {"id": str(k)}
    
#     mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
#     rlsbaccuracy = mydocrlsb[0]['accuracy']
#     rlsbprecision = mydocrlsb[0]['precision']
#     rlsbrecall = mydocrlsb[0]['recall']
#     rlsbf1 = mydocrlsb[0]['f1score']
#     rlsbrew = mydocrlsb[0]['cum_rew']
#     rlsbdtt = mydocrlsb[0]['avg_dtt']

#     mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
#     rlblaccuracy = mydocrlbl[0]['accuracy']
#     rlblprecision = mydocrlbl[0]['precision']
#     rlblrecall = mydocrlbl[0]['recall']
#     rlblf1 = mydocrlbl[0]['f1score']
#     rlblrew = mydocrlbl[0]['cum_rew']
#     rlbldtt = mydocrlbl[0]['avg_dtt']
#     #mvp: 0.1, 0.2, 0.3, 0.4
#     #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
#     #oap: 0.1, 0.15, 0.2, 0.25, 0.3

#     fig_learn_speed.add_trace(
#         go.Scatter(x=x, y=rlsbaccuracy, name="CARES_SB", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
#         col=1,
#         row=1,        
#         )
#     fig_learn_speed.add_trace(
#         go.Scatter(x=x, y=rlblaccuracy, name="CARES_BL", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
#         col=1,
#         row=1,
#     )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlsbdtt, name="CARES_SB_DTT", line=dict(color='black', width=2, dash='solid'),showlegend=False),
    #     col=1,
    #     row=2
    # )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlbldtt, name="CARES_BL_DTT", line=dict(color='black', width=2, dash='dash'),showlegend=False),
    #     col=1,
    #     row=2
    # )

# fig_learn_speed.add_annotation(
# x=10, 
# y=10,
# ax=100,
# ay=0,
# xref='x',
# yref='y',
# axref='x',
# ayref='y',
# text="Initial DTT: 10",
# font=dict(
#     color="red",
#     size=16
# ),
# showarrow=True,
# arrowhead=1,
# col=1,
# row=2
# )


# fig_learn_speed.add_annotation(
# x=10, 
# y=50,
# ax=250,
# ay=40,
# xref='x',
# yref='y',
# axref='x',
# ayref='y',
# text="Initial DTT: 50",
# font=dict(
#     color="red",
#     size=16
# ),
# showarrow=True,
# arrowhead=1,
# col=1,
# row=2
# )

# fig_learn_speed.add_annotation(
# x=10, 
# y=90,
# ax=100,
# ay=0,
# xref='x',
# yref='y',
# axref='x',
# ayref='y',
# text="Initial DTT: 90",
# font=dict(
#     color="red",
#     size=16
# ),showarrow=True,
# arrowhead=1,
# col=1,
# row=2
# )

############################################

# #! draw RW
for j in rwcombo:
    print(j)
    xvalue = getxvalue(j)
    x = np.arange(int(xvalue)/100)

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
    rtm_dtt = mydoc_rtm[0]['dtt']

    dtm_accuracy = mydoc_dtm[0]['accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1score = mydoc_dtm[0]['f1score']
    dtm_dtt = mydoc_dtm[0]['dtt']

    istm_accuracy = mydoc_istm[0]['accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']
    istm_f1score = mydoc_istm[0]['f1score']
    istm_dtt = mydoc_istm[0]['dtt']

    rtmd_accuracy = mydoc_rtmd[0]['accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']
    rtmd_dtt = mydoc_rtmd[0]['dtt']

    dtmd_accuracy = mydoc_dtmd[0]['accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']
    dtmd_dtt = mydoc_dtmd[0]['dtt']

    istmd_accuracy = mydoc_istmd[0]['accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']
    istmd_dtt = mydoc_istmd[0]['dtt']

    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtm_accuracy, name="RTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmd_accuracy, name="RTMD", ),col=1, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtm_accuracy, name="DTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmd_accuracy, name="DTMD", ),col=1, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istm_accuracy, name="ISTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=istmd_accuracy, name="ISTMD", ),col=1, row=1)

    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmf11, name="RTM-f1",marker=dict(size=12,symbol="circle")) ,col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmd_dtt, name="RTMD_DTT",),col=1, row=2)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmf11, name="DTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmd_dtt, name="DTMD_DTT",),col=1, row=2)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istmf11, name="ISTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istmd_dtt, name="ISTMD_DTT",),col=1, row=2)

st.plotly_chart(fig_learn_speed, use_container_width=True)

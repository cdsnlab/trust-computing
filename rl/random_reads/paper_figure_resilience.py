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
from collections import defaultdict


import faulthandler
faulthandler.enable()

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']

    # cares_rl_sb = db['cares_rl_sb_de001halfsteps']
    # cares_rl_bl = db['cares_rl_bl_de001halfsteps']
    
    cares_rl_sb = db['cares_rl_sb_de001halfsteps']
    cares_rl_bl = db['cares_rl_bl_de001halfsteps']

    rtmcoll = db['rtm95_r_vs']
    dtmcoll = db['dtm95_r_vs']
    istmcoll = db['istm95_r_vs']
    rtmdcoll = db['rtm_d95_r_vs']
    dtmdcoll = db['dtm_d95_r_vs']
    istmdcoll = db['istm_d95_r_vs']
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
# s=[59999]
s=[12000]
i=[10]
mvp=[0.2]
mbp=[0.9]
oap=[0.1, 0.2, 0.3]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

fig_acc_oap = make_subplots(
    cols=4,
    rows=1,
    subplot_titles=(" (a) Detection Accuracy ", " (b) Precision", " (c) Recall", " (d) F1 Score"),
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_acc_oap.update_yaxes(
    showgrid=True, 
    linewidth=2,
    showline=True, 
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True, 
    linecolor='black', 
    title_standoff=1,
)
fig_acc_oap.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    # title_text="OA probability.",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

)
fig_acc_oap.update_layout(
    title=dict(
        text = 'P<sub>a</sub>',
        xanchor='center',
        yanchor='bottom',
        x=0.45,
        y=0.05,

    ),
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=400, 
    width=1200,
    font=dict(
        size=18,
    ), 
)
fig_acc_oap.layout.annotations[0].update(y=1.1, font=dict(size=20))
fig_acc_oap.layout.annotations[1].update(y=1.1, font=dict(size=20))
fig_acc_oap.layout.annotations[2].update(y=1.1, font=dict(size=20))
fig_acc_oap.layout.annotations[3].update(y=1.1, font=dict(size=20))
# print(fig_acc_oap)

#! draw oap

rlsb=defaultdict(list)
rlbl=defaultdict(list)

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
    rlsb['acc'].append(rlsbaccuracy[-1])
    rlsb['pre'].append(rlsbprecision[-1])
    rlsb['rec'].append(rlsbrecall[-1])
    rlsb['f1'].append(rlsbf1[-1])
    
    rlbl['acc'].append(rlblaccuracy[-1])
    rlbl['pre'].append(rlblf1[-1])
    rlbl['rec'].append(rlblaccuracy[-1])
    rlbl['f1'].append(rlblf1[-1])
    

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rlsb['acc'], name="CARES-S", mode='lines', line=dict(color='black', width=2, dash='solid')))
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rlbl['acc'], name="CARES-B", mode='lines', line=dict(color='black', width=2, dash='dash')))

#TODO missing precision, recall in the middle
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlsb['pre'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=2,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlbl['pre'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=2,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlsb['rec'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=3,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlbl['rec'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=3,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlsb['f1'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=4,
    row=1
)
fig_acc_oap.add_trace(
    go.Scatter(x=oap, y=rlbl['f1'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=4,
    row=1
)


rtm=defaultdict(list)
rtmd=defaultdict(list)
dtm=defaultdict(list)
dtmd=defaultdict(list)
istm=defaultdict(list)
istmd=defaultdict(list)

rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

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

    rtm['acc'].append(rtm_accuracy[-1])
    rtm['pre'].append(rtm_precision[-1])
    rtm['rec'].append(rtm_recall[-1])
    rtm['f1'].append(rtm_f1score[-1])

    rtmd['acc'].append(rtmd_accuracy[-1])
    rtmd['pre'].append(rtmd_precision[-1])
    rtmd['rec'].append(rtmd_recall[-1])
    rtmd['f1'].append(rtmd_f1score[-1])

    dtm['acc'].append(dtm_accuracy[-1])
    dtm['pre'].append(dtm_precision[-1])
    dtm['rec'].append(dtm_recall[-1])
    dtm['f1'].append(dtm_f1score[-1])

    dtmd['acc'].append(dtmd_accuracy[-1])
    dtmd['pre'].append(dtmd_precision[-1])
    dtmd['rec'].append(dtmd_recall[-1])
    dtmd['f1'].append(dtmd_f1score[-1])

    istm['acc'].append(istm_accuracy[-1])
    istm['pre'].append(istm_precision[-1])
    istm['rec'].append(istm_recall[-1])
    istm['f1'].append(istm_f1score[-1])

    istmd['acc'].append(istmd_accuracy[-1])
    istmd['pre'].append(istmd_precision[-1])
    istmd['rec'].append(istmd_recall[-1])
    istmd['f1'].append(istmd_f1score[-1])


fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtm['acc'], name="RTM", mode='lines', line=dict(color='green', width=2, dash="solid") ),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmd['acc'], name="RTMD", mode='lines', line=dict(color='green', width=2, dash="dash")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtm['acc'], name="DTM", mode='lines', line=dict(color='orange', width=2, dash="solid")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmd['acc'], name="DTMD", mode='lines', line=dict(color='orange', width=2, dash="dash")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istm['acc'], name="ISTM", mode='lines', line=dict(color='blue', width=2, dash="solid")),col=1, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmd['acc'], name="ISTMD", mode='lines', line=dict(color='blue', width=2, dash="dash")),col=1, row=1)

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtm['pre'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmd['pre'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtm['pre'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmd['pre'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istm['pre'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmd['pre'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=2, row=1)

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtm['rec'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=3, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmd['rec'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtm['rec'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmd['rec'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istm['rec'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmd['rec'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=3, row=1)

fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtm['f1'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=4, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=rtmd['f1'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtm['f1'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=dtmd['f1'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istm['f1'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_oap.add_trace(go.Scatter(x=oap, y=istmd['f1'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=4, row=1)

st.title("Outside attack graph")
st.plotly_chart(fig_acc_oap, use_container_width=False, filename='latex')

######################################################MVP
######################################################
######################################################

d = [5]
lr=[0.1]
df=[0.1]
eps=[0.5]
fd=[1]
# s=[59999]
s=[12000]
i=[10]
mvp=[0.1, 0.2, 0.3, 0.4]
mbp=[0.9]
oap=[0.3]
allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)


fig_acc_mvp = make_subplots(
    cols=4,
    rows=1,
    subplot_titles=(" (a) Detection Accuracy ", " (b) Precision", " (c) Recall", " (d) F1 Score")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_acc_mvp.update_yaxes(
    showgrid=True, 
    linewidth=2,
    showline=True, 
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True, 
    linecolor='black', 
    title_standoff=1,
)
fig_acc_mvp.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    # title_text="MV Proportion.",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

)
fig_acc_mvp.update_layout( 
    title=dict(
        text = 'P<sub>v<sub>m</sub></sub>',
        xanchor='center',
        yanchor='bottom',
        x=0.45,
        y=0.05,

    ),
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=400, 
    width=1200,
    font=dict(
        size=18,
    ), 
)
fig_acc_mvp.layout.annotations[0].update(y=1.1, font=dict(size=20))
fig_acc_mvp.layout.annotations[1].update(y=1.1, font=dict(size=20))
fig_acc_mvp.layout.annotations[2].update(y=1.1, font=dict(size=20))
fig_acc_mvp.layout.annotations[3].update(y=1.1, font=dict(size=20))
#! draw mvp
rlsb=defaultdict(list)
rlbl=defaultdict(list)
for k in allcombination:
    print(k)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'f1score':1, 'cum_rew':1, 'avg_dtt':1}))
    # print(mydocrlsb[0])
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

    rlsb['acc'].append(rlsbaccuracy[-1])
    rlsb['pre'].append(rlsbprecision[-1])
    rlsb['rec'].append(rlsbrecall[-1])
    rlsb['f1'].append(rlsbf1[-1])
    
    rlbl['acc'].append(rlblaccuracy[-1])
    rlbl['pre'].append(rlblf1[-1])
    rlbl['rec'].append(rlblaccuracy[-1])
    rlbl['f1'].append(rlblf1[-1])

#! Line chart
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rlsb['acc'], name="CARES-S", mode='lines', line=dict(color='black', width=2, dash='solid')))
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rlbl['acc'], name="CARES-B", mode='lines', line=dict(color='black', width=2, dash='dash')))

#TODO missing precision, recall in the middle
fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlsb['pre'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=2,
    row=1
)
fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlbl['pre'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=2,
    row=1
)
fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlsb['rec'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=3,
    row=1
)
fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlbl['rec'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=3,
    row=1
)

fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlsb['f1'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=4,
    row=1
)
fig_acc_mvp.add_trace(
    go.Scatter(x=mvp, y=rlbl['f1'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=4,
    row=1
)

rtm=defaultdict(list)
rtmd=defaultdict(list)
dtm=defaultdict(list)
dtmd=defaultdict(list)
istm=defaultdict(list)
istmd=defaultdict(list)

rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)
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

    rtm['acc'].append(rtm_accuracy[-1])
    rtm['pre'].append(rtm_precision[-1])
    rtm['rec'].append(rtm_recall[-1])
    rtm['f1'].append(rtm_f1score[-1])

    rtmd['acc'].append(rtmd_accuracy[-1])
    rtmd['pre'].append(rtmd_precision[-1])
    rtmd['rec'].append(rtmd_recall[-1])
    rtmd['f1'].append(rtmd_f1score[-1])

    dtm['acc'].append(dtm_accuracy[-1])
    dtm['pre'].append(dtm_precision[-1])
    dtm['rec'].append(dtm_recall[-1])
    dtm['f1'].append(dtm_f1score[-1])

    dtmd['acc'].append(dtmd_accuracy[-1])
    dtmd['pre'].append(dtmd_precision[-1])
    dtmd['rec'].append(dtmd_recall[-1])
    dtmd['f1'].append(dtmd_f1score[-1])

    istm['acc'].append(istm_accuracy[-1])
    istm['pre'].append(istm_precision[-1])
    istm['rec'].append(istm_recall[-1])
    istm['f1'].append(istm_f1score[-1])

    istmd['acc'].append(istmd_accuracy[-1])
    istmd['pre'].append(istmd_precision[-1])
    istmd['rec'].append(istmd_recall[-1])
    istmd['f1'].append(istmd_f1score[-1])


fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtm['acc'], name="RTM", mode='lines', line=dict(color='green', width=2, dash="solid") ),col=1, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmd['acc'], name="RTMD", mode='lines', line=dict(color='green', width=2, dash="dash")),col=1, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtm['acc'], name="DTM", mode='lines', line=dict(color='orange', width=2, dash="solid")),col=1, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmd['acc'], name="DTMD", mode='lines', line=dict(color='orange', width=2, dash="dash")),col=1, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istm['acc'], name="ISTM", mode='lines', line=dict(color='blue', width=2, dash="solid")),col=1, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmd['acc'], name="ISTMD", mode='lines', line=dict(color='blue', width=2, dash="dash")),col=1, row=1)
#TODO missing precision, recall in the middle
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtm['pre'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=2, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmd['pre'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtm['pre'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmd['pre'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istm['pre'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmd['pre'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=2, row=1)

fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtm['rec'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=3, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmd['rec'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtm['rec'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmd['rec'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istm['rec'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmd['rec'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=3, row=1)

fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtm['f1'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=4, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=rtmd['f1'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtm['f1'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=dtmd['f1'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istm['f1'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_mvp.add_trace(go.Scatter(x=mvp, y=istmd['f1'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=4, row=1)

st.title("Inside Attack graph")
st.plotly_chart(fig_acc_mvp, use_container_width=False)
# fig_acc_mvp.write_image('here.png')

######################################################MBP
######################################################
######################################################
d = [5]
lr=[0.1]
df=[0.1]
eps=[0.5]
fd=[1]
# s=[59999]
s=[12000]
i=[10]
mvp=[0.2]
mbp=[0.1, 0.3, 0.5, 0.7, 0.9]
oap=[0.1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

fig_acc_mbp = make_subplots(
    cols=4,
    rows=1,
    subplot_titles=(" (a) Detection Accuracy ", " (b) Precision", " (c) Recall", " (d) F1 Score")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_acc_mbp.update_yaxes(
    showgrid=True, 
    linewidth=2,
    showline=True, 
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True, 
    linecolor='black', 
    title_standoff=1,
)
fig_acc_mbp.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    # title_text="MB Prob.",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

)
fig_acc_mbp.update_layout(
    title=dict(
        text = 'P<sub>d<sub>m</sub></sub>',
        xanchor='center',
        yanchor='bottom',
        x=0.45,
        y=0.05,

    ),
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=400, 
    width=1200,
    font=dict(
        size=18,
    ), 
)
fig_acc_mbp.layout.annotations[0].update(y=1.1, font=dict(size=20))
fig_acc_mbp.layout.annotations[1].update(y=1.1, font=dict(size=20))
fig_acc_mbp.layout.annotations[2].update(y=1.1, font=dict(size=20))
fig_acc_mbp.layout.annotations[3].update(y=1.1, font=dict(size=20))
#! draw mbp
rlsb=defaultdict(list)
rlbl=defaultdict(list)
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

    rlsb['acc'].append(rlsbaccuracy[-1])
    rlsb['pre'].append(rlsbprecision[-1])
    rlsb['rec'].append(rlsbrecall[-1])
    rlsb['f1'].append(rlsbf1[-1])
    
    rlbl['acc'].append(rlblaccuracy[-1])
    rlbl['pre'].append(rlblf1[-1])
    rlbl['rec'].append(rlblaccuracy[-1])
    rlbl['f1'].append(rlblf1[-1])
    

fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rlsb['acc'], name="CARES-S", mode='lines', line=dict(color='black', width=2, dash='solid')))
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rlbl['acc'], name="CARES-B", mode='lines', line=dict(color='black', width=2, dash='dash')))

fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlsb['pre'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=2,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlbl['pre'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=2,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlsb['rec'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=3,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlbl['rec'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=3,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlsb['f1'], name="RLSB-f1", mode='lines', line=dict(color='black', width=2, dash='solid'), showlegend=False),
    col=4,
    row=1
)
fig_acc_mbp.add_trace(
    go.Scatter(x=mbp, y=rlbl['f1'], name="RLBL-f1", mode='lines', line=dict(color='black', width=2, dash='dash'), showlegend=False),
    col=4,
    row=1
)

# #! draw RW
rtm=defaultdict(list)
rtmd=defaultdict(list)
dtm=defaultdict(list)
dtmd=defaultdict(list)
istm=defaultdict(list)
istmd=defaultdict(list)

# s=[59999]
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap)

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

    rtm['acc'].append(rtm_accuracy[-1])
    rtm['pre'].append(rtm_precision[-1])
    rtm['rec'].append(rtm_recall[-1])
    rtm['f1'].append(rtm_f1score[-1])

    rtmd['acc'].append(rtmd_accuracy[-1])
    rtmd['pre'].append(rtmd_precision[-1])
    rtmd['rec'].append(rtmd_recall[-1])
    rtmd['f1'].append(rtmd_f1score[-1])

    dtm['acc'].append(dtm_accuracy[-1])
    dtm['pre'].append(dtm_precision[-1])
    dtm['rec'].append(dtm_recall[-1])
    dtm['f1'].append(dtm_f1score[-1])

    dtmd['acc'].append(dtmd_accuracy[-1])
    dtmd['pre'].append(dtmd_precision[-1])
    dtmd['rec'].append(dtmd_recall[-1])
    dtmd['f1'].append(dtmd_f1score[-1])

    istm['acc'].append(istm_accuracy[-1])
    istm['pre'].append(istm_precision[-1])
    istm['rec'].append(istm_recall[-1])
    istm['f1'].append(istm_f1score[-1])

    istmd['acc'].append(istmd_accuracy[-1])
    istmd['pre'].append(istmd_precision[-1])
    istmd['rec'].append(istmd_recall[-1])
    istmd['f1'].append(istmd_f1score[-1])
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtm['acc'], name="RTM", mode='lines', line=dict(color='green', width=2, dash="solid") ),col=1, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmd['acc'], name="RTMD", mode='lines', line=dict(color='green', width=2, dash="dash")),col=1, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtm['acc'], name="DTM", mode='lines', line=dict(color='orange', width=2, dash="solid")),col=1, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmd['acc'], name="DTMD", mode='lines', line=dict(color='orange', width=2, dash="dash")),col=1, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istm['acc'], name="ISTM", mode='lines', line=dict(color='blue', width=2, dash="solid")),col=1, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmd['acc'], name="ISTMD", mode='lines', line=dict(color='blue', width=2, dash="dash")),col=1, row=1)

fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtm['pre'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=2, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmd['pre'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtm['pre'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmd['pre'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=2, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istm['pre'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=2, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmd['pre'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=2, row=1)

fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtm['rec'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=3, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmd['rec'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtm['rec'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmd['rec'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=3, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istm['rec'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=3, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmd['rec'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=3, row=1)

fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtm['f1'], name="RTM-f1",mode='lines', line=dict(color='green', width=2, dash="solid"), showlegend=False) ,col=4, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=rtmd['f1'], name="RTMD-f1",mode='lines', line=dict(color='green', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtm['f1'], name="DTM-f1",mode='lines', line=dict(color='orange', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=dtmd['f1'], name="DTMD-f1",mode='lines', line=dict(color='orange', width=2, dash="dash"), showlegend=False),col=4, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istm['f1'], name="ISTM-f1",mode='lines', line=dict(color='blue', width=2, dash="solid"), showlegend=False),col=4, row=1)
fig_acc_mbp.add_trace(go.Scatter(x=mbp, y=istmd['f1'], name="ISTMD-f1",mode='lines', line=dict(color='blue', width=2, dash="dash"), showlegend=False),col=4, row=1)

st.title("MBP graph")
st.plotly_chart(fig_acc_mbp, use_container_width=False)


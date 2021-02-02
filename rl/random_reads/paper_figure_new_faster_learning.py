'''
This file draws the step_accuracy of detecting malicious vehicles with different schemes.
input files: CARES_RL*, RTM*, DTM*, ISTM* 
output diagram: Detection accuracy 

'''

import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product, starmap
from collections import namedtuple
from pymongo import MongoClient

import faulthandler


def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']

    cares_rl_sb = db['cares_rl_sb_newdeltas'] #! newdeltas works well 
    cares_rl_bl = db['cares_rl_bl_newdeltas']

    cares_rl_sb_custom=db['cares_rl_sb_custom_manual']
    cares_rl_bl_custom=db['cares_rl_bl_custom_manual']

    rtmcoll = db['rtm_pnt']
    dtmcoll = db['dtm_pnt']
    istmcoll = db['istm_pnt']
    rtmdcoll = db['rtm_d_50iter'] #! 50iter works well 
    dtmdcoll = db['dtm_d_50iter'] #!
    istmdcoll = db['istm_d_50iter']

    return cares_rl_sb, cares_rl_bl, cares_rl_sb_custom, cares_rl_bl_custom, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv): 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_ppvnpvthr=ppvnpv):
        allcombinations.append(str(output))
    return allcombinations

def reconstruct_rtm(i, s,mvp, mbp, oap, ppvnpv):
    allcombinations=[]
    for output in named_product(v_i = i, v_s=s, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_ppvnpvthr=ppvnpv):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')
            return temp[1]
            
data_load_state = st.text('Loading data...')
cares_rl_sb, cares_rl_bl, cares_rl_sb_custom, cares_rl_bl_custom, rtmcoll, dtmcoll, istmcoll, rtmdcoll, dtmdcoll, istmdcoll = connect()
data_load_state.text('Loading data...done!')

d = [10]
lr=[0.1]
df=[0.1]
eps=[0.1]
fd=[1]
s=[12000]
i=[10]
mvp=[0.1]
mbp=[1.0]
oap=[0.3] 
ppvnpv=[0.5]

markertypes=['circle', 'square', 'diamond'] # RTMD, DTMD, ISTMD
colortypes=['green', 'orange', 'blue']

color = ['red', 'blue', 'green']

# st.title("Detection Accuracy! ")

############# oap accuracy
fig_oap_acc = make_subplots(
    cols=1,
    rows=1,
    column_titles=(["Accuracy (%)"]),

)
fig_oap_acc.update_yaxes(
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
fig_oap_acc.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    # title_text="P<sub>a</sub>",
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=1,
)
fig_oap_acc.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=500, 
    width=500,
    font=dict(
        size=28,
    ), 
    margin=dict(
        l=5,
        r=5,
        b=30,
        t=50,
        pad=4
    ),
)

############ oap precision
fig_oap_pre = make_subplots(
    cols=1,
    rows=1,
    column_titles=(["Precision (%)"])

)
fig_oap_pre.update_yaxes(
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
fig_oap_pre.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=1,

)
fig_oap_pre.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=500, 
    width=500,
    font=dict(
        size=28,
    ), 
    margin=dict(
        l=5,
        r=5,
        b=30,
        t=50,
        pad=4
    ),
)

###################oap recall
fig_oap_rec = make_subplots(
    cols=1,
    rows=1,
    column_titles=(["Recall (%)"])

)
fig_oap_rec.update_yaxes(
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
fig_oap_rec.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=1,

)
fig_oap_rec.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=500, 
    width=500,
    font=dict(
        size=28,
    ), 
    margin=dict(
        l=5,
        r=5,
        b=30,
        t=50,
        pad=4
    ),
)
###################oap f1
fig_oap_f1 = make_subplots(
    cols=1,
    rows=1,
    column_titles=(["F1 Score (%)"])

)
fig_oap_f1.update_yaxes(
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
fig_oap_f1.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    mirror=True, 
    linecolor='black',
    title_standoff=1,

)
fig_oap_f1.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    height=500, 
    width=500,
    font=dict(
        size=28,
    ), 
    margin=dict(
        l=5,
        r=5,
        b=30,
        t=50,
        pad=4
    ),
)
fig_oap_acc.layout.annotations[0].update(y=1.03,font=dict(size=32))
fig_oap_pre.layout.annotations[0].update(y=1.03,font=dict(size=32))
fig_oap_rec.layout.annotations[0].update(y=1.03,font=dict(size=32))
fig_oap_f1.layout.annotations[0].update(y=1.03,font=dict(size=32))

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)
for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000
    x = np.arange(int(xvalue)/100)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlsbaccuracy = mydocrlsb[0]['cum_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['cum_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    fig_oap_acc.add_trace(
        go.Scatter(x=x, y=rlsbaccuracy, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
        col=1,
        row=1,        
        )
    fig_oap_acc.add_trace(
        go.Scatter(x=x, y=rlblaccuracy, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
        col=1,
        row=1,
    )
    fig_oap_pre.add_trace(
        go.Scatter(x=x, y=rlsbprecision, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
        col=1,
        row=1,        
    )
    fig_oap_pre.add_trace(
        go.Scatter(x=x, y=rlblprecision, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
        col=1,
        row=1,
    )
    fig_oap_rec.add_trace(
        go.Scatter(x=x, y=rlsbrecall, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
        col=1,
        row=1,        
        )
    fig_oap_rec.add_trace(
        go.Scatter(x=x, y=rlblrecall, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
        col=1,
        row=1,
    )
    fig_oap_f1.add_trace(
        go.Scatter(x=x, y=rlsbf1, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
        col=1,
        row=1,        
        )
    fig_oap_f1.add_trace(
        go.Scatter(x=x, y=rlblf1, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
        col=1,
        row=1,
    )

d=[1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000

    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb_custom.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
    # rlsbaccuracy = mydocrlsb[0]['cum_accuracy']
    # rlsbprecision = mydocrlsb[0]['precision']
    # rlsbrecall = mydocrlsb[0]['recall']
    # rlsbrew = mydocrlsb[0]['cum_rew']
    # rlsbdtt = mydocrlsb[0]['avg_dtt']
    # rlsbf1 = mydocrlsb[0]['f1score']
    # rlsberror = mydocrlsb[0]['error']

    # mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # rlblaccuracy = mydocrlbl[0]['cum_accuracy']
    # rlblprecision = mydocrlbl[0]['precision']
    # rlblrecall = mydocrlbl[0]['recall']
    # rlblrew = mydocrlbl[0]['cum_rew']
    # rlbldtt = mydocrlbl[0]['avg_dtt']
    # rlblf1 = mydocrlbl[0]['f1score']
    # rlblerror = mydocrlbl[0]['error']

    # fig_oap_acc.add_trace(
    #     go.Scatter(x=x, y=rlsbaccuracy, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    #     )
    # fig_oap_acc.add_trace(
    #     go.Scatter(x=x, y=rlblaccuracy, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
    #     col=1,
    #     row=1,
    # )
    # fig_oap_pre.add_trace(
    #     go.Scatter(x=x, y=rlsbprecision, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    # )
    # fig_oap_pre.add_trace(
    #     go.Scatter(x=x, y=rlblprecision, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
    #     col=1,
    #     row=1,
    # )
    # fig_oap_rec.add_trace(
    #     go.Scatter(x=x, y=rlsbrecall, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    #     )
    # fig_oap_rec.add_trace(
    #     go.Scatter(x=x, y=rlblrecall, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
    #     col=1,
    #     row=1,
    # )
    # fig_oap_f1.add_trace(
    #     go.Scatter(x=x, y=rlsbf1, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    #     )
    # fig_oap_f1.add_trace(
    #     go.Scatter(x=x, y=rlblf1, name="CARES-B (5)",marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
    #     col=1,
    #     row=1,
    # )




    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlsbaccuracy, name="CARES-S (5)", marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    #     )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlblaccuracy, marker=dict(color='red', size=20, symbol='x'), line=dict(color='red', width=4, dash='dash'), showlegend=False),
    #     col=1,
    #     row=1,
    # )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=[t / 100 for t in rlsbdtt], name="CARES-S", line=dict(color='black', width=4, dash='solid'),showlegend=False),
    #     col=2,
    #     row=1
    # )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=[t / 100 for t in rlbldtt], name="CARES-B", line=dict(color='black', width=4, dash='dot'),showlegend=False),
    #     col=2,
    #     row=1
    # )
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="SB-variable {}".format(i[idx])))
    # # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="BL-variable {}".format(i[idx])))

    # sb_results.add_trace(go.Scatter(x=x, y=[t / 100 for t in rlsbdtt], line=dict(width=4, color=color[idx],dash=linetype[idx]), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=2) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 

    # sb_results.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=3) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 


############################################
# mbp=[1.0]
# mvp=[0.4]


# mvp=[0.2]
# mbp=[1.0]
# oap=[0.3] 
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap, ppvnpv)

# #! draw RW
for idx, j in enumerate(rwcombo):
    print(j)
    xvalue = getxvalue(j)
    xvalue=12000

    x = np.arange(int(xvalue)/100)

    myquery = {'id': str(j)}
    # mydoc_rtm = list(rtmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    # mydoc_dtm = list(dtmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    # mydoc_istm = list(istmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))

    # rtm_accuracy = mydoc_rtm[0]['cum_accuracy']
    # rtm_precision = mydoc_rtm[0]['precision']
    # rtm_recall = mydoc_rtm[0]['recall']   
    # rtm_f1score = mydoc_rtm[0]['f1score']
    # rtm_dtt = mydoc_rtm[0]['dtt']

    # dtm_accuracy = mydoc_dtm[0]['cum_accuracy']
    # dtm_precision = mydoc_dtm[0]['precision']
    # dtm_recall = mydoc_dtm[0]['recall']   
    # dtm_f1score = mydoc_dtm[0]['f1score']
    # dtm_dtt = mydoc_dtm[0]['dtt']

    # istm_accuracy = mydoc_istm[0]['cum_accuracy']
    # istm_precision = mydoc_istm[0]['precision']
    # istm_recall = mydoc_istm[0]['recall']
    # istm_f1score = mydoc_istm[0]['f1score']
    # istm_dtt = mydoc_istm[0]['dtt']

    rtmd_accuracy = mydoc_rtmd[0]['cum_accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']
    rtmd_dtt = mydoc_rtmd[0]['dtt']

    dtmd_accuracy = mydoc_dtmd[0]['cum_accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']
    dtmd_dtt = mydoc_dtmd[0]['dtt']

    istmd_accuracy = mydoc_istmd[0]['cum_accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']
    istmd_dtt = mydoc_istmd[0]['dtt']

    # fig_oap_acc.add_trace(go.Scatter(x=x, y=rtm_accuracy, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    fig_oap_acc.add_trace(go.Scatter(x=x, y=rtmd_accuracy, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)

    # fig_oap_acc.add_trace(go.Scatter(x=x, y=dtm_accuracy, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_acc.add_trace(go.Scatter(x=x, y=dtmd_accuracy, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)

    # fig_oap_acc.add_trace(go.Scatter(x=x, y=istm_accuracy, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_acc.add_trace(go.Scatter(x=x, y=istmd_accuracy, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)

    # fig_oap_pre.add_trace(go.Scatter(x=x, y=rtm_precision, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    fig_oap_pre.add_trace(go.Scatter(x=x, y=rtmd_precision, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    # fig_oap_pre.add_trace(go.Scatter(x=x, y=dtm_precision, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_pre.add_trace(go.Scatter(x=x, y=dtmd_precision, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    # fig_oap_pre.add_trace(go.Scatter(x=x, y=istm_precision, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_pre.add_trace(go.Scatter(x=x, y=istmd_precision, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)

    # fig_oap_rec.add_trace(go.Scatter(x=x, y=rtm_recall, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    fig_oap_rec.add_trace(go.Scatter(x=x, y=rtmd_recall, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    # fig_oap_rec.add_trace(go.Scatter(x=x, y=dtm_recall, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_rec.add_trace(go.Scatter(x=x, y=dtmd_recall, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    # fig_oap_rec.add_trace(go.Scatter(x=x, y=istm_recall, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_rec.add_trace(go.Scatter(x=x, y=istmd_recall, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)

    # fig_oap_f1.add_trace(go.Scatter(x=x, y=rtm_f1score, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    fig_oap_f1.add_trace(go.Scatter(x=x, y=rtmd_f1score, name="RTMD", marker=dict(color=colortypes[0], size=20, symbol=markertypes[0]), line=dict(color=colortypes[0], width=4, dash="solid"), showlegend=False ),col=1, row=1)
    # fig_oap_f1.add_trace(go.Scatter(x=x, y=dtm_f1score, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_f1.add_trace(go.Scatter(x=x, y=dtmd_f1score, name="DTMD", marker=dict(color=colortypes[1], size=20, symbol=markertypes[1]), line=dict(color=colortypes[1], width=4, dash="solid"), showlegend=False),col=1, row=1)
    # fig_oap_f1.add_trace(go.Scatter(x=x, y=istm_f1score, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)
    fig_oap_f1.add_trace(go.Scatter(x=x, y=istmd_f1score, name="ISTMD", marker=dict(color=colortypes[2], size=20, symbol=markertypes[2]), line=dict(color=colortypes[2], width=4, dash="solid"), showlegend=False),col=1, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmf11, name="RTM-f1",marker=dict(size=12,symbol="circle")) ,col=1, row=2)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in rtmd_dtt], name="RTMD", line=dict(color=color[0], width=4,),showlegend=False),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmf11, name="DTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in dtmd_dtt], name="DTMD", line=dict(color=color[1], width=4,),showlegend=False),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istmf11, name="ISTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in istmd_dtt], name="ISTMD",line=dict(color=color[2], width=4,),showlegend=False),col=2, row=1)

st.plotly_chart(fig_oap_acc, use_container_width=False)
st.plotly_chart(fig_oap_pre, use_container_width=False)
st.plotly_chart(fig_oap_rec, use_container_width=False)
st.plotly_chart(fig_oap_f1, use_container_width=False)

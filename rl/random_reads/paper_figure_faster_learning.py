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
faulthandler.enable()

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']

    cares_rl_sb = db['cares_rl_sb_pnt']
    cares_rl_bl = db['cares_rl_bl_pnt']

    cares_rl_sb_custom=db['cares_rl_sb_custom_pnt']
    cares_rl_bl_custom=db['cares_rl_bl_custom_pnt']

    rtmcoll = db['rtm_pnt']
    dtmcoll = db['dtm_pnt']
    istmcoll = db['istm_pnt']
    rtmdcoll = db['rtm_d_pnt']
    dtmdcoll = db['dtm_d_pnt']
    istmdcoll = db['istm_d_pnt']

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

d = [5]
lr=[0.1]
df=[0.1]
eps=[0.5]
fd=[1]
s=[12000]
i=[10]
mvp=[0.3]
mbp=[0.9]
oap=[0.3]
ppvnpv=[0.9]

color = ['red', 'blue', 'green']

st.title("Detection Accuracy! ")

fig_learn_speed = make_subplots(
    cols=1,
    rows=1,
    # horizontal_spacing = 0.15,
    # subplot_titles=("(a) Detection accuracy", " (b) Dynamic Trust Threshold ")
    #specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)
fig_learn_speed.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    title_text="Detection Accuracy (%)",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True, 
    showline=True,
    zeroline=False, 
    linecolor='black',
    title_standoff=1, 
    col=1, 
    row=1,
)
fig_learn_speed.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Number of interactions",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black', 
    title_standoff=2, 
    col=1, 
    row=1,
    
)
# fig_learn_speed.update_yaxes(
#     showgrid=True, 
#     linewidth=2, 
#     # title_text="DTT",
#     gridcolor="gray", 
#     gridwidth=1, 
#     range=[0, 1], 
#     mirror=True, 
#     showline=True,
#     zeroline=False, 
#     linecolor='black', 
#     title_standoff=1,
#     col=2, 
#     row=1,
# )
# fig_learn_speed.update_xaxes(
#     showgrid=True, 
#     linewidth=2, 
#     showline=True,
#     zeroline=False, 
#     title_text="Steps",
#     gridcolor="gray", 
#     gridwidth=1, 
#     range=[0, 120], 
#     mirror=True, 
#     linecolor='black', 
#     col=2, 
#     row=1,
# )

fig_learn_speed.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    autosize=False,
    height=500, 
    width=500, 
    margin=dict(
        l=6,
        r=15,
        b=6,
        t=6,
        pad=5
    ),font=dict(
        size=24,
    ), 
    # legend=dict(
    #     orientation='h',
    #     yanchor="top",
    #     y=1.5,
    #     xanchor="center",
    #     x=0.01,
    #     bgcolor="rgba(255,255,255,255)",
    #     bordercolor="Black",
    #     borderwidth=2,
    #     font=dict(
    #         size=24,
    #     )
    # ),
)
# fig_learn_speed.update_layout(height=600, width=600)

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)
rwcombo = reconstruct_rtm(i, s, mvp, mbp, oap, ppvnpv)
for k in allcombination:
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000
    x = np.arange(int(xvalue)/100)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlsbaccuracy = mydocrlsb[0]['step_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['step_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
    #mvp: 0.1, 0.2, 0.3, 0.4
    #mbp: 0.1, 0.2, 0.3, 0.4, 0.5
    #oap: 0.1, 0.15, 0.2, 0.25, 0.3

    fig_learn_speed.add_trace(
        go.Scatter(x=x, y=rlsbaccuracy, name="CARES-S (5)", mode='lines', line=dict(color='black', width=4, dash='solid'), showlegend=False),
        col=1,
        row=1,        
        )
    fig_learn_speed.add_trace(
        go.Scatter(x=x, y=rlblaccuracy, name="CARES-B (5)", mode='lines', line=dict(color='black', width=4, dash='dot'), showlegend=False),
        col=1,
        row=1,
    )
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
    rlsbaccuracy = mydocrlsb[0]['step_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['step_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlsbaccuracy, name="CARES-S (v)", mode='lines', line=dict(color='black', width=4, dash='solid'), showlegend=False),
    #     col=1,
    #     row=1,        
    #     )
    # fig_learn_speed.add_trace(
    #     go.Scatter(x=x, y=rlblaccuracy, name="CARES-B (v)", mode='lines', line=dict(color='black', width=4, dash='dot'), showlegend=False),
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

# #! draw RW
for idx, j in enumerate(rwcombo):
    print(j)
    xvalue = getxvalue(j)
    xvalue=12000

    x = np.arange(int(xvalue)/100)

    myquery = {'id': str(j)}
    mydoc_rtm = list(rtmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtm = list(dtmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istm = list(istmcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_rtmd = list(rtmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_dtmd = list(dtmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))
    mydoc_istmd = list(istmdcoll.find(myquery, {"_id":0, "cum_accuracy":1, "step_accuracy":1, 'precision':1, 'recall':1, 'f1score':1, 'dtt':1}))

    rtm_accuracy = mydoc_rtm[0]['step_accuracy']
    rtm_precision = mydoc_rtm[0]['precision']
    rtm_recall = mydoc_rtm[0]['recall']   
    rtm_f1score = mydoc_rtm[0]['f1score']
    rtm_dtt = mydoc_rtm[0]['dtt']

    dtm_accuracy = mydoc_dtm[0]['step_accuracy']
    dtm_precision = mydoc_dtm[0]['precision']
    dtm_recall = mydoc_dtm[0]['recall']   
    dtm_f1score = mydoc_dtm[0]['f1score']
    dtm_dtt = mydoc_dtm[0]['dtt']

    istm_accuracy = mydoc_istm[0]['step_accuracy']
    istm_precision = mydoc_istm[0]['precision']
    istm_recall = mydoc_istm[0]['recall']
    istm_f1score = mydoc_istm[0]['f1score']
    istm_dtt = mydoc_istm[0]['dtt']

    rtmd_accuracy = mydoc_rtmd[0]['step_accuracy']
    rtmd_precision = mydoc_rtmd[0]['precision']
    rtmd_recall = mydoc_rtmd[0]['recall']
    rtmd_f1score = mydoc_rtmd[0]['f1score']
    rtmd_dtt = mydoc_rtmd[0]['dtt']

    dtmd_accuracy = mydoc_dtmd[0]['step_accuracy']
    dtmd_precision = mydoc_dtmd[0]['precision']
    dtmd_recall = mydoc_dtmd[0]['recall']
    dtmd_f1score = mydoc_dtmd[0]['f1score']
    dtmd_dtt = mydoc_dtmd[0]['dtt']

    istmd_accuracy = mydoc_istmd[0]['step_accuracy']
    istmd_precision = mydoc_istmd[0]['precision']
    istmd_recall = mydoc_istmd[0]['recall']
    istmd_f1score = mydoc_istmd[0]['f1score']
    istmd_dtt = mydoc_istmd[0]['dtt']

    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtm_accuracy, name="RTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmd_accuracy, name="RTMD", line=dict(color=color[0], width=4, dash='solid'), showlegend=False ),col=1, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtm_accuracy, name="DTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmd_accuracy, name="DTMD", line=dict(color=color[1], width=4, dash='solid'), showlegend=False),col=1, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istm_accuracy, name="ISTM", ),col=1, row=1)
    fig_learn_speed.add_trace(go.Scatter(x=x, y=istmd_accuracy, name="ISTMD", line=dict(color=color[2], width=4, dash='solid'), showlegend=False),col=1, row=1)

    # fig_learn_speed.add_trace(go.Scatter(x=x, y=rtmf11, name="RTM-f1",marker=dict(size=12,symbol="circle")) ,col=1, row=2)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in rtmd_dtt], name="RTMD", line=dict(color=color[0], width=4,),showlegend=False),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=dtmf11, name="DTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in dtmd_dtt], name="DTMD", line=dict(color=color[1], width=4,),showlegend=False),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=istmf11, name="ISTM-f1",marker=dict(size=12,symbol="circle")),col=2, row=1)
    # fig_learn_speed.add_trace(go.Scatter(x=x, y=[t / 100 for t in istmd_dtt], name="ISTMD",line=dict(color=color[2], width=4,),showlegend=False),col=2, row=1)

st.plotly_chart(fig_learn_speed, use_container_width=False)

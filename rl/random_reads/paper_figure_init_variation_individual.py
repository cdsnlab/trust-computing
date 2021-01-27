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
from plotly.subplots import make_subplots

import faulthandler
faulthandler.enable()

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']

    # cares_rl_sb = db['cares_rl_sbe']
    cares_rl_sb = db['cares_rl_sb_pnt']
    cares_rl_bl = db['cares_rl_bl_pnt']
    cares_rl_sb_custom=db['cares_rl_sb_pnt']
    cares_rl_bl_custom=db['cares_rl_bl_pnt']
    return cares_rl_sb, cares_rl_bl, cares_rl_sb_custom, cares_rl_bl_custom

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv): 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap, v_ppvnpvthr=ppvnpv):
        allcombinations.append(str(output))
    return allcombinations

def getxvalue(key): #parse for v_s
    elements = key.split(',')
    for i in elements:
        if "v_s" in i:
            temp=i.split('=')
            return temp[1]
            
data_load_state = st.text('Loading data...')
cares_rl_sb, cares_rl_bl, cares_rl_sb_custom, cares_rl_bl_custom = connect()
data_load_state.text('Loading data...done!')


d=[5]
lr = [0.1]
df=[0.1]
eps=[0.5]
fd=[1]
s=[12000]
i=[10,50,90]
mvp=[0.3]
mbp=[0.9]
oap=[0.3]
ppvnpv=[0.9]

######################################## SB
#########################################
#########################################

#######
sb_results_acc = make_subplots(
    cols=1,
    rows=1,
)
#* ACC
sb_results_acc.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Detection Accuracy (%)",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True,  
    linecolor='black', 
    title_standoff=1,

    )
sb_results_acc.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
sb_results_acc.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ),    
)

sb_results_dtt = make_subplots(
    cols=1,
    rows=1,
)
#* DTT
sb_results_dtt.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Dynamic Trust Threshold",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 1], 
    mirror=True,  
    linecolor='black',
    title_standoff=1,

    )
sb_results_dtt.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
sb_results_dtt.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=510, width=640, 
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ), 
    legend=dict(
        orientation='h',
        yanchor="bottom",
        y=1.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        bordercolor="Black",
        borderwidth=2,
        font=dict(
            size=29,
        )
    ),
    
)


sb_results_rew = make_subplots(
    cols=1,
    rows=1,
)
#* REW
sb_results_rew.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Average Rewards",
    gridcolor="gray", 
    gridwidth=1, 
    range=[-1.1, 2.1], 
    mirror=True,  
    linecolor='black', 
    title_standoff=1,

    )
sb_results_rew.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
sb_results_rew.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ), 
    # legend=dict(
    #     orientation='h',
    #     yanchor="bottom",
    #     y=1.1,
    #     xanchor="right",
    #     x=0.65,
    #     bgcolor="rgba(0,0,0,0)",
    #     bordercolor="Black",
    #     borderwidth=2,
    #     font=dict(
    #         size=24,
    #     )
    # ),
    
)


symbols = ['circle','diamond','cross']
linetype = ['dot', 'dash', 'solid']
color = ['red', 'blue', 'green']
allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000

    x = np.arange(int(xvalue)/100)

    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlsbaccuracy = mydocrlsb[0]['step_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    # mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # rlblaccuracy = mydocrlbl[0]['cum_accuracy']
    # rlblprecision = mydocrlbl[0]['precision']
    # rlblrecall = mydocrlbl[0]['recall']
    # rlblrew = mydocrlbl[0]['cum_rew']
    # rlbldtt = mydocrlbl[0]['avg_dtt']
    # rlblf1 = mydocrlbl[0]['f1score']
    # rlblerror = mydocrlbl[0]['error']
    
    sb_results_acc.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=False))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="SB-Initial DTT {}".format(i[idx])))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx])))

    sb_results_dtt.add_trace(go.Scatter(x=x, y=[t / 100 for t in rlsbdtt], line=dict(width=4, color=color[idx],dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=True)) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="SB-Initial DTT {}".format(i[idx]))) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlbldtt, name="BL"))
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx]))) 

    sb_results_rew.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"),  name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=False))
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="SB-Initial DTT {}".format(i[idx]))) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx]))) 

#*custom version (RLBL-delta:1, )
d=[1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000

    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb_custom.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
    rlsbaccuracy = mydocrlsb[0]['cum_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['cum_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']

    # sb_results.add_trace(go.Scatter(x=x, y=rlsbaccuracy,line=dict(width=4, color=color[idx],dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=True), row=1, col=1)
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="SB-variable {}".format(i[idx])))
    # # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="BL-variable {}".format(i[idx])))

    # sb_results.add_trace(go.Scatter(x=x, y=[t / 100 for t in rlsbdtt], line=dict(width=4, color=color[idx],dash=linetype[idx]), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=2) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 

    # sb_results.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=3) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 


d=[5]

######################################## BL
#########################################
#########################################
#######
bl_results_acc = make_subplots(
    cols=1,
    rows=1,
)
#* ACC
bl_results_acc.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Detection Accuracy (%)",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 100], 
    mirror=True,  
    linecolor='black', 
    title_standoff=1,

    )
bl_results_acc.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
bl_results_acc.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ),    
)

bl_results_dtt = make_subplots(
    cols=1,
    rows=1,
)
#* DTT
bl_results_dtt.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Dynamic Trust Threshold",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 1], 
    mirror=True,  
    linecolor='black',
    title_standoff=1,

    )
bl_results_dtt.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
bl_results_dtt.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=510, width=640, margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ), 
    legend=dict(
        orientation='h',
        yanchor="bottom",
        y=1.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        bordercolor="Black",
        borderwidth=2,
        font=dict(
            size=29,
        )
    ),
    
)


bl_results_rew = make_subplots(
    cols=1,
    rows=1,
)
#* REW
bl_results_rew.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True, 
    zeroline=False, 
    # title_text="Average Rewards",
    gridcolor="gray", 
    gridwidth=1, 
    range=[-1.1, 2.1], 
    mirror=True,  
    linecolor='black', 
    title_standoff=1,

    )
bl_results_rew.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    title_text="Steps",
    gridcolor="gray", 
    gridwidth=1, 
    range=[0, 120], 
    mirror=True, 
    linecolor='black',
    title_standoff=5,

    )
bl_results_rew.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    font=dict(
        size=24,
    ), 
    # legend=dict(
    #     orientation='h',
    #     yanchor="bottom",
    #     y=1.1,
    #     xanchor="right",
    #     x=0.65,
    #     bgcolor="rgba(0,0,0,0)",
    #     bordercolor="Black",
    #     borderwidth=2,
    #     font=dict(
    #         size=24,
    #     )
    # ),
    
)


allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000

    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    # mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # rlsbaccuracy = mydocrlsb[0]['step_accuracy']
    # rlsbprecision = mydocrlsb[0]['precision']
    # rlsbrecall = mydocrlsb[0]['recall']
    # rlsbrew = mydocrlsb[0]['cum_rew']
    # rlsbdtt = mydocrlsb[0]['avg_dtt']
    # rlsbf1 = mydocrlsb[0]['f1score']
    # rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['step_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']

    bl_results_acc.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=False))   
    # fig_da_init_bl.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx])))

    bl_results_dtt.add_trace(go.Scatter(x=x, y=[t / 100 for t in rlbldtt], line=dict(width=4, color=color[idx],dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=True)) 
    # fig_dtt_init_bl.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx]))) 

    bl_results_rew.add_trace(go.Scatter(x=x, y=rlblrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"),name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=False)) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="SB-Initial DTT {}".format(i[idx]))) 
    # fig_rew_init_bl.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-Initial DTT {}".format(i[idx]))) 

#*custom version (RLBL-delta:1, )
d=[1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap, ppvnpv)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    xvalue=12000

    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb_custom.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
    rlsbaccuracy = mydocrlsb[0]['cum_accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "cum_accuracy": 1, "step_accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['cum_accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
    # bl_results.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=4, color='black',dash=linetype[idx]), name="𝜃𝑖𝑛𝑖𝑡: {}".format(i[idx]/100),showlegend=True), row=1, col=1)
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="SB-variable {}".format(i[idx])))
    # # fig_da_init.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_da_init.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=2, color="yellow",dash=linetype[idx]), name="BL-variable {}".format(i[idx])))

    # bl_results.add_trace(go.Scatter(x=x, y=[t / 100 for t in rlbldtt], line=dict(width=4, color='black',dash=linetype[idx]), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=2) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_dtt_init.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 

    # bl_results.add_trace(go.Scatter(x=x, y=rlblrew, mode='markers', marker=dict(color='black', size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="init θ: {}".format(i[idx]),showlegend=True), row=1, col=3) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="SB-variable {}".format(i[idx]))) 
    # fig_rew_init.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color='yellow',dash=linetype[idx]), name="BL-variable {}".format(i[idx]))) 


st.title("SB Scheme: ")
st.plotly_chart(sb_results_acc, use_container_width=False)
st.plotly_chart(sb_results_dtt, use_container_width=False)
st.plotly_chart(sb_results_rew, use_container_width=False)



st.title("BL Scheme: ")
st.plotly_chart(bl_results_acc , use_container_width=False)
st.plotly_chart(bl_results_dtt , use_container_width=False)
st.plotly_chart(bl_results_rew , use_container_width=False)
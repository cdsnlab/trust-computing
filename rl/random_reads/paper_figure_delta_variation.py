'''
This file draws the difference it makes in how much the threshold hold move each time.
input files: CARES_RL * 
output diagram: Learning accuracy with RL algorithms
'''

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itertools import product, starmap
from collections import namedtuple
from pymongo import MongoClient

import faulthandler
faulthandler.enable()

def connect():
    client = MongoClient('localhost', 27017)
    db = client['trustdb']

    # cares_rl_sb = db['cares_rl_sbe']
    cares_rl_sb = db['cares_rl_sb95_r']
    cares_rl_bl = db['cares_rl_bl95_r']
    cares_rl_sb_custom=db['cares_rl_sb_custom95_r']
    cares_rl_bl_custom=db['cares_rl_bl_custom95_r']
    return cares_rl_sb, cares_rl_bl, cares_rl_sb_custom, cares_rl_bl_custom

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d,v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i, v_mvp=mvp, v_mbp=mbp, v_oap=oap):
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

d=[1,5,9]
lr = [0.1]
df=[0.1]
eps=[0.5]
# eps=[0.1]
fd=[1]
s=[59999]
i=[50]
mvp=[0.2]
mbp=[0.5]
oap=[0.1]


######################################## SB
#########################################
#########################################

#######
fig_detection_accuracy = make_subplots(
    cols=1,
    rows=1,
    subplot_titles=([""]),
    # specs=[[{"secondary_y": True}]]
)

fig_detection_accuracy.update_yaxes(
    showgrid=True, linewidth=2, title_text="Detection Accuracy (%)",gridcolor="gray", gridwidth=1, range=[90, 100], mirror=True, showline=True,zeroline=False, linecolor='black', 
    )
fig_detection_accuracy.update_xaxes(
    showgrid=True, linewidth=2,gridcolor="gray", gridwidth=1, range=[0, 600], title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_detection_accuracy.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),font=dict(
        size=24,
    ), 
    # legend=dict(
    # yanchor="bottom",
    # y=0.01,
    # xanchor="right",
    # x=0.99,
    # bgcolor="LightSteelBlue",
    # bordercolor="Black",
    # borderwidth=2
    # )
    )
#######
fig_rew = make_subplots(
    cols=1,
    rows=1,
    # subplot_titles=(["Reward"])
)
fig_rew.update_yaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1,  showline=True,zeroline=False, linecolor='black',mirror=True, title_text="Average Reward"
    )
fig_rew.update_xaxes(
    showgrid=True, linewidth=2,gridcolor="gray", gridwidth=1, range=[0, 600], title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_rew.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),
    font=dict(
        size=24,
    ),
    # legend=dict(
        
    #     bgcolor="LightSteelBlue",
    #     bordercolor="Black",
    #     borderwidth=2
    # )
    )
########
fig_dtt = make_subplots(
    cols=1,
    rows=1,
    # subplot_titles=(["Dynamic Trust Threshold (DTT) "])
)
fig_dtt.update_yaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1, range=[0, 100], mirror=True, showline=True, zeroline=False, linecolor='black', title_text="Trust threshold",
    )
fig_dtt.update_xaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1, range=[0, 600],title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_dtt.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,
height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),
    font=dict(
        size=24,
    ),
    # legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01,
    #     bgcolor="LightSteelBlue",
    #     bordercolor="Black",
    #     borderwidth=2
    # )
)

# fig_beta = make_subplots(
#     cols=1,
#     rows=1,
#     # subplot_titles=(["Reward"])
# )
# fig_beta.update_yaxes(showgrid=True, gridcolor="gray", range=[0, 100],  showline=True, linecolor='black')
# fig_beta.update_xaxes(showgrid=True, gridcolor="gray", mirror=True, showline=True, linecolor='black')
# fig_beta.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',height=400, width=600)

symbols = ['circle','diamond','cross']
linetype = ['dot', 'dash', 'solid']
color = ['red', 'blue', 'green']
allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

#* orignal version
for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']

    fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="Delta  {}".format(d[idx]),showlegend=True))
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="S-Delta  {}".format(d[idx])), secondary_y=True,) #! if u r going to add another y-axis
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-delta {}".format(d[idx])))

    
    fig_dtt.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="Delta  {}".format(d[idx]),showlegend=True)) 
    # fig_dtt.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-delta {}".format(d[idx]))) 

    # fig_rew.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color=color[idx], size=10, symbol='x'),line=dict(width=2, color=color[idx],dash=linetype[idx]), name="S-Delta  {}".format(d[idx]))) 
    fig_rew.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="Delta  {}".format(d[idx]),showlegend=True)) 
    # fig_rew.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-delta {}".format(d[idx]))) 

#*custom version (RLBL-delta:1, )
d=[1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb_custom.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
        
    fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy,line=dict(width=4, color="black",dash="solid"), name="Variable",showlegend=True))
    # # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=1), error_y = dict(type='data',array= rlsberror, visible=True), name="me"))
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=2, color="yellow",dash="solid"), name="BL-variable"))
    
    fig_dtt.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=4, color='black',dash="solid"), name="Variable",showlegend=True)) 
    # fig_dtt.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color='yellow',dash="solid"), name="BL-variable")) 

    fig_rew.add_trace(go.Scatter(x=x, y=rlsbrew, mode='markers', marker=dict(color='black', size=10, symbol='circle', opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="Variable",showlegend=True)) 
    # fig_rew.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color='yellow',dash="solid"), name="BL-variable"))


st.title("SB Scheme: Accuracy over time")
st.plotly_chart(fig_detection_accuracy, use_container_width=False)

st.title("SB Scheme: Reward over time")
st.plotly_chart(fig_rew, use_container_width=False)

st.title("SB Scheme: DTT over time")
st.plotly_chart(fig_dtt, use_container_width=False)


######################################## BL
#########################################
#########################################

d=[1,5,9]
lr = [0.1]
df=[0.1]
eps=[0.5]
# eps=[0.1]
fd=[1]
s=[59999]
i=[50]
mvp=[0.2]
mbp=[0.5]
oap=[0.1]

####
fig_detection_accuracy_bl = make_subplots(
    cols=1,
    rows=1,
    subplot_titles=([""]),
    # specs=[[{"secondary_y": True}]]
)

fig_detection_accuracy_bl.update_yaxes(
    showgrid=True, linewidth=2, title_text="Detection Accuracy (%)",gridcolor="gray", gridwidth=1, range=[90, 100], mirror=True, showline=True,zeroline=False, linecolor='black', 
    )
fig_detection_accuracy_bl.update_xaxes(
    showgrid=True, linewidth=2,gridcolor="gray", gridwidth=1, range=[0, 600], title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_detection_accuracy_bl.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),font=dict(
        size=24,
    ), 
    # legend=dict(
    # yanchor="bottom",
    # y=0.01,
    # xanchor="right",
    # x=0.99,
    # bgcolor="LightSteelBlue",
    # bordercolor="Black",
    # borderwidth=2
    # )
    )
#######
fig_rew_bl = make_subplots(
    cols=1,
    rows=1,
    # subplot_titles=(["Reward"])
)
fig_rew_bl.update_yaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1,  showline=True,zeroline=False, linecolor='black',mirror=True, title_text="Average Reward"
    )
fig_rew_bl.update_xaxes(
    showgrid=True, linewidth=2,gridcolor="gray", gridwidth=1, range=[0, 600], title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_rew_bl.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),
    font=dict(
        size=24,
    ),
    # legend=dict(
        
    #     bgcolor="LightSteelBlue",
    #     bordercolor="Black",
    #     borderwidth=2
    # )
    )
########
fig_dtt_bl = make_subplots(
    cols=1,
    rows=1,
    # subplot_titles=(["Dynamic Trust Threshold (DTT) "])
)
fig_dtt_bl.update_yaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1, range=[0, 100], mirror=True, showline=True, zeroline=False, linecolor='black', title_text="Trust threshold",
    )
fig_dtt_bl.update_xaxes(
    showgrid=True, linewidth=2, gridcolor="gray", gridwidth=1, range=[0, 600],title_text="Interaction number",mirror=True, showline=True,zeroline=False, linecolor='black'
    )
fig_dtt_bl.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', autosize=False,
height=400, width=600, margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=4
    ),
    font=dict(
        size=24,
    ),
    # legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01,
    #     bgcolor="LightSteelBlue",
    #     bordercolor="Black",
    #     borderwidth=2
    # )
)

#######################################################DELTA
#* orignal version
allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
    
    fig_detection_accuracy_bl.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="Delta  {}".format(d[idx]),showlegend=True)) 
    
    fig_dtt_bl.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=4, color=color[idx],dash=linetype[idx]), name="Delta  {}".format(d[idx]),showlegend=True)) 
    
    fig_rew_bl.add_trace(go.Scatter(x=x, y=rlblrew, mode='markers', marker=dict(color=color[idx], size=10, symbol=symbols[idx], opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="Delta  {}".format(d[idx]),showlegend=True)) 
    # fig_rew.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color=color[idx],dash=linetype[idx]), name="BL-delta {}".format(d[idx]))) 


#*custom version (RLBL-delta:1, )
d=[1]

allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb_custom.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
    rlsbaccuracy = mydocrlsb[0]['accuracy']
    rlsbprecision = mydocrlsb[0]['precision']
    rlsbrecall = mydocrlsb[0]['recall']
    rlsbrew = mydocrlsb[0]['cum_rew']
    rlsbdtt = mydocrlsb[0]['avg_dtt']
    rlsbf1 = mydocrlsb[0]['f1score']
    rlsberror = mydocrlsb[0]['error']

    mydocrlbl=list(cares_rl_bl_custom.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    rlblaccuracy = mydocrlbl[0]['accuracy']
    rlblprecision = mydocrlbl[0]['precision']
    rlblrecall = mydocrlbl[0]['recall']
    rlblrew = mydocrlbl[0]['cum_rew']
    rlbldtt = mydocrlbl[0]['avg_dtt']
    rlblf1 = mydocrlbl[0]['f1score']
    rlblerror = mydocrlbl[0]['error']
        
  
    fig_detection_accuracy_bl.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=4, color="black",dash="solid"), name="Variable",showlegend=True))
    # fig_detection_accuracy_bl.add_trace(go.Scatter(x=x, y=rlblaccuracy,line=dict(width=2, color="yellow",dash="solid"), name="BL-variable"))
    
    fig_dtt_bl.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=4, color='black',dash="solid"), name="Variable",showlegend=True))
    # fig_dtt_bl.add_trace(go.Scatter(x=x, y=rlsbdtt, line=dict(width=2, color='yellow',dash="solid"), name="SB-variable")) 
    # fig_dtt_bl.add_trace(go.Scatter(x=x, y=rlbldtt, line=dict(width=2, color='yellow',dash="solid"), name="BL-variable")) 

    fig_rew_bl.add_trace(go.Scatter(x=x, y=rlblrew, mode='markers', marker=dict(color='black', size=10, symbol='circle', opacity=0.2, line=dict(color='black')),line=dict(width=4, color='black',dash="solid"), name="Variable",showlegend=True)) 
    # fig_rew_bl.add_trace(go.Scatter(x=x, y=rlsbrew, line=dict(width=2, color='yellow',dash="solid"), name="SB-variable")) 
    # fig_rew_bl.add_trace(go.Scatter(x=x, y=rlblrew, line=dict(width=2, color='yellow',dash="solid"), name="BL-variable"))


st.title("BL Scheme: Accuracy over time")
st.plotly_chart(fig_detection_accuracy_bl , use_container_width=False)

st.title("BL Scheme: Reward over time")
st.plotly_chart(fig_rew_bl, use_container_width=False)

st.title("BL Scheme: DTT over time")
st.plotly_chart(fig_dtt_bl, use_container_width=False)
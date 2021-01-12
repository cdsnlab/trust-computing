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

    cares_rl_sb = db['cares_rl_sb']
    cares_rl_bl = db['cares_rl_bl']
    cares_rl_sb_custom=db['cares_rl_sb_custom_action']
    cares_rl_bl_custom=db['cares_rl_bl_custom_action']
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
#reconstruct multiselect options, match it with 
# d =   st.multiselect("Delta", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13], default=[1])
# lr =  st.multiselect("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
# df =  st.multiselect("Discount factor", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.1])
# eps = st.multiselect("Epsilon", [0.01, 0.05, 0.1, 0.2, 0.5], default=[0.5])
# fd =  st.multiselect("Feedback delay",[1, 2, 5, 10, 20, 50, 100,200], default=[1])
# s =   st.multiselect("Total number of steps", [59999], default=[59999])
# i =   st.multiselect("Initial starting value", [10, 50, 90], default=[50])
# mvp =   st.multiselect("Malicious vehicle probability", [0.1, 0.2, 0.3, 0.4], default=[0.2])
# mbp =   st.multiselect("Malicious Behavior probability", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.5])
# oap =   st.multiselect("Outside attack probability", [0.1, 0.15, 0.2, 0.25, 0.3], default=[0.2])
d=[5]
lr = [0.1]
df=[0.1]
eps=[0.5]
fd=[1]
s=[59999]
i=[10]
mvp=[0.2]
mbp=[0.5]
oap=[0.2]
#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.s
allcombination = reconstruct(d, lr, df, eps, fd, s, i, mvp, mbp, oap)

st.title("Accuracy over time")
fig_detection_accuracy = make_subplots(
    cols=1,
    rows=1,
    # subplot_titles=("Malicious Vehicle Detection Accuracy ")
)
fig_detection_accuracy.update_yaxes(showgrid=True, gridcolor="black", range=[50, 100], mirror=True, showline=True, linecolor='black')
fig_detection_accuracy.update_xaxes(showgrid=True, gridcolor="black", mirror=True, showline=True, linecolor='black')
fig_detection_accuracy.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',height=400, width=600)


#* orignal version
for idx, k in enumerate(allcombination):
    print(k)
    xvalue = getxvalue(k)
    x = np.arange(int(xvalue)/100)
    myquery = {"id": str(k)}
    
    mydocrlsb=list(cares_rl_sb.find(myquery, {"_id":0, "accuracy": 1, 'precision':1, 'recall':1, 'cum_rew':1, 'avg_dtt':1, 'f1score':1, 'error':1}))
    # print(mydocrlsb)
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
        
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=2), name="CARES_SB"))
    # fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=2), name="CARES_BL"))

#*custom version (RLBL-delta:1, )
d=[1]
i=[10, 50, 90]

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
        
    fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlsbaccuracy, line=dict(width=2), name="CARES_SB_CUSTOM"))
    fig_detection_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=2), name="CARES_BL_CUSTOM"))

    #! DTT 그려봐! 
    #! custom 버릴거야???
    #! 
    # fig_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=4),error_y = dict(type='data',array= rlsberror, visible=True), name="BL: Delta {}".format(d[i])))
    # fig_accuracy.add_trace(go.Scatter(x=x, y=rlblaccuracy, line=dict(width=4), name="BL: Delta {}".format(d[i])))
    
    # fig_precision.add_trace(go.Scatter(x=x, y=rlblprecision, name="BL: Delta {}".format(d[i])))
    # fig_recall.add_trace(go.Scatter(x=x, y=rlblrecall, name="BL: Delta {}".format(d[i])))
    # fig_f1.add_trace(go.Scatter(x=x, y=rlblf1, name="BL: Delta {}".format(d[i])))

    # print(x)

    # fig_dtt.add_trace(go.Scatter(x=x, y=rlsbdtt, name="Average DTT value"))
    # # fig_dtt.add_trace(go.Scatter(x=x, y=avg_gt, name="Average GT value"))
    # fig_cum_rew.add_trace(go.Scatter(x=x, y=rlsbrew, name="SB"))
    # fig_cum_rew.add_trace(go.Scatter(x=x, y=rlblrew, name="BL"))


st.plotly_chart(fig_detection_accuracy, use_container_width=True)
# st.plotly_chart(fig_precision, use_container_width=True)
# st.plotly_chart(fig_recall, use_container_width=True)
# st.plotly_chart(fig_f1, use_container_width=True)
# st.plotly_chart(fig_cum_rew, use_container_width=True)
# st.plotly_chart(fig_dtt, use_container_width=True)

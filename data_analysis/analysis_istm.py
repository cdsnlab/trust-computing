#* this file is used by RTM schemes
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient


NUM_VEHICLES = 100
NUM_INTERACTIONS = 600
m_=defaultdict(list)
b_=defaultdict(list)

data = pd.read_csv('/home/spencer1/trust-computing/sampledata/is_df_0_0.5mbp0.2oap0.2mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0, nrows=3200000)

fig = go.FigureWidget(
    layout=go.Layout(title="Trust values plotting for ISTM DATASET ",xaxis=dict(title="Indirect"),yaxis=dict(title="Direct", range=[0,100] ))
)

for i in range (NUM_VEHICLES * NUM_INTERACTIONS):
    
    i_tv = data['indirect_tv'][i] *100
    d_tv = data['direct_tv'][i] *100
    
    status = data['status'][i]
    # print(i_tv, d_tv, status)

    # trust_values.append(float(good_history / (bad_history + good_history) * 100))
    # statuses.append(data['status'][index-1])
    if status == 1:
        m_['itv'].append(i_tv)
        m_['dtv'].append(d_tv)
        m_['s'].append(status)
    else:
        b_['itv'].append(i_tv)
        b_['dtv'].append(d_tv)
        b_['s'].append(status)
# print(len(m_['itv']))
# print(len(b_['itv']))

fig.update_traces(mode='markers', marker_line_width=20, marker_size=100)

fig.add_trace(go.Scatter(mode='markers', x=m_['itv'], y=m_['dtv'], name="malicious", marker=dict(
        color='red',
        size=10,
        )))

fig.add_trace(go.Scatter(mode='markers', x=b_['itv'], y=b_['dtv'], name="benign", marker=dict(
        color='blue',
        size=10,
        symbol='x'
        )))

st.plotly_chart(fig, use_container_width=True)

# count=0
# for i in range (len(trajectories)):
    
#     latplots = []
#     lonplots = []

#     allplots = []
#     res=json.loads(trajectories[i]) 
#     print(len(res))
#     for j in range(len(res)):
#         allplots.append([res[j][1], res[j][0]])
#         latplots.append(res[j][1])
#         lonplots.append(res[j][0])
#     row = {"index": str(count), "lon": lonplots, "lat": latplots, "both": allplots}
#     mdb.insert_one(row)
#     print(row)

#     count+=1
#     if count > MAXPLOTS: #* so it doesn't break :( 
#         break
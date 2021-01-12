#* this file is used by RTM schemes
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient


BENIGN_STATUS = 0
MALICIOUS_STATUS = 1
NUM_VEHICLES = 100
NUM_INTERACTIONS = 600
NUM_DATA_PER_CONTEXT = 500
DATA_PER_VEHICLE = 64 * NUM_DATA_PER_CONTEXT

b_tv = []
b_s = []
m_tv = []
m_s  = []
data = pd.read_csv('/home/spencer/trust-computing/sampledata/ce_db_0_0.5mbp0.2oap0.2mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0, nrows=3200000)

fig = go.FigureWidget(
    layout=go.Layout(title="Trust values plottingfor CE_DB",xaxis=dict(title="Trust value"),yaxis=dict(title="Benign (0) vs Malicious (1)", range=[-0.1, 1.1] ))
)

for i in range (NUM_VEHICLES):
    index = DATA_PER_VEHICLE * i + NUM_DATA_PER_CONTEXT
    good_history = data['good_history'][index-1]
    bad_history = data['bad_history'][index-1]
    trustvalue = float(good_history / (bad_history + good_history) * 100)
    status = data['status'][index-1]

    # trust_values.append(float(good_history / (bad_history + good_history) * 100))
    # statuses.append(data['status'][index-1])
    if status == 1:
        m_s.append(status)
        m_tv.append(trustvalue)
    else:
        b_s.append(status)
        b_tv.append(trustvalue)

# for i in range(len(m_s)):
    
fig.add_trace(go.Scatter(mode='markers', x=m_tv, y=m_s, name="malicious", marker=dict(
        color='red',
        size=12,
        )))
# for j in range(len(b_s)):
fig.add_trace(go.Scatter(mode='markers', x=b_tv, y=b_s, name="benign", marker=dict(
        color='blue',
        size=12,
        symbol='x'

        )))

st.plotly_chart(fig, use_container_width=True)

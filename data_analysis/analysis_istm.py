#* this file is used by RTM schemes
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from plotly.subplots import make_subplots

NUM_VEHICLES = 100
NUM_INTERACTIONS = 100


data = pd.read_csv('/home/spencer/trust-computing/sampledata/is_df_0_0.5mbp0.2oap0.2mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0)
# data = pd.read_csv('/home/spencer/trust-computing/is_df_0_0.5mbp0.2oap0.2mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0)


fig = make_subplots(
    cols=1,
    rows=NUM_INTERACTIONS+1,
    # subplot_titles=("Malicious Vehicle Detection Accuracy ")
)
fig.update_layout(height=60000, width=100, title_text="ISTM dataset (split by NUM_ITERACTION)")

# for i in range (NUM_VEHICLES * NUM_INTERACTIONS):
for i in range (NUM_INTERACTIONS):
    m_=defaultdict(list)
    b_=defaultdict(list)
    for j in range(NUM_VEHICLES):
        index = j+i*NUM_VEHICLES
        i_tv = data['indirect_tv'][index] *100
        d_tv = data['direct_tv'][index] *100
        
        status = data['status'][index]

        if status == 1:
            m_['itv'].append(i_tv)
            m_['dtv'].append(d_tv)
            m_['s'].append(status)
        else:
            b_['itv'].append(i_tv)
            b_['dtv'].append(d_tv)
            b_['s'].append(status)

    fig.add_trace(go.Scatter(mode='markers', x=m_['itv'], y=m_['dtv'], name="malicious", marker=dict(
            color='red',
            size=5,
            )),
            col=1,
            row=i+1
    )

    fig.add_trace(go.Scatter(mode='markers', x=b_['itv'], y=b_['dtv'], name="benign", marker=dict(
            color='blue',
            size=5,
            symbol='x',
            )),
            col=1,
            row=i+1
    )

st.plotly_chart(fig, use_container_width=True)

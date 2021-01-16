#* this file is used by CZARES schemes
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from plotly.subplots import make_subplots


NUM_VEHICLES = 100
NUM_INTERACTIONS = 1
MBP_LIST=[0.1, 0.2, 0.3, 0.4, 0.5]
OAP_LIST=[0.1, 0.15, 0.2, 0.25, 0.3]
MVP_LIST=[0.1, 0.2, 0.3, 0.4]

# data = pd.read_csv('/home/spencer/trust-computing/sampledata/cares_df_0_0.1mbp0.3oap0.1mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0)
# data = pd.read_csv('/home/spencer/trust-computing/is_df_0_0.5mbp0.2oap0.2mvp.csv', sep=',', error_bad_lines=False, encoding='latin1', header=0)
# print(set(zip(MBP_LIST, OAP_LIST, MVP_LIST)))

list_3 = [ "MBP"+str(x) +"OAP"+ str(y) +"MVP"+ str(z) for x in MBP_LIST for y in OAP_LIST for z in MVP_LIST]
# print(list_3)

fig = make_subplots(
    cols=len(MVP_LIST),
    rows=len(MBP_LIST)*len(OAP_LIST),
    subplot_titles=(list_3)
)
fig.update_layout(height=len(MBP_LIST)*len(OAP_LIST)*500, width=len(MVP_LIST)*600, title_text="CARES dataset (split by NUM_ITERACTION)")

rowcount=0
for a in (MBP_LIST):
    for b in (OAP_LIST):
        count=1
        for c in (MVP_LIST):
            data = pd.read_csv('/home/spencer/trust-computing/sampledata/cares_df_0_{}mbp{}oap{}mvp.csv'.format(a,b,c), sep=',', error_bad_lines=False, encoding='latin1', header=0)

            for i in range (NUM_INTERACTIONS):
                m_=defaultdict(list)
                b_=defaultdict(list)
                for j in range(NUM_VEHICLES):
                    index = j+i*NUM_VEHICLES+10000
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
                        ),showlegend=False),
                        col=count,
                        row=rowcount+1,
                )

                fig.add_trace(go.Scatter(mode='markers', x=b_['itv'], y=b_['dtv'], name="benign", marker=dict(
                        color='blue',
                        size=5,
                        symbol='x',
                        ),showlegend=False),
                        col=count,
                        row=rowcount+1,
                )
            count+=1
        rowcount+=1

st.plotly_chart(fig, use_container_width=False)
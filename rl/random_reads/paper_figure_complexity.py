
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product, starmap
from collections import namedtuple
from pymongo import MongoClient

import faulthandler
content=[]
x=[["RTMD"], ["DTMD"], ["ISTMD"], ["CARES-S"], ["CARES-B"]]
y=[[453], [284], [393], [397], [408]]

content.append((["RTMD"], [453]))
content.append((["DTMD"], [284]))
content.append((["ISTMD"], [393]))
content.append((["CARES-S"], [397]))
content.append((["CARES-B"], [408]))

errors=[[5.5], [1.1], [2.3], [2.3], [3.0]]

colortypes=['green', 'orange', 'blue', 'red', 'red']


fig_comp = make_subplots(
    cols=1,
    rows=1,
    
)

fig_comp.update_yaxes(
    showgrid=True, 
    linewidth=2, 
    title_text="Simulation runtime (ms)",
    gridcolor="gray", 
    gridwidth=1, 
    # range=[0, 100], 
    mirror=True, 
    showline=True,
    zeroline=False, 
    linecolor='black',
    title_standoff=10, 
    col=1, 
    row=1,
)
fig_comp.update_xaxes(
    showgrid=True, 
    linewidth=2, 
    showline=True,
    zeroline=False, 
    # title_text="Number of interactions",
    gridcolor="gray", 
    gridwidth=1, 
    # range=[0, 120], 
    mirror=True, 
    linecolor='black', 
    title_standoff=2, 
    col=1, 
    row=1,
)
fig_comp.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    autosize=False,
    height=450, 
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
for i in range(len(content)):
    fig_comp.add_trace(
        go.Bar(x=content[i][0], y=content[i][1], 
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=errors[i],
            visible=True,
            thickness=2,
            width=15,
        ),
    
        marker_color=colortypes[i], 
        showlegend=False),
    )

# fig_comp.add_trace(
#     go.Bar(x=x, y=y, 
#     error_y=dict(
#         type='data', # value of error bar given in data coordinates
#         array=errors,
#         visible=True),

#     marker_color=colortypes, 
#     showlegend=False),
# )
st.plotly_chart(fig_comp, use_container_width=False)

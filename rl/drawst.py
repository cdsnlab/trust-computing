import numpy as np
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product, starmap
from collections import namedtuple
import json

filename = 'result/test.json'

def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))

def load_data(filename):
    with open(filename, 'r') as data:
        json_decoded = json.load(data)
        #print(json_decoded)
    return json_decoded

def reconstruct(d, lr, df, eps, fd, s, i): #* returns a string line 
    allcombinations=[]
    for output in named_product(v_d=d, v_lr=lr, v_df=df, v_eps=eps, v_fd=fd, v_s=s, v_i=i):
        allcombinations.append(str(output))
    return allcombinations



data_load_state = st.text('Loading data...')
jsondata = load_data(filename)
data_load_state.text('Loading data...done!')
st.title("graph showing combination of various experiment parameters")
#reconstruct selectbox options, match it with 
d = st.selectbox("Delta", [1, 3, 5])
lr = st.selectbox("Learning rate", [0.01,0.1,0.5, 0.9])
df = st.selectbox("Discount factor", [0.1, 0.5, 0.9])
eps = st.selectbox("Epsilon", [0.1, 0.5, 0.9])
fd = st.selectbox("Feedback delay", [100, 200, 500])
s = st.selectbox("Total number of steps", [1000, 10000, 50000])
i = st.selectbox("Initial starting value", [10, 50, 90] )

#! for all pairs of d, lr, df, eps, fd, s, i, create a string and find it from the JSON.
allcombination = reconstruct(d, lr, df, eps, fd, s, i)

fig = go.FigureWidget()
st.plotly_chart(fig, use_container_width=True)
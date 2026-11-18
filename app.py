from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title='MarketLab | Stock Intelligence', page_icon='📈', layout='wide')
st.markdown('''<style>
/* MarketLab — restrained editorial layout */
.block-container {
    max-width: 1380px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
.hero {
    position: relative;
    padding: 1.25rem 1.5rem 1.15rem;
    margin-bottom: 1.6rem;
    background: rgba(120, 132, 112, .075);
    border: 1px solid rgba(120, 132, 112, .22);
    border-left: 4px solid #788774;
    border-radius: 5px;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 650;
    letter-spacing: -0.7px;
    margin: 0 0 .3rem 0;
}
.hero p {
    margin: 0 0 .65rem 0;
    color: var(--text-color);
    opacity: .78;
    font-size: .98rem;
}
.small-muted {
    color: var(--text-color);
    opacity: .62;
    font-size: .82rem;
}
h2, h3 {
    letter-spacing: -.25px;
}
div[data-testid="stMetric"] {
    padding: .8rem .9rem;
    background: rgba(128, 128, 128, .045);
    border: 1px solid rgba(128, 128, 128, .18);
    border-radius: 6px;
}
div[data-testid="stMetricLabel"] {
    font-size: .82rem;
}
div[data-testid="stMetricValue"] {
    font-weight: 620;
}
div[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, .2);
}
div[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}
div[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .28rem .45rem;
    border-radius: 4px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 5px;
    min-height: 2.45rem;
    font-weight: 550;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(128, 128, 128, .2);
    border-radius: 6px;
    overflow: hidden;
}
hr {
    margin: 1.35rem 0;
    border-color: rgba(128, 128, 128, .2);
}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1> MarketLab</h1><p>Multi-stock classification </p></div>',unsafe_allow_html=True)


from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title='MarketLab | Stock Intelligence',
    page_icon='📈',
    layout='wide'
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.3rem 1.5rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    background: linear-gradient(
        120deg,
        rgba(20,120,120,.12),
        rgba(80,90,170,.08)
    );
}

.metric-card {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
}

.small-muted {
    color: gray;
    font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 MarketLab</h1>
    <p>Multi-stock classification · market research · virtual portfolio</p>
    <span class="small-muted">
        Experimental analytics only — not investment advice.
    </span>
</div>
""", unsafe_allow_html=True)


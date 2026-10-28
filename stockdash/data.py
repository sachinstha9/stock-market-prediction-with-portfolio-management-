from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf


def download_history(tickers, period="10y", interval="1d"):
    out = {}
    for ticker in dict.fromkeys(t.upper().strip() for t in tickers if t.strip()):
        frame = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                            progress=False, threads=False)
        
        if frame is None or frame.empty:
            continue

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}

        if not required.issubset(frame.columns):
            continue

        frame = frame[list(required)].copy()
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])

        if len(frame) > 60:
            out[ticker] = frame

    if not out:
        raise RuntimeError("No usable price history returned by yfinance.")
    
    return out

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

def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy().sort_index()
    close, volume = df["Close"], df["Volume"].replace(0, np.nan)

    for n in (1, 5, 10, 20):
        df[f"return_{n}d"] = close.pct_change(n)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, min_periods=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    for n in (10, 20, 50):
        df[f"sma_ratio_{n}"] = close / close.rolling(n, min_periods=n).mean() - 1

    df["ema_ratio_12"] = close / ema12 - 1
    df["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    df["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
    daily = close.pct_change()
    df["volatility_10"] = daily.rolling(10, min_periods=10).std() * np.sqrt(252)
    df["volatility_20"] = daily.rolling(20, min_periods=20).std() * np.sqrt(252)
    df["high_low_range"] = (df["High"] - df["Low"]) / close
    df["close_open_return"] = close / df["Open"] - 1
    
    return df.replace([np.inf, -np.inf], np.nan)


def make_supervised(prices_by_ticker, horizon=5, mode="UP/DOWN", sideways_band=0.01):
    """Labels use future close only for targets, never as model features."""
    if horizon not in (1, 5, 10, 20):
        raise ValueError("horizon must be one of 1, 5, 10, 20")
    frames = []
    for ticker, prices in prices_by_ticker.items():
        df = add_features(prices)
        future_return = df["Close"].shift(-horizon) / df["Close"] - 1
        df["target_return"] = future_return
        if mode == "UP/DOWN":
            df["target"] = (future_return > 0).astype("int8")
        elif mode == "UP/DOWN/SIDEWAYS":
            df["target"] = np.select([future_return > sideways_band, future_return < -sideways_band], [1, -1], default=0)
        else:
            raise ValueError("Unsupported classification mode")
        df["ticker"] = ticker
        # The final horizon rows have no known target and must not be trained.
        df.loc[future_return.isna(), "target"] = np.nan
        frames.append(df)
    combined = pd.concat(frames).sort_index()
    combined = combined.dropna(subset=FEATURES + ["target", "target_return"])
    combined["target"] = combined["target"].astype(int)
    return combined

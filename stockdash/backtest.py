from __future__ import annotations

import numpy as np
import pandas as pd

def performance_metrics(equity: pd.Series, daily_returns: pd.Series):
    equity = equity.dropna()
    r = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    peak = equity.cummax()
    drawdown = equity / peak - 1
    sharpe = float(np.sqrt(252) * r.mean() / r.std(ddof=1)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    return {"total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_drawdown": float(drawdown.min()), "sharpe": sharpe}

def backtest_signals(prices, signals, initial_cash=10000,
                     transaction_cost_bps=10, slippage_bps=5, position_fraction=0.95):
 
    close = prices["Close"].astype(float).sort_index()
 
    signal = signals.reindex(close.index).fillna(0).astype(int).clip(0, 1)
    position = signal.shift(2).fillna(0) * position_fraction
 
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps + slippage_bps) / 10000
 
    asset_returns = close.pct_change().fillna(0)
    strategy_returns = position * asset_returns - cost

    equity = initial_cash * (1 + strategy_returns).cumprod()
    buy_hold = initial_cash * (1 + asset_returns).cumprod()
 
    result = pd.DataFrame({
        "close": close,
        "signal": signal,
        "position": position,
        "asset_return": asset_returns,
        "strategy_return": strategy_returns,
        "equity": equity,
        "buy_hold": buy_hold
    })
 
    metrics = {
        "strategy": performance_metrics(equity, strategy_returns),
        "buy_hold": performance_metrics(buy_hold, asset_returns),
        "transactions": int((turnover > 0).sum()),
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps
    }
 
    return result, metrics
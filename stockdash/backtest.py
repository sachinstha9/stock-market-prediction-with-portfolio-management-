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
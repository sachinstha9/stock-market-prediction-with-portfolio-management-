
import numpy as np
import pandas as pd
import pytest

from stockdash.data import add_features, make_supervised
from stockdash.modeling import walk_forward_splits
from stockdash.backtest import backtest_signals
from stockdash.portfolio import PortfolioDB


def prices(n=260, start='2020-01-01', seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)

    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))

    return pd.DataFrame({
        'Open': close * (1 + rng.normal(0, 0.002, n)),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': rng.integers(10000, 100000, n)
    }, index=idx)


def test_features_are_causal():
    p = prices()
    a = add_features(p)

    altered = p.copy()
    altered.loc[altered.index[-1], 'Close'] *= 3
    b = add_features(altered)

    pd.testing.assert_series_equal(
        a.iloc[:-1]['return_5d'],
        b.iloc[:-1]['return_5d']
    )

    pd.testing.assert_series_equal(
        a.iloc[:-1]['rsi_14'],
        b.iloc[:-1]['rsi_14']
    )


def test_supervised_targets_shift_future_and_drop_tail():
    p = prices(100)
    d = make_supervised({'TEST': p}, horizon=5)

    expected = (p.Close.shift(-5) / p.Close - 1 > 0).astype(int)

    assert d.index.max() <= p.index[-6]
    assert d.iloc[0].target == expected.loc[d.index[0]]
    assert 'target_return' not in d.columns


def test_walk_forward_is_chronological_and_purged():
    p1 = prices(320, seed=1)
    p2 = prices(320, seed=2)

    d = make_supervised({'A': p1, 'B': p2}, horizon=5)
    dates = pd.to_datetime(d.index)

    for tr, te in walk_forward_splits(d, 5, n_splits=3):
        assert dates[tr].max() < dates[te].min()
        assert (dates[te].min() - dates[tr].max()).days >= 5
        assert set(dates[tr]).isdisjoint(set(dates[te]))


def test_backtest_lags_signal_and_costs():
    p = prices(30)
    sig = pd.Series(1, index=p.index)

    no_cost, _ = backtest_signals(
        p, sig,
        transaction_cost_bps=0,
        slippage_bps=0
    )

    with_cost, _ = backtest_signals(
        p, sig,
        transaction_cost_bps=10,
        slippage_bps=5
    )

    assert no_cost.position.iloc[0] == 0
    assert with_cost.equity.iloc[-1] <= no_cost.equity.iloc[-1]


def test_portfolio_buy_sell_cash_and_pnl(tmp_path):
    db = PortfolioDB(tmp_path / 'test.sqlite')

    assert db.cash() == 100000

    db.trade('ABC', 'BUY', 10, 100, fees=2)

    assert db.cash() == pytest.approx(98998)
    assert db.holdings().iloc[0].quantity == pytest.approx(10)

    result = db.trade('ABC', 'SELL', 4, 120, fees=1)

    assert result['realized_pnl'] == pytest.approx(78.2)
    assert db.holdings().iloc[0].quantity == pytest.approx(6)

    with pytest.raises(ValueError):
        db.trade('ABC', 'SELL', 7, 120)

    with pytest.raises(ValueError):
        db.trade('XYZ', 'BUY', 10000, 100)

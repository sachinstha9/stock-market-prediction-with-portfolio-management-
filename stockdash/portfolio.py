
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK(id=1),
    cash REAL NOT NULL
);

INSERT OR IGNORE INTO account(id, cash) VALUES(1, 100000.0);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL CHECK(quantity > 0),
    price REAL NOT NULL CHECK(price > 0),
    fees REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_class INTEGER NOT NULL,
    probabilities TEXT,
    actual_class INTEGER,
    actual_return REAL
);
"""


class PortfolioDB:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as c:
            c.executescript(SCHEMA)

    def connect(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def cash(self):
        with self.connect() as c:
            return float(
                c.execute("SELECT cash FROM account WHERE id=1").fetchone()[0]
            )

    def holdings(self):
        tx = self.transactions()

        if tx.empty:
            return pd.DataFrame(
                columns=["ticker", "quantity", "cost_basis"]
            )

        rows = []

        for ticker, g in tx.groupby("ticker"):
            qty = 0.0
            basis = 0.0

            for r in g.sort_values("ts").itertuples():
                q = float(r.quantity)
                p = float(r.price)
                fee = float(r.fees)

                if r.side == "BUY":
                    qty += q
                    basis += q * p + fee
                else:
                    avg = basis / qty if qty else 0
                    basis -= avg * q
                    qty -= q

            if qty > 1e-8:
                rows.append({
                    "ticker": ticker,
                    "quantity": qty,
                    "cost_basis": basis,
                    "average_cost": basis / qty,
                })

        return pd.DataFrame(rows)

    def transactions(self):
        with self.connect() as c:
            return pd.read_sql_query(
                "SELECT * FROM transactions ORDER BY id", c
            )

    def trade(self, ticker, side, quantity, price, fees=0):
        ticker = ticker.upper().strip()
        side = side.upper()
        quantity = float(quantity)
        price = float(price)
        fees = float(fees)

        if (
            not ticker
            or side not in ("BUY", "SELL")
            or quantity <= 0
            or price <= 0
            or fees < 0
        ):
            raise ValueError("Invalid trade details")

        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")

            cash = float(
                c.execute(
                    "SELECT cash FROM account WHERE id=1"
                ).fetchone()[0]
            )

            existing = c.execute(
                """
                SELECT COALESCE(
                    SUM(CASE side
                        WHEN 'BUY' THEN quantity
                        ELSE -quantity
                    END), 0
                )
                FROM transactions
                WHERE ticker=?
                """,
                (ticker,),
            ).fetchone()[0]

            if side == "SELL" and quantity > existing + 1e-8:
                raise ValueError(
                    "Cannot sell more shares than currently held"
                )

            realized = 0.0

            if side == "SELL":
                buys = c.execute(
                    """
                    SELECT quantity, price, fees
                    FROM transactions
                    WHERE ticker=? AND side='BUY'
                    ORDER BY id
                    """,
                    (ticker,),
                ).fetchall()

                sells = c.execute(
                    """
                    SELECT quantity
                    FROM transactions
                    WHERE ticker=? AND side='SELL'
                    ORDER BY id
                    """,
                    (ticker,),
                ).fetchall()

                prior_sold = sum(r[0] for r in sells)
                remaining = quantity
                sold_before = prior_sold

                for b in buys:
                    available = max(
                        0.0, b["quantity"] - sold_before
                    )
                    sold_before = max(
                        0.0, sold_before - b["quantity"]
                    )

                    take = min(available, remaining)

                    if take:
                        realized += (
                            take * (price - b["price"])
                            - b["fees"] * take / b["quantity"]
                        )

                    remaining -= take

                    if remaining <= 1e-8:
                        break

                realized -= fees

            if side == "BUY":
                new_cash = cash - quantity * price - fees
            else:
                new_cash = cash + quantity * price - fees

            if new_cash < -1e-8:
                raise ValueError("Insufficient virtual cash")

            c.execute(
                """
                INSERT INTO transactions(
                    ticker, side, quantity, price, fees, realized_pnl
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (ticker, side, quantity, price, fees, realized),
            )

            c.execute(
                "UPDATE account SET cash=? WHERE id=1",
                (new_cash,),
            )

        return {"cash": new_cash, "realized_pnl": realized}

    def add_watch(self, ticker):
        with self.connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO watchlist(ticker) VALUES(?)",
                (ticker.upper().strip(),),
            )

    def remove_watch(self, ticker):
        with self.connect() as c:
            c.execute(
                "DELETE FROM watchlist WHERE ticker=?",
                (ticker.upper().strip(),),
            )

    def watchlist(self):
        with self.connect() as c:
            return [
                r[0]
                for r in c.execute(
                    "SELECT ticker FROM watchlist ORDER BY ticker"
                ).fetchall()
            ]

    def record_prediction(
        self, ticker, horizon, predicted_class, probabilities
    ):
        with self.connect() as c:
            c.execute(
                """
                INSERT INTO prediction_history(
                    ticker, horizon, predicted_class, probabilities
                )
                VALUES(?, ?, ?, ?)
                """,
                (
                    ticker,
                    int(horizon),
                    int(predicted_class),
                    json.dumps(probabilities),
                ),
            )

    def predictions(self):
        with self.connect() as c:
            return pd.read_sql_query(
                "SELECT * FROM prediction_history ORDER BY id DESC", c
            )

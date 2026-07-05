"""
data/db.py
SQLite signal database — schema, ingest, and query helpers.
Receives JSON payloads from TradingView webhook alerts fired by SR-Probability v9fix.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "signals.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,          -- ISO8601 UTC bar-close timestamp
            symbol        TEXT    NOT NULL,           -- XAUUSD, USDJPY
            timeframe     TEXT    NOT NULL DEFAULT '5m',
            direction     TEXT    NOT NULL,           -- BUY | SELL
            trigger_type  TEXT    NOT NULL,           -- Rejection | Retest Hold | Break & Retest
            score         REAL    NOT NULL,           -- Pine probability score 0-100
            ftr_confirmed INTEGER NOT NULL DEFAULT 0, -- 1 if +FTR tag present
            session       TEXT    NOT NULL,           -- A (pre-market) | B (late) | NONE
            zone_top      REAL,
            zone_bot      REAL,
            zone_touches  INTEGER,
            entry_price   REAL,
            sl_price      REAL,
            tp_price      REAL,
            close_price   REAL,                      -- filled post-trade
            outcome       TEXT,                      -- WIN | LOSS | SCRATCH | OPEN
            pnl_r         REAL,                      -- P&L in R-multiples
            raw_json      TEXT                       -- full original payload
        );

        CREATE TABLE IF NOT EXISTS regime_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            regime      INTEGER NOT NULL,            -- 0 = low-vol, 1 = trending, 2 = high-vol
            atr_14      REAL,
            vol_ratio   REAL
        );

        CREATE INDEX IF NOT EXISTS idx_signals_ts     ON signals(ts);
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome);
        """)


def ingest_webhook(payload: dict) -> int:
    """
    Insert a webhook payload from TradingView into the signals table.
    Expected payload keys (all from Pine alert message template):
        ts, symbol, direction, trigger_type, score, ftr_confirmed,
        session, zone_top, zone_bot, zone_touches, entry_price, sl_price, tp_price
    Returns the new row id.
    """
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO signals
              (ts, symbol, timeframe, direction, trigger_type, score,
               ftr_confirmed, session, zone_top, zone_bot, zone_touches,
               entry_price, sl_price, tp_price, outcome, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?)
        """, (
            payload.get("ts", datetime.utcnow().isoformat()),
            payload.get("symbol", "XAUUSD"),
            payload.get("timeframe", "5m"),
            payload["direction"],
            payload["trigger_type"],
            float(payload["score"]),
            int(payload.get("ftr_confirmed", 0)),
            payload.get("session", "NONE"),
            payload.get("zone_top"),
            payload.get("zone_bot"),
            payload.get("zone_touches"),
            payload.get("entry_price"),
            payload.get("sl_price"),
            payload.get("tp_price"),
            json.dumps(payload),
        ))
        return cur.lastrowid


def update_outcome(signal_id: int, close_price: float, outcome: str, pnl_r: float) -> None:
    with get_connection() as conn:
        conn.execute("""
            UPDATE signals
            SET close_price=?, outcome=?, pnl_r=?
            WHERE id=?
        """, (close_price, outcome, pnl_r, signal_id))


def load_signals(
    symbol: Optional[str] = None,
    outcome_filter: Optional[str] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """Load signals into a DataFrame, optional filters applied server-side."""
    clauses, params = [], []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if outcome_filter:
        clauses.append("outcome = ?")
        params.append(outcome_filter)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM signals {where} ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts"])
    return df


def log_regime(ts: str, symbol: str, regime: int, atr_14: float, vol_ratio: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO regime_log (ts, symbol, regime, atr_14, vol_ratio) VALUES (?,?,?,?,?)",
            (ts, symbol, regime, atr_14, vol_ratio),
        )

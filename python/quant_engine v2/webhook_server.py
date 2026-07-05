"""
webhook_server.py
Flask micro-server receiving TradingView JSON webhook payloads.
Run alongside Streamlit: python webhook_server.py (port 5001)

TradingView Alert Message format (set in Pine alert dialog):
{
  "ticker":        "{{ticker}}",
  "timeframe":     "{{interval}}",
  "direction":     "BUY",
  "trigger_type":  "Rejection +FTR",
  "score":         84.0,
  "ftr_confirmed": true,
  "session":       "A",
  "zone_top":      4196.948,
  "zone_bot":      4171.940,
  "entry_price":   4172.33,
  "sl_price":      4168.00,
  "tp_price":      4185.00,
  "bar_time":      "{{time}}"
}
"""

from flask import Flask, request, jsonify
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from data.db import init_db, get_connection

app = Flask(__name__)
init_db()

REQUIRED_FIELDS = {"ticker", "direction", "trigger_type", "score"}


def _insert_signal(payload: dict) -> int:
    sql = """
    INSERT INTO signals
        (ts, symbol, timeframe, direction, trigger_type, score,
         ftr_confirmed, session, zone_top, zone_bot, zone_touches,
         entry_price, sl_price, tp_price, outcome, raw_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cur = conn.execute(sql, (
            payload.get("bar_time", now),
            payload.get("ticker", "UNKNOWN"),
            payload.get("timeframe", "5m"),
            payload["direction"],
            payload["trigger_type"],
            float(payload["score"]),
            int(bool(payload.get("ftr_confirmed", False))),
            payload.get("session", ""),
            payload.get("zone_top"),
            payload.get("zone_bot"),
            payload.get("zone_touches"),
            payload.get("entry_price"),
            payload.get("sl_price"),
            payload.get("tp_price"),
            "OPEN",
            json.dumps(payload),
        ))
        return cur.lastrowid


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "invalid JSON"}), 400

        missing = REQUIRED_FIELDS - set(payload.keys())
        if missing:
            return jsonify({"error": f"missing fields: {missing}"}), 400

        sig_id = _insert_signal(payload)
        return jsonify({"status": "ok", "signal_id": sig_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive", "ts": datetime.utcnow().isoformat()}), 200


if __name__ == "__main__":
    print("Webhook server listening on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)

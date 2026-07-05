# SR-Probability Quantitative Engine

External Python quantitative stack for the SR-Probability Engine v9fix TradingView indicator.
Runs locally on Xubuntu Linux (i3 CPU / 12 GB RAM) or deploys to Streamlit Cloud.

## Architecture

```
TradingView (Pine Script v9fix)
    │
    │  JSON webhook alerts (POST /webhook)
    ▼
webhook_server.py  ──► signals.db (SQLite)
                                │
                                ▼
                    Streamlit Dashboard (app.py)
                    ├── Overview          (equity curve, win rate, score distribution)
                    ├── Signal Log        (trade log, inline outcome update)
                    ├── Session Analysis  (24h win-rate heatmap, Mann-Whitney test)
                    ├── Calibration       (Platt scaling, isotonic regression, Brier score)
                    ├── Kelly Sizing      (fractional Kelly by segment, Wilson CI)
                    ├── Regime            (3-state HMM, ATR + volume features)
                    └── Optimizer         (Optuna TPE, walk-forward validated)
```

## Prerequisites

- Python 3.11+
- pip

## Local Installation (Xubuntu / Linux)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/sr-prob-engine.git
cd sr-prob-engine

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install system dependencies (run once)
chmod +x setup.sh && sudo ./setup.sh

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Initialize the database
python -c "from data.db import init_db; init_db(); print('DB ready')"

# 6. Run the Streamlit dashboard
streamlit run app.py

# 7. (Optional) Run the webhook server in a separate terminal
python webhook_server.py
```

Dashboard runs at: http://localhost:8501
Webhook receiver: http://localhost:5055/webhook

## TradingView Webhook Configuration

In TradingView, create an alert on the SR-Probability v9fix indicator.

Set the **Webhook URL** to:
```
http://YOUR_LOCAL_IP:5055/webhook
```
(Use ngrok or a VPS reverse proxy to expose this publicly for TradingView.)

Set the **Message** to this JSON template:
```json
{
  "ts":           "{{time}}",
  "symbol":       "{{ticker}}",
  "timeframe":    "{{interval}}",
  "direction":    "BUY",
  "trigger_type": "Rejection",
  "score":        75,
  "ftr_confirmed": 1,
  "session":      "A",
  "zone_top":     0,
  "zone_bot":     0,
  "zone_touches": 0,
  "entry_price":  {{close}},
  "sl_price":     0,
  "tp_price":     0
}
```

Note: Replace `direction`, `trigger_type`, `score`, `ftr_confirmed`, and `session`
with the actual values from your Pine alert variables. The `{{close}}` macro
is the only TradingView built-in used here.

## Streamlit Cloud Deployment

1. Push this repo to GitHub (ensure `signals.db` and `assets/*.pkl` are in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file path to `app.py`.
4. Streamlit Cloud automatically runs `setup.sh` before installing requirements.

Note: Streamlit Cloud does not persist file-system state between restarts.
For production signal logging, switch to PostgreSQL by setting `DATABASE_URL`
in `.streamlit/secrets.toml` and updating `data/db.py` accordingly.

## Project Structure

```
sr-prob-engine/
├── app.py                        # Streamlit dashboard (single-file, 7 pages)
├── webhook_server.py             # Flask webhook receiver
├── requirements.txt
├── setup.sh                      # System dependency bootstrap
├── .gitignore
├── .streamlit/
│   ├── config.toml               # Theme, port, performance settings
│   └── secrets.toml.example      # Copy to secrets.toml and fill in
├── data/
│   ├── __init__.py
│   ├── db.py                     # SQLite schema, ingest, query helpers
│   └── .gitkeep
├── quant/
│   ├── __init__.py
│   ├── signal_generator.py       # Python reimplementation of Pine v9fix logic
│   ├── session_analysis.py       # 24h win-rate heatmap, chi-square, Mann-Whitney
│   ├── calibration.py            # Platt scaling, isotonic regression, Brier score CV
│   ├── kelly.py                  # Fractional Kelly, Wilson CI, growth rate
│   ├── regime.py                 # 3-state Gaussian HMM, regime inference
│   └── optimizer.py              # Optuna TPE study, walk-forward splits
└── assets/
    └── .gitkeep                  # Trained model .pkl files saved here (not committed)
```

## Strategy Context

The quantitative engine extends the SR-Probability v9fix Pine Script indicator,
which implements a session-boundary liquidity sweep strategy for Gold (XAU/USD)
and USD/JPY. Key components:

- **Session Gate**: Signals restricted to PH pre-market (12:00-09:00 AM) and
  late session (19:00-23:59 PM), implemented as UTC window comparisons.
- **FTR Filter**: Higher Low (BUY) and Lower High (SELL) structural confirmation
  coded as `low > ta.lowest(low, N)[1]` and `high < ta.highest(high, N)[1]`.
- **Session Volume Inversion**: Low volume scores higher during session windows
  (confirming sweep exhaustion), the inverse of standard volume scoring.
- **Global Cooldown**: `lastAnySignalBar` evaluated inline inside zone loops
  to prevent multi-zone same-bar firing.

## Six Pre-Trade Conditions (must all pass before entry)

1. Current time is within the active session window.
2. Price is at a drawn zone from the indicator panel.
3. For BUY: current low > prior swing low (Higher Low intact).
4. For SELL: current high < prior swing high (Lower High intact).
5. Volume is below 20-bar MA (confirming sweep, not breakout).
6. Indicator has printed a signal triangle on bar close.

## Dependencies

| Package       | Purpose                              |
|---------------|--------------------------------------|
| streamlit     | Dashboard UI                         |
| pandas/numpy  | Data manipulation                    |
| scipy         | Chi-square, Mann-Whitney, statistics |
| scikit-learn  | Calibration, isotonic regression     |
| scikit-learn  | Regime detection (GaussianMixture)   |
| optuna        | Bayesian parameter optimization      |
| optuna        | Bayesian parameter optimization (TPE)|
| plotly        | Interactive charts                   |
| flask         | Webhook server                       |
| sqlalchemy    | Database abstraction                 |
| joblib        | Model persistence                    |

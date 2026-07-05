"""
app.py
SR-Probability Quantitative Engine — Streamlit Dashboard
Borax | Xubuntu i3/12GB Local | Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.db import init_db, load_signals, get_connection

st.set_page_config(
    page_title="SR-Prob Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: dark terminal aesthetic ──────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080d0e; color: #b8cdd0;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; font-size: 13px;
}
[data-testid="stSidebar"] { background-color: #0b1214; border-right: 1px solid #182325; }
[data-testid="stSidebar"] label { color: #5d8a8f; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="metric-container"] { background: #0e1719; border: 1px solid #182325; border-radius: 3px; padding: 10px; }
[data-testid="metric-container"] label { color: #5d8a8f; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00c9a7; font-size: 22px; font-weight: 700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 11px; }
h1 { color: #00c9a7; font-size: 15px; letter-spacing: 0.06em; font-weight: 700; border-bottom: 1px solid #182325; padding-bottom: 6px; margin-bottom: 12px; }
h2 { color: #00c9a7; font-size: 12px; letter-spacing: 0.05em; }
h3 { color: #5d8a8f; font-size: 11px; }
.stDataFrame { border: 1px solid #182325; border-radius: 3px; }
.stButton > button { background: #0b1214; border: 1px solid #00c9a7; color: #00c9a7; font-family: monospace; font-size: 10px; border-radius: 2px; letter-spacing: 0.08em; padding: 4px 12px; }
.stButton > button:hover { background: #00c9a7; color: #080d0e; }
.stTabs [data-baseweb="tab-list"] { background: #0b1214; border-bottom: 1px solid #182325; }
.stTabs [data-baseweb="tab"] { color: #5d8a8f; font-size: 10px; letter-spacing: 0.08em; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00c9a7; border-bottom: 2px solid #00c9a7; }
hr { border-color: #182325; margin: 12px 0; }
.badge {
    display: inline-block; padding: 3px 12px; border-radius: 2px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
}
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
.pulse { animation: pulse 2.4s infinite; }
</style>
""", unsafe_allow_html=True)

init_db()

# ── Plotly base layout (reusable) ─────────────────────────────────────────────
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="#080d0e", plot_bgcolor="#080d0e",
        font=dict(family="monospace", size=11, color="#5d8a8f"),
        xaxis=dict(gridcolor="#182325", zerolinecolor="#182325"),
        yaxis=dict(gridcolor="#182325", zerolinecolor="#182325"),
        legend=dict(bgcolor="#080d0e", bordercolor="#182325"),
        margin=dict(l=45, r=15, t=25, b=35),
    )
    base.update(kwargs)
    return base

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SR-PROB ENGINE")
    st.markdown("---")
    page = st.radio("", [
        "Overview", "Signal Log", "Session Analysis",
        "Calibration", "Kelly Sizing", "Regime", "Optimizer"
    ], label_visibility="collapsed")
    st.markdown("---")
    sym_opt = st.selectbox("SYMBOL", ["XAUUSD", "USDJPY", "ALL"])
    st.markdown("---")
    st.markdown('<span style="color:#182325;font-size:9px;">Borax Local · Xubuntu · i3/12GB</span>',
                unsafe_allow_html=True)

sym = None if sym_opt == "ALL" else sym_opt

# ── Cached data load (30s TTL, lightweight on i3) ────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _signals(symbol):
    return load_signals(symbol=symbol, limit=5000)

signals = _signals(sym)
closed  = signals[signals["outcome"].isin(["WIN", "LOSS"])].copy() if not signals.empty else pd.DataFrame()

def _metrics(df):
    if df.empty:
        return dict(n=0, wr=0, avg_r=0, total_r=0, max_dd=0, expectancy=0)
    n    = len(df)
    wr   = (df["outcome"] == "WIN").mean()
    pnl  = df["pnl_r"].values
    avg_r = pnl.mean()
    cumul = np.cumsum(pnl)
    dd    = (np.maximum.accumulate(cumul) - cumul).max() if len(cumul) else 0
    exp   = wr * df[df["outcome"]=="WIN"]["pnl_r"].mean() + \
            (1-wr) * df[df["outcome"]=="LOSS"]["pnl_r"].mean() \
            if len(df[df["outcome"]=="WIN"]) and len(df[df["outcome"]=="LOSS"]) else 0
    return dict(n=n, wr=wr, avg_r=avg_r, total_r=cumul[-1] if len(cumul) else 0,
                max_dd=dd, expectancy=round(exp, 3))

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "Overview":
    st.markdown("# SYSTEM OVERVIEW")

    # Regime badge
    rctx = st.session_state.get("regime_ctx")
    if rctx:
        c = rctx["color"]; lb = rctx["label"]; mod = rctx["score_threshold_modifier"]
        st.markdown(
            f'<span class="badge pulse" style="background:{c}18;border:1px solid {c};color:{c};">'
            f'REGIME: {lb}</span>&nbsp;'
            f'<span style="color:#5d8a8f;font-size:10px;">Threshold adj: '
            f'<b style="color:{c}">{mod:+d}</b> pts</span>',
            unsafe_allow_html=True)
    st.markdown("---")

    m = _metrics(closed)
    cols = st.columns(6)
    cols[0].metric("CLOSED", m["n"])
    cols[1].metric("WIN RATE", f"{m['wr']:.1%}")
    cols[2].metric("AVG R", f"{m['avg_r']:+.3f}")
    cols[3].metric("TOTAL R", f"{m['total_r']:+.2f}")
    cols[4].metric("MAX DD", f"{m['max_dd']:.2f}R")
    cols[5].metric("EXPECTANCY", f"{m['expectancy']:+.3f}R")
    st.markdown("---")

    if not closed.empty:
        t1, t2, t3 = st.tabs(["EQUITY", "WIN RATE BY TYPE", "SCORE DIST"])

        with t1:
            eq = closed["pnl_r"].cumsum().reset_index(drop=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=eq, mode="lines",
                line=dict(color="#00c9a7", width=1.4),
                fill="tozeroy", fillcolor="rgba(0,201,167,0.04)",
            ))
            fig.update_layout(height=260, **dark_layout(
                yaxis=dict(gridcolor="#182325", title="Cumulative R", zerolinecolor="#182325")
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with t2:
            if "trigger_type" in closed.columns:
                wr_df = (closed.groupby("trigger_type")
                         .apply(lambda g: pd.Series({
                             "win_rate": (g["outcome"]=="WIN").mean(),
                             "n": len(g), "avg_r": g["pnl_r"].mean(),
                         })).reset_index())
                fig2 = go.Figure(go.Bar(
                    x=wr_df["trigger_type"], y=wr_df["win_rate"],
                    text=[f"n={n}" for n in wr_df["n"]],
                    textposition="outside",
                    marker_color=["#00c9a7" if w >= 0.55 else "#d63031" for w in wr_df["win_rate"]],
                ))
                fig2.update_layout(height=250, **dark_layout(
                    yaxis=dict(gridcolor="#182325", tickformat=".0%", zerolinecolor="#182325")
                ))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        with t3:
            if "score" in closed.columns:
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(
                    x=closed["score"], nbinsx=18, name="All",
                    marker_color="#182325", opacity=0.9))
                fig3.add_trace(go.Histogram(
                    x=closed[closed["outcome"]=="WIN"]["score"], nbinsx=18, name="Wins",
                    marker_color="#00c9a7", opacity=0.6))
                fig3.update_layout(barmode="overlay", height=250, **dark_layout(
                    xaxis=dict(gridcolor="#182325", title="Pine Score", zerolinecolor="#182325"),
                    yaxis=dict(gridcolor="#182325", title="Count", zerolinecolor="#182325"),
                ))
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No closed trades yet. Log signals via the Signal Log page.")

# =============================================================================
# PAGE: SIGNAL LOG
# =============================================================================
elif page == "Signal Log":
    st.markdown("# SIGNAL LOG")

    c1, c2, c3 = st.columns([4, 1, 1])
    c2.markdown("")
    if c3.button("REFRESH"):
        st.cache_data.clear(); st.rerun()

    if signals.empty:
        st.warning("No signals. Set up TradingView webhook -> http://localhost:5001/webhook")
    else:
        view_cols = ["id","ts","symbol","direction","trigger_type","score",
                     "ftr_confirmed","session","entry_price","sl_price","tp_price",
                     "outcome","pnl_r"]
        available = [c for c in view_cols if c in signals.columns]
        st.dataframe(signals[available].head(200), use_container_width=True, height=460)

        st.markdown("---")
        st.markdown("## UPDATE OUTCOME")
        with st.form("outcome_form"):
            f1, f2, f3, f4 = st.columns(4)
            sig_id   = f1.number_input("Signal ID", min_value=1, step=1)
            close_px = f2.number_input("Close Price", step=0.001, format="%.3f")
            outcome  = f3.selectbox("Outcome", ["WIN", "LOSS", "SCRATCH"])
            pnl_r    = f4.number_input("P&L (R)", step=0.01, format="%.2f")
            if st.form_submit_button("SAVE OUTCOME"):
                sql = "UPDATE signals SET outcome=?, pnl_r=?, close_price=? WHERE id=?"
                with get_connection() as conn:
                    conn.execute(sql, (outcome, pnl_r, close_px, int(sig_id)))
                st.cache_data.clear()
                st.success(f"Signal {int(sig_id)} -> {outcome} / {pnl_r}R")
                st.rerun()

# =============================================================================
# PAGE: SESSION ANALYSIS
# =============================================================================
elif page == "Session Analysis":
    st.markdown("# SESSION WINDOW ANALYSIS")
    st.markdown("Empirical win-rate heatmap across 24h. Chi-square significance vs overall baseline.")

    if len(closed) < 20:
        st.warning(f"Requires >= 20 closed signals. Have: {len(closed)}")
    else:
        from quant.session_analysis import (
            session_winrate_heatmap, heatmap_matrix,
            best_session_windows, compare_session_vs_nonSession
        )

        @st.cache_data(ttl=120, show_spinner=False)
        def _heatmap(df_hash):
            return session_winrate_heatmap(closed, min_bucket_size=3)

        hmap = _heatmap(len(closed))

        if hmap.empty:
            st.info("Not enough data per time bucket yet.")
        else:
            pivot = heatmap_matrix(hmap)
            if not pivot.empty:
                fig = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[f":{m:02d}" for m in pivot.columns],
                    y=[f"{h:02d}:xx PH" for h in pivot.index],
                    colorscale=[[0, "#d63031"], [0.4, "#182325"], [1, "#00c9a7"]],
                    zmid=closed["outcome"].eq("WIN").mean(),
                    colorbar=dict(title="Win Rate", tickformat=".0%",
                                  tickfont=dict(color="#5d8a8f", size=10)),
                    hoverongaps=False,
                    hovertemplate="PH %{y}%{x}<br>Win Rate: %{z:.1%}<extra></extra>",
                ))
                fig.update_layout(height=380, **dark_layout(
                    title=dict(text="WIN RATE BY PH TIME BUCKET",
                               font=dict(size=11, color="#5d8a8f")),
                    xaxis=dict(gridcolor="#182325", title="Minute (30-min buckets)", zerolinecolor="#182325"),
                    yaxis=dict(gridcolor="#182325", title="PH Hour", zerolinecolor="#182325"),
                ))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("## SIGNIFICANT BUCKETS")
                best = best_session_windows(hmap, top_n=8, min_n=3)
                if not best.empty:
                    st.dataframe(best, use_container_width=True)
                else:
                    st.info("No statistically significant buckets yet.")

            with col2:
                st.markdown("## SESSION vs NON-SESSION")
                a_start = st.number_input("UTC Window A Start (HHMM)", value=1600, step=30)
                a_end   = st.number_input("UTC Window A End (HHMM)",   value=100,  step=30)
                b_start = st.number_input("UTC Window B Start (HHMM)", value=1100, step=30)
                b_end   = st.number_input("UTC Window B End (HHMM)",   value=1559, step=30)

                if st.button("RUN MANN-WHITNEY TEST"):
                    result = compare_session_vs_nonSession(
                        closed, int(a_start), int(a_end), int(b_start), int(b_end)
                    )
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        superior = result["session_superior"]
                        badge_c = "#00c9a7" if superior else "#d63031"
                        verdict = "STATISTICALLY SUPERIOR" if superior else "NOT SIGNIFICANT"
                        st.markdown(
                            f'<span class="badge" style="background:{badge_c}18;'
                            f'border:1px solid {badge_c};color:{badge_c};">{verdict}</span>',
                            unsafe_allow_html=True)
                        cols = st.columns(3)
                        cols[0].metric("Win Rate IN",  f"{result['win_rate_in']:.1%}")
                        cols[1].metric("Win Rate OUT", f"{result['win_rate_out']:.1%}")
                        cols[2].metric("p-value",      f"{result['p_value']:.4f}")
                        st.markdown(
                            f"n(in): **{result['n_in_session']}** | "
                            f"n(out): **{result['n_out_session']}** | "
                            f"Avg R in: **{result['avg_r_in']:+.3f}** | "
                            f"Avg R out: **{result['avg_r_out']:+.3f}**"
                        )

# =============================================================================
# PAGE: CALIBRATION
# =============================================================================
elif page == "Calibration":
    st.markdown("# SIGNAL CALIBRATION")
    st.markdown("Maps Pine heuristic score to true posterior win probability.")

    if len(closed) < 30:
        st.warning(f"Requires >= 30 closed signals. Have: {len(closed)}")
    else:
        if st.button("FIT CALIBRATORS (Platt + Isotonic)"):
            with st.spinner("Fitting..."):
                from quant.calibration import fit_platt, fit_isotonic
                fit_platt(closed); fit_isotonic(closed)
            st.success("Calibrators fitted.")

        try:
            from quant.calibration import calibration_report, cross_validated_brier
            cal_df = calibration_report(closed)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#182325", dash="dash", width=1), name="Perfect"))
            color_map = {"Raw Pine Score": "#5d8a8f", "Platt Scaling": "#fdcb6e",
                         "Isotonic Regression": "#00c9a7"}
            for method, grp in cal_df.groupby("method"):
                fig.add_trace(go.Scatter(
                    x=grp["mean_predicted"], y=grp["fraction_win"],
                    mode="lines+markers", name=method,
                    line=dict(color=color_map.get(method, "#fff"), width=2),
                    marker=dict(size=5),
                ))
            fig.update_layout(height=320, **dark_layout(
                xaxis=dict(gridcolor="#182325", title="Predicted Prob", zerolinecolor="#182325"),
                yaxis=dict(gridcolor="#182325", title="Actual Win Rate", zerolinecolor="#182325"),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            brier = cross_validated_brier(closed)
            b_summary = brier.groupby("method")["brier"].agg(["mean","std"]).round(4)
            b_summary.columns = ["Mean Brier", "Std"]
            st.markdown("## BRIER SCORE (5-FOLD CV)")
            st.dataframe(b_summary)
            st.caption("Lower = better. 0.0 = perfect. 0.25 = uninformed baseline.")

        except FileNotFoundError:
            st.info("Fit calibrators first.")

# =============================================================================
# PAGE: KELLY SIZING
# =============================================================================
elif page == "Kelly Sizing":
    st.markdown("# KELLY POSITION SIZING")

    if len(closed) < 20:
        st.warning(f"Requires >= 20 closed signals. Have: {len(closed)}")
    else:
        from quant.kelly import compute_kelly, kelly_confidence_bounds, expected_growth_rate

        col1, col2 = st.columns(2)
        group_by = col1.selectbox("SEGMENT BY", ["trigger_type", "session", "direction"])
        fraction = col2.selectbox("KELLY FRACTION", ["half", "quarter", "full"])

        kelly_df = compute_kelly(closed, group_by=group_by, fraction=fraction)
        st.dataframe(kelly_df, use_container_width=True)

        st.markdown("---")
        st.markdown("## CONFIDENCE BOUNDS (Wilson 95% CI)")
        ok = kelly_df[kelly_df.get("note", pd.Series(["OK"]*len(kelly_df))).values == "OK"] \
             if "note" in kelly_df.columns else kelly_df

        frac_val = {"half": 0.5, "quarter": 0.25, "full": 1.0}[fraction]
        for _, row in ok.iterrows():
            if "win_rate" not in row or "b_ratio" not in row:
                continue
            bounds = kelly_confidence_bounds(
                p=row["win_rate"], b=row["b_ratio"],
                n=int(row["n_trades"]), fraction=frac_val
            )
            egrp = expected_growth_rate(row["win_rate"], row["b_ratio"], bounds["kelly_central"])
            seg  = row[group_by]
            st.markdown(
                f"**{seg}** -- {fraction}-Kelly: "
                f"`{bounds['kelly_lower']:.4f}` to `{bounds['kelly_upper']:.4f}` "
                f"(central: `{bounds['kelly_central']:.4f}`) | "
                f"p CI: [{bounds['p_lower']:.3f}, {bounds['p_upper']:.3f}] | "
                f"Growth rate G(f): `{egrp:.5f}`"
            )

# =============================================================================
# PAGE: REGIME
# =============================================================================
elif page == "Regime":
    st.markdown("# REGIME DETECTION (3-STATE HMM)")
    st.markdown("Gaussian HMM on ATR-14 (price-normalized) + volume ratio. States: Low-Vol | Trending | High-Vol.")

    uploaded = st.file_uploader("UPLOAD OHLCV CSV (ts, open, high, low, close, volume)", type="csv")
    if uploaded:
        @st.cache_data(show_spinner=False)
        def _load_ohlcv(name, size):
            df = pd.read_csv(uploaded, parse_dates=["ts"]).set_index("ts").sort_index()
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna()

        ohlcv = _load_ohlcv(uploaded.name, uploaded.size)
        st.markdown(f"Loaded **{len(ohlcv):,}** bars -- {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")

        c1, c2 = st.columns(2)
        if c1.button("FIT HMM"):
            with st.spinner("Fitting..."):
                from quant.regime import fit_regime_model
                fit_regime_model(ohlcv)
            st.success("HMM fitted and saved to assets/.")

        if c2.button("PREDICT + VISUALIZE"):
            from quant.regime import predict_regimes, current_regime_summary, REGIME_COLORS, REGIME_LABELS
            regimes = predict_regimes(ohlcv)
            rctx = current_regime_summary(ohlcv)
            st.session_state["regime_ctx"] = rctx

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ohlcv.index, y=ohlcv["close"],
                mode="lines", line=dict(color="#182325", width=0.7), name="Price"
            ))
            for rid, color in REGIME_COLORS.items():
                mask = regimes == rid
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=ohlcv.index[mask], y=ohlcv["close"][mask],
                        mode="markers",
                        marker=dict(color=color, size=2.5, opacity=0.6),
                        name=REGIME_LABELS[rid],
                    ))
            fig.update_layout(height=300, **dark_layout())
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            c = rctx["color"]; lb = rctx["label"]; mod = rctx["score_threshold_modifier"]
            st.markdown(
                f'<span class="badge pulse" style="background:{c}18;border:1px solid {c};color:{c};">'
                f'CURRENT REGIME: {lb}</span> &nbsp;'
                f'<span style="color:#5d8a8f;font-size:10px;">'
                f'ATR%: {rctx["atr_norm"]:.4f} | Vol Ratio: {rctx["vol_ratio"]:.3f} | '
                f'Score Adj: <b style="color:{c}">{mod:+d}</b></span>',
                unsafe_allow_html=True)

            if not closed.empty and "ts" in closed.columns:
                closed_ts = pd.to_datetime(closed["ts"], utc=True)
                regime_at = regimes.reindex(closed_ts, method="nearest")
                cl2 = closed.copy()
                cl2["regime"] = regime_at.values
                wr_r = (cl2.groupby("regime")
                        .apply(lambda g: pd.Series({
                            "label":    REGIME_LABELS.get(int(g.name), str(g.name)),
                            "n":        len(g),
                            "win_rate": (g["outcome"]=="WIN").mean(),
                            "avg_r":    g["pnl_r"].mean(),
                        })).reset_index(drop=True))
                st.markdown("---")
                st.markdown("## WIN RATE BY REGIME")
                st.dataframe(wr_r, use_container_width=False)

# =============================================================================
# PAGE: OPTIMIZER
# =============================================================================
elif page == "Optimizer":
    st.markdown("# PARAMETER OPTIMIZER")
    st.markdown("Optuna TPE Bayesian search with walk-forward validation. Optimizes session windows + strategy params.")

    uploaded_opt = st.file_uploader("UPLOAD OHLCV CSV", type="csv", key="opt")
    if uploaded_opt:
        @st.cache_data(show_spinner=False)
        def _opt_data(name, size):
            df = pd.read_csv(uploaded_opt, parse_dates=["ts"]).set_index("ts").sort_index()
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna()

        ohlcv_opt = _opt_data(uploaded_opt.name, uploaded_opt.size)
        st.markdown(f"**{len(ohlcv_opt):,}** bars loaded.")

        c1, c2, c3 = st.columns(3)
        n_trials  = c1.slider("TRIALS",        20, 300, 100, step=10)
        n_splits  = c2.slider("WALK-FWD FOLDS", 3,   8,   5)
        objective = c3.selectbox("OBJECTIVE", ["calmar", "sharpe"])

        if st.button("RUN OPTIMIZATION"):
            from quant.signal_generator import generate_signals
            from quant.optimizer import create_study, best_params_report

            prog = st.progress(0, "Running Optuna...")
            study = create_study(
                ohlcv_opt, generate_signals,
                n_trials=n_trials, objective_fn=objective,
                n_splits=n_splits,
                storage_url="sqlite:///optuna_study.db",
            )
            prog.progress(100, "Done.")

            st.success(f"Best {objective}: {study.best_value:.4f}")
            st.markdown("## BEST PARAMETERS")
            st.json(study.best_params)

            st.markdown("## TOP 20 TRIALS")
            st.dataframe(best_params_report(study), use_container_width=True)

            st.markdown("---")
            st.markdown("## OPTIMIZATION HISTORY")
            vals = [t.value for t in study.trials if t.value is not None]
            best_so_far = pd.Series(vals).cummax().values
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=vals, mode="lines",
                line=dict(color="#5d8a8f", width=0.8), name="Trial value"))
            fig.add_trace(go.Scatter(y=best_so_far, mode="lines",
                line=dict(color="#00c9a7", width=1.4), name="Best so far"))
            fig.update_layout(height=240, **dark_layout(
                yaxis=dict(gridcolor="#182325", title=objective.capitalize(), zerolinecolor="#182325")
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pump & Dump | Real-Time Detection",
    page_icon="📈",
    layout="wide",
)

# --- Custom CSS (Matching feature.html) ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;900&display=swap');

    /* Global styles */
    .stApp {
        background-color: #0B0F1A;
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }

    /* Main Ticker Styles */
    .ticker-container {
        display: flex;
        justify-content: space-around;
        background-color: #161E2D;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .ticker-item {
        text-align: center;
    }
    .ticker-symbol {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .ticker-price {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38BDF8;
    }

    /* Analyzer Panel Styles */
    .analyzer-card {
        background-color: #161E2D;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        height: 100%;
    }
    .status-normal { color: #34D399; font-weight: bold; }
    .status-warning { color: #F87171; font-weight: bold; }
    
    /* Metrics */
    .metric-label { color: #64748B; font-size: 0.9rem; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; }

    /* Hide Streamlit Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        models = {
            "5s": {
                "rf": joblib.load("random_forest_5s_model.joblib"),
                "scaler": joblib.load("scaler_5s.joblib"),
            },
            "25s": {
                "rf": joblib.load("random_forest_25s_model.joblib"),
                "scaler": joblib.load("scaler_25s.joblib"),
            },
        }
        return models
    except Exception as e:
        st.error(
            f"Error loading models: {e}. Ensure .joblib files are in the root directory."
        )
        return None


models = load_models()


# --- Data Acquisition (Binance via CCXT) ---
def get_recent_trades(symbol="BTC/USDT", minutes=10):
    exchange = ccxt.binance()
    since = exchange.milliseconds() - (minutes * 60 * 1000)
    trades = exchange.fetch_trades(symbol, since=since)

    df = pd.DataFrame(
        trades, columns=["timestamp", "datetime", "side", "price", "amount"]
    )
    df["btc_volume"] = df["price"] * df["amount"]
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    return df


# --- Feature Engineering (Logic from features.py) ---
def compute_live_features(df, time_freq, rolling_freq_val):
    # Only use 'buy' side as per model requirements
    df_buy = df[df["side"] == "buy"].copy()

    if df_buy.empty:
        return None

    # Grouping
    df_resampled = df_buy.resample(time_freq)

    # Feature calculations
    features = {}

    # 1. Rush Orders (Simplified for live stream)
    # Count trades per millisecond/second to find "bunched" orders
    rush_counts = df_buy.groupby(df_buy.index).size()
    rush_binary = (rush_counts > 1).astype(int)
    # Resample rush binary to match time_freq
    features["std_rush_order"] = (
        rush_binary.resample(time_freq)
        .sum()
        .rolling(window=10)
        .std()
        .pct_change()
        .fillna(0)
        .iloc[-1]
    )
    features["avg_rush_order"] = (
        rush_binary.resample(time_freq)
        .sum()
        .rolling(window=10)
        .mean()
        .pct_change()
        .fillna(0)
        .iloc[-1]
    )

    # 2. Volume & Trades
    vol_sum = df_resampled["btc_volume"].sum()
    features["std_trades"] = (
        df_resampled["price"]
        .count()
        .rolling(window=10)
        .std()
        .pct_change()
        .fillna(0)
        .iloc[-1]
    )
    features["std_volume"] = (
        vol_sum.rolling(window=10).std().pct_change().fillna(0).iloc[-1]
    )
    features["avg_volume"] = (
        vol_sum.rolling(window=10).mean().pct_change().fillna(0).iloc[-1]
    )

    # 3. Price
    price_mean = df_resampled["price"].mean()
    features["std_price"] = (
        price_mean.rolling(window=10).std().pct_change().fillna(0).iloc[-1]
    )
    features["avg_price"] = (
        price_mean.rolling(window=10).mean().pct_change().fillna(0).iloc[-1]
    )
    features["avg_price_max"] = (
        df_resampled["price"]
        .max()
        .rolling(window=10)
        .mean()
        .pct_change()
        .fillna(0)
        .iloc[-1]
    )

    # 4. Time components
    now = datetime.now()
    features["hour_sin"] = np.sin(2 * np.pi * now.hour / 23)
    features["hour_cos"] = np.cos(2 * np.pi * now.hour / 23)
    features["minute_sin"] = np.sin(2 * np.pi * now.minute / 59)
    features["minute_cos"] = np.cos(2 * np.pi * now.minute / 59)

    return pd.DataFrame([features])


# --- Main App Interface ---
st.title("⚡ Pump & Dump Detection Dashboard")

# Top Coin Selection for Analyzer
selected_coin = st.sidebar.selectbox(
    "Target Analyzer Coin", ["BTC/USDT", "ETH/USDT", "AAVE/USDT"]
)

# Ticker Row
ticker_cols = st.columns(3)
placeholder_tickers = [ticker_cols[i].empty() for i in range(3)]

# Multi-Chart View (One for each)
st.markdown("### 📊 Market Overviews")
chart_cols = st.columns(3)
chart_placeholders = [chart_cols[i].empty() for i in range(3)]

# Analyzer Panels
st.markdown(f"### 🔍 Deep Dive: {selected_coin} Analysis")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        '<div class="analyzer-card"><h3>5s Micro-Analyzer</h3><div id="5s-output"></div></div>',
        unsafe_allow_html=True,
    )
    panel_5s = st.empty()

with col2:
    st.markdown(
        '<div class="analyzer-card"><h3>25s Macro-Analyzer</h3><div id="25s-output"></div></div>',
        unsafe_allow_html=True,
    )
    panel_25s = st.empty()

# --- Live Loop ---
symbols = ["BTC/USDT", "ETH/USDT", "AAVE/USDT"]
while True:
    try:
        # 1. Fetch Data for all symbols
        all_dfs = {}
        for sym in symbols:
            all_dfs[sym] = get_recent_trades(sym, minutes=10)

        target_df = all_dfs[selected_coin]

        # 2. Update Tickers & Charts
        for i, sym in enumerate(symbols):
            sym_df = all_dfs[sym]
            last_price = sym_df["price"].iloc[-1]

            # Ticker
            with placeholder_tickers[i]:
                st.markdown(
                    f'<div class="ticker-item"><div class="ticker-symbol">{sym}</div><div class="ticker-price">${last_price:,.2f}</div></div>',
                    unsafe_allow_html=True,
                )

            # Chart
            resampled = sym_df.resample("1min").agg({"price": "ohlc"})
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=resampled.index,
                        open=resampled["price"]["open"],
                        high=resampled["price"]["high"],
                        low=resampled["price"]["low"],
                        close=resampled["price"]["close"],
                        name=sym,
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False,
            )
            chart_placeholders[i].plotly_chart(fig, use_container_width=True)

        # 4. Context Layer Calculation (Target Coin)
        buy_vol = target_df[target_df["side"] == "buy"]["btc_volume"].sum()
        sell_vol = target_df[target_df["side"] == "sell"]["btc_volume"].sum()
        total_vol = buy_vol + sell_vol
        buy_ratio = (buy_vol / total_vol) * 100 if total_vol > 0 else 0

        short_mean = target_df["price"].tail(100).mean()
        long_mean = target_df["price"].mean()
        is_bearish = short_mean < long_mean

        # 5. Model Inference (Target Coin)
        for key, freq, rolling in [("5s", "5S", 10), ("25s", "25S", 10)]:
            panel = panel_5s if key == "5s" else panel_25s

            with panel.container():
                features_df = compute_live_features(target_df, freq, rolling)

                if is_bearish:
                    st.warning("⚠️ Market Sentiment: BEARISH")
                    st.info(
                        "Price is trending down. Detection paused to prevent false positives."
                    )
                elif buy_ratio < 40:
                    st.warning(f"⚠️ Low Buy Pressure: {buy_ratio:.1f}%")
                    st.info("Sellers dominate. Natural cooling detected.")
                elif features_df is not None and models:
                    # Scaling and Prediction
                    scaler = models[key]["scaler"]
                    rf = models[key]["rf"]

                    # Align features with training order (as seen in create_model.py)
                    # Note: pump_index added as it was in X = df.drop(columns=['date', 'symbol', 'gt'])
                    feature_cols = [
                        "pump_index",
                        "std_rush_order",
                        "avg_rush_order",
                        "std_trades",
                        "std_volume",
                        "avg_volume",
                        "std_price",
                        "avg_price",
                        "avg_price_max",
                        "hour_sin",
                        "hour_cos",
                        "minute_sin",
                        "minute_cos",
                    ]

                    # Ensure pump_index exists in inference df
                    features_df["pump_index"] = 0

                    # Fill missing columns if any
                    for col in feature_cols:
                        if col not in features_df.columns:
                            features_df[col] = 0

                    input_scaled = scaler.transform(features_df[feature_cols])
                    prediction = rf.predict(input_scaled)[0]
                    prob = rf.predict_proba(input_scaled)[0][1]

                    if prediction == 1:
                        st.markdown(
                            '<div class="status-warning">🚨 PUMP DETECTED</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="status-normal">✅ NORMAL ACTIVITY</div>',
                            unsafe_allow_html=True,
                        )

                    st.progress(prob)
                    st.write(f"Confidence Score: {prob:.2%}")

                    with st.expander("View Raw Features"):
                        st.write(features_df[feature_cols])
                else:
                    st.info("Gathering data for initial analyzer window...")

        time.sleep(30)
        st.rerun()

    except Exception as e:
        st.error(f"Live Update Error: {e}")
        time.sleep(10)

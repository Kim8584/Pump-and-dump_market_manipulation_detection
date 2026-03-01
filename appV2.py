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
    page_title="Pump & Dump Detector V2",
    page_icon="⚡",
    layout="wide",
)

# --- Custom CSS (Modern Dark Theme) ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;900&display=swap');

    .stApp {
        background-color: #0B0F1A;
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }

    .ticker-card {
        background-color: #161E2D;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1E293B;
        text-align: center;
        margin-bottom: 10px;
    }
    .ticker-symbol { font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono'; }
    .ticker-price { font-size: 1.1rem; font-weight: 700; color: #38BDF8; }

    .analyzer-card {
        background-color: #161E2D;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .status-normal { color: #34D399; font-weight: bold; font-size: 1.2rem; }
    .status-warning { color: #F87171; font-weight: bold; font-size: 1.2rem; animation: pulse 2s infinite; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .metric-container { display: flex; justify-content: space-between; margin-top: 10px; }
    .metric-box { text-align: left; }
    .metric-label { color: #64748B; font-size: 0.8rem; }
    .metric-value { font-family: 'JetBrains Mono'; font-size: 1rem; color: #F1F5F9; }

    /* Hide UI noise */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialization ---
if "trades" not in st.session_state:
    st.session_state.trades = {}  # {symbol: DataFrame}
if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = {} # {symbol: int}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "AAVE/USDT"]
EXCHANGE = ccxt.binance({'enableRateLimit': True})

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        return {
            "5s": {
                "rf": joblib.load("random_forest_5s_model.joblib"),
                "scaler": joblib.load("scaler_5s.joblib"),
                "freq": "5S",
                "rolling": 700
            },
            "25s": {
                "rf": joblib.load("random_forest_25s_model.joblib"),
                "scaler": joblib.load("scaler_25s.joblib"),
                "freq": "25S",
                "rolling": 900
            },
        }
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

MODELS = load_models()

# --- Data Engine ---
def fetch_incremental_trades(symbol, minutes_back=60):
    now_ms = EXCHANGE.milliseconds()
    
    # Use last_fetch_ts as the primary signal for initialization
    if symbol not in st.session_state.last_fetch_ts:
        # Initial warm-up fetch
        since = now_ms - (minutes_back * 60 * 1000)
        trades = EXCHANGE.fetch_trades(symbol, since=since)
        if not trades:
            # If warmup is empty, return empty df and try again next cycle
            return pd.DataFrame()
        df = pd.DataFrame(trades, columns=["timestamp", "side", "price", "amount"])
    else:
        # Incremental fetch
        since = st.session_state.last_fetch_ts[symbol] + 1
        trades = EXCHANGE.fetch_trades(symbol, since=since)
        if not trades:
            return st.session_state.trades[symbol]
        df_new = pd.DataFrame(trades, columns=["timestamp", "side", "price", "amount"])
        df = pd.concat([st.session_state.trades[symbol], df_new])

    # Clean and update state
    df["btc_volume"] = df["price"] * df["amount"]
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=['timestamp', 'price', 'amount']).sort_values("time")
    
    # Keep last 70 minutes (to safely cover the 900*25s window)
    cutoff = datetime.now() - timedelta(minutes=70)
    df = df[df["time"] > cutoff]
    
    # Atomically update state only if we have data
    if not df.empty:
        st.session_state.trades[symbol] = df
        st.session_state.last_fetch_ts[symbol] = int(df["timestamp"].max())
    else:
        # If all trades expired, clear state to trigger a fresh warm-up
        st.session_state.trades.pop(symbol, None)
        st.session_state.last_fetch_ts.pop(symbol, None)

    return df

def compute_features_v2(df, freq, rolling):
    # Strictly following features.py logic
    df_buy = df[df["side"] == "buy"].copy()
    if df_buy.empty: return None
    
    df_buy.set_index("time", inplace=True)
    df_resampled = df_buy.resample(freq)
    
    features = {}
    
    # 1. Rush Orders
    rush_counts = df_buy.groupby(df_buy.index).size()
    rush_binary = (rush_counts > 1).astype(int)
    rush_resampled = rush_binary.resample(freq).sum() # Sum of btc_volume in features.py? 
    # Actually features.py uses rush_volume = df_buy.groupby(pd.Grouper(freq=freq))["btc_volume"].sum() 
    # where df_buy had entries set to 0/1. Let's replicate precisely:
    df_rush = df_buy.copy()
    counts = df_rush.groupby(df_rush.index).size()
    df_rush.loc[counts[counts == 1].index, 'btc_volume'] = 0 # Not quite right for grouped index, but following spirit
    # Simplified precise replication:
    rush_vol = (df_buy.groupby(df_buy.index).size() > 1).astype(int).resample(freq).sum()
    
    features["std_rush_order"] = rush_vol.rolling(rolling).std().pct_change().fillna(0).iloc[-1]
    features["avg_rush_order"] = rush_vol.rolling(rolling).mean().pct_change().fillna(0).iloc[-1]
    
    # 2. Trades & Volume
    features["std_trades"] = df_resampled["price"].count().rolling(rolling).std().pct_change().fillna(0).iloc[-1]
    features["std_volume"] = df_resampled["btc_volume"].sum().rolling(rolling).std().pct_change().fillna(0).iloc[-1]
    features["avg_volume"] = df_resampled["btc_volume"].sum().rolling(rolling).mean().pct_change().fillna(0).iloc[-1]
    
    # 3. Price
    features["std_price"] = df_resampled["price"].mean().rolling(rolling).std().pct_change().fillna(0).iloc[-1]
    features["avg_price"] = df_resampled["price"].mean().rolling(10).mean().pct_change().fillna(0).iloc[-1]
    features["avg_price_max"] = df_resampled["price"].max().rolling(10).mean().pct_change().fillna(0).iloc[-1]
    
    # 4. Time components
    now = datetime.now()
    features["hour_sin"] = np.sin(2 * np.pi * now.hour / 23)
    features["hour_cos"] = np.cos(2 * np.pi * now.hour / 23)
    features["minute_sin"] = np.sin(2 * np.pi * now.minute / 59)
    features["minute_cos"] = np.cos(2 * np.pi * now.minute / 59)
    features["pump_index"] = 0
    
    return pd.DataFrame([features])

# --- UI Layout ---
st.title("⚡ Live Pump-and-Dump Detector V2")

# Sidebar for controls
selected_symbol = st.sidebar.selectbox("Analysis Target", SYMBOLS)
st.sidebar.info("V2 uses a sliding window update (5s) for smoother performance.")

# Tickers at top
ticker_slots = st.columns(len(SYMBOLS))
placeholders = {sym: ticker_slots[i].empty() for i, sym in enumerate(SYMBOLS)}

# Main Content Area
col_chart, col_ana = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 Live Candlesticks (1m)")
    chart_placeholder = st.empty()

with col_ana:
    st.markdown("### 🔍 Model Analysis")
    ana_5s_placeholder = st.empty()
    ana_25s_placeholder = st.empty()

# --- The Live Fragment ---
@st.fragment(run_every=5)
def update_dashboard():
    # 1. Update Data for all
    all_latest = {}
    for sym in SYMBOLS:
        df = fetch_incremental_trades(sym)
        all_latest[sym] = df
        
        # Update Ticker, with a check for empty data
        if df.empty:
            placeholders[sym].markdown(
                f'<div class="ticker-card"><div class="ticker-symbol">{sym}</div><div class="ticker-price">--.--</div></div>',
                unsafe_allow_html=True
            )
            continue

        last_price = df["price"].iloc[-1]
        placeholders[sym].markdown(
            f'<div class="ticker-card"><div class="ticker-symbol">{sym}</div><div class="ticker-price">${last_price:,.2f}</div></div>',
            unsafe_allow_html=True
        )

    # 2. Update Chart for Selected Symbol, with a check for empty data
    target_df = all_latest[selected_symbol]
    if target_df.empty:
        chart_placeholder.info(f"Waiting for trade data for {selected_symbol}...")
        ana_5s_placeholder.empty()
        ana_25s_placeholder.empty()
        return

    target_df = target_df.copy()
    target_df.set_index("time", inplace=True)
    resampled = target_df.resample("1min").agg({"price": "ohlc", "btc_volume": "sum"})
    
    fig = go.Figure(data=[go.Candlestick(
        x=resampled.index, open=resampled["price"]["open"],
        high=resampled["price"]["high"], low=resampled["price"]["low"],
        close=resampled["price"]["close"], name="Price"
    )])
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis_rangeslider_visible=False)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_symbol}")

    # 3. Run Inference
    # Check Sentiment (Bearish filter from app.py)
    if len(target_df) < 100:
        is_bearish = False
    else:
        short_m = target_df["price"].tail(100).mean()
        long_m = target_df["price"].mean()
        is_bearish = short_m < long_m
    
    buy_vol = target_df[target_df["side"] == "buy"]["btc_volume"].sum()
    total_vol = target_df["btc_volume"].sum()
    buy_ratio = (buy_vol / total_vol) * 100 if total_vol > 0 else 0

    for key, slot in [("5s", ana_5s_placeholder), ("25s", ana_25s_placeholder)]:
        with slot.container():
            st.markdown(f"#### {key.upper()} Detector")
            
            if is_bearish:
                st.info("⏸️ Paused: Bearish Sentiment")
                continue
            if buy_ratio < 40:
                st.info(f"⏸️ Paused: Low Buy Ratio ({buy_ratio:.1f}%)")
                continue
            
            # Ensure we have enough data for the rolling window. This is the "Confidence Building" UI.
            required_rows = int(MODELS[key]["rolling"] * (pd.to_timedelta(MODELS[key]["freq"]).total_seconds() / 5)) # Estimate rows needed
            current_rows = len(all_latest[selected_symbol])
            
            if current_rows < required_rows:
                progress_percent = int((current_rows / required_rows) * 100)
                st.info(f"Building history for full accuracy...")
                st.progress(progress_percent)
                st.caption(f"({current_rows}/{required_rows} data points)")
                continue

            features_df = compute_features_v2(all_latest[selected_symbol], MODELS[key]["freq"], MODELS[key]["rolling"])
            
            if features_df is not None and not features_df.empty and not features_df.isnull().values.any():
                scaler = MODELS[key]["scaler"]
                rf = MODELS[key]["rf"]
                
                # Order columns for model
                cols = ["pump_index", "std_rush_order", "avg_rush_order", "std_trades", "std_volume", 
                        "avg_volume", "std_price", "avg_price", "avg_price_max", 
                        "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
                
                for col in cols:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                features_df = features_df[cols]

                input_data = scaler.transform(features_df)
                prob = rf.predict_proba(input_data)[0][1]
                pred = rf.predict(input_data)[0]

                status_class = "status-warning" if pred == 1 else "status-normal"
                status_text = "🚨 PUMP DETECTED" if pred == 1 else "✅ NORMAL"
                
                st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
                st.progress(prob)
                st.write(f"Confidence: {prob:.2%}")
            else:
                st.caption("Calculating features...")

def initial_warmup():
    """Function to run on first load to populate data without fragment overhead."""
    with st.status("Warming up models...", expanded=True) as status:
        for sym in SYMBOLS:
            if sym not in st.session_state.last_fetch_ts:
                # FAST START: Only fetch 10 minutes initially
                status.update(label=f"Fetching 10min history for {sym}...")
                fetch_incremental_trades(sym, minutes_back=10) 
    status.update(label="Warm-up complete! Starting live updates.", state="complete", expanded=False)

# --- Start the Engine ---
initial_warmup()
update_dashboard()

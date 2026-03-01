import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- Configuration & Data Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data/presentation_data.csv')
MODEL_DIR = os.path.dirname(BASE_DIR)

# Load the models and scalers
def load_assets():
    assets = {}
    for freq in ['5s', '25s']:
        try:
            model_path = os.path.join(MODEL_DIR, f'random_forest_{freq}_model.joblib')
            scaler_path = os.path.join(MODEL_DIR, f'scaler_{freq}.joblib')
            print(f"Loading assets for {freq} from {MODEL_DIR}")
            assets[freq] = {
                'model': joblib.load(model_path),
                'scaler': joblib.load(scaler_path)
            }
        except Exception as e:
            print(f"Error loading {freq} assets: {e}")
            assets[freq] = None
    return assets

assets = load_assets()

# Load trade data
try:
    df_trades = pd.read_csv(DATA_PATH, comment='#')
    # Ensure timestamp is numeric before converting to datetime
    df_trades['timestamp'] = pd.to_numeric(df_trades['timestamp'], errors='coerce')
    df_trades = df_trades.dropna(subset=['timestamp'])
    df_trades['time'] = pd.to_datetime(df_trades['timestamp'], unit='ms')
except Exception as e:
    print(f"Error loading data: {e}")
    df_trades = pd.DataFrame(columns=['timestamp', 'side', 'price', 'amount', 'time'])

# --- Feature Engineering & Prediction Logic ---
def compute_features(df, freq, rolling_window):
    if df.empty:
        return pd.DataFrame()

    df_buy = df[df["side"] == "buy"].copy().set_index("time")
    
    # Resample
    df_resampled = df_buy.resample(freq)
    
    # Calculate features
    # Rush orders: trades at the exact same timestamp
    rush_vol = (df_buy.groupby(df_buy.index).size() > 1).astype(int).resample(freq).sum()
    
    features = pd.DataFrame(index=df_resampled.mean(numeric_only=True).index)
    features["std_rush_order"] = rush_vol.rolling(rolling_window, min_periods=1).std().pct_change(fill_method=None)
    features["avg_rush_order"] = rush_vol.rolling(rolling_window, min_periods=1).mean().pct_change(fill_method=None)
    features["std_trades"] = df_resampled["price"].count().rolling(rolling_window, min_periods=1).std().pct_change(fill_method=None)
    features["std_volume"] = df_resampled["amount"].sum().rolling(rolling_window, min_periods=1).std().pct_change(fill_method=None)
    features["avg_volume"] = df_resampled["amount"].sum().rolling(rolling_window, min_periods=1).mean().pct_change(fill_method=None)
    features["std_price"] = df_resampled["price"].mean().rolling(rolling_window, min_periods=1).std().pct_change(fill_method=None)
    
    # Time features
    features["hour_sin"] = np.sin(2 * np.pi * features.index.hour / 23)
    features["hour_cos"] = np.cos(2 * np.pi * features.index.hour / 23)
    features["minute_sin"] = np.sin(2 * np.pi * features.index.minute / 59)
    features["minute_cos"] = np.cos(2 * np.pi * features.index.minute / 59)

    # Model specific price features
    features["avg_price"] = df_resampled["price"].mean().rolling(10, min_periods=1).mean().pct_change(fill_method=None)
    features["avg_price_max"] = df_resampled["price"].max().rolling(10, min_periods=1).mean().pct_change(fill_method=None)
    features["pump_index"] = 0

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features

def predict(df_feat, freq):
    if df_feat.empty or assets[freq] is None:
        return pd.Series(0, index=df_feat.index, name="prediction_prob")

    feature_cols = [
        "pump_index", "std_rush_order", "avg_rush_order", "std_trades",
        "std_volume", "avg_volume", "std_price", "avg_price",
        "avg_price_max", "hour_sin", "hour_cos", "minute_sin", "minute_cos"
    ]
    
    # Scale and predict
    try:
        input_scaled = assets[freq]['scaler'].transform(df_feat[feature_cols])
        probs = assets[freq]['model'].predict_proba(input_scaled)[:, 1]
        return pd.Series(probs, index=df_feat.index, name="prediction_prob")
    except Exception as e:
        print(f"Prediction error for {freq}: {e}")
        return pd.Series(0, index=df_feat.index, name="prediction_prob")

# Pre-compute both frequencies for responsiveness
print("Pre-computing 5S features and predictions...")
df_features_5s = compute_features(df_trades, '5s', 700)
df_features_5s['prediction_prob'] = predict(df_features_5s, '5s')

print("Pre-computing 25S features and predictions...")
df_features_25s = compute_features(df_trades, '25s', 900)
df_features_25s['prediction_prob'] = predict(df_features_25s, '25s')

# --- Narrative Events ---
CONTEXT_EVENTS = {
    1677693600: {"time": "18:00 UTC", "text": "Market base state: Low volatility detected across major pairs."},
    1677696300: {"time": "18:45 UTC", "text": "BTC liquidity shift: Minor price correction observed."},
    1677697140: {"time": "18:59 UTC", "text": "Anomaly Trigger: Coordinated social volume spike detected."},
    1677697200: {"time": "19:00 UTC", "text": "PUMP INITIATED: Hyper-velocity buy-side pressure detected."},
    1677697260: {"time": "19:01 UTC", "text": "Peak Volatility: Exhaustion gap appearing in order book."},
    1677698400: {"time": "19:20 UTC", "text": "Equilibrium reached: Price stabilizing post-event."}
}

# --- App Layout ---
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
    ]
)
server = app.server

app.layout = html.Div(className="main-container", children=[
    # Header
    html.Div(className="header", children=[
        html.Div(className="title-section", children=[
            html.H1("V4 MARKET SENTINEL"),
            html.P("Pump & Dump / Market Manipulation Real-Time Analysis"),
        ]),
        html.Div(className="toggle-container", children=[
            html.Button("5S MICRO", id="btn-5s", className="toggle-btn active", n_clicks=0),
            html.Button("25S MACRO", id="btn-25s", className="toggle-btn", n_clicks=0),
        ]),
    ]),

    # Main Grid
    html.Div(className="dashboard-grid", children=[
        # Left Panel: Features
        html.Div(className="panel", children=[
            html.Div(className="panel-header", children=[
                html.Div(className="panel-title", children=[
                    html.I(className="bi bi-graph-up"),
                    html.Span("FEATURE DYNAMICS")
                ]),
                html.Div(id="feature-freq-label", className="panel-subtitle", children="Displaying 5-Second Interval Features")
            ]),
            html.Div(id="feature-graphs-container", style={"flex": "1", "overflowY": "auto", "paddingRight": "10px"}, children=[
                dcc.Graph(id='feat-1', config={'displayModeBar': False}),
                dcc.Graph(id='feat-2', config={'displayModeBar': False}),
                dcc.Graph(id='feat-3', config={'displayModeBar': False}),
            ])
        ]),

        # Center Panel: Main Visualization
        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
            # Trade Density
            html.Div(className="panel", style={"flex": "2"}, children=[
                html.Div(className="panel-header", children=[
                    html.Div(className="panel-title", children="TRADE VOLUME DENSITY"),
                    html.Div(className="panel-subtitle", children="Comparing Sell (Purple) vs Buy (Yellow) Intensity")
                ]),
                dcc.Graph(id='main-density-graph', style={"flex": "1"})
            ]),
            # Confidence
            html.Div(className="panel", style={"flex": "1"}, children=[
                html.Div(className="panel-header", children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                        html.Div(className="panel-title", children="DETECTION CONFIDENCE"),
                        html.Div(id="prediction-badge")
                    ])
                ]),
                dcc.Graph(id='confidence-graph', style={"flex": "1"})
            ]),
        ]),

        # Right Panel: Narrative & Metrics
        html.Div(className="panel", children=[
            html.Div(className="panel-header", children=[
                html.Div(className="panel-title", children="MARKET NARRATIVE"),
                html.Div(className="panel-subtitle", children="Historical Context & Live Alerts")
            ]),
            html.Div(id="timeline-container", className="timeline-container", style={"overflowY": "auto", "flex": "1"})
        ]),
    ]),

    # Controls
    html.Div(className="controls-container", children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "10px"}, children=[
            html.Span("PLAYBACK TIMELINE", style={"fontWeight": "700", "fontSize": "0.8rem", "color": "#60a5fa"}),
            html.Span(id="current-time-display", style={"fontFamily": "JetBrains Mono", "fontSize": "0.9rem"})
        ]),
        dcc.Slider(
            id='time-slider',
            min=df_trades['timestamp'].min(),
            max=df_trades['timestamp'].max(),
            value=df_trades['timestamp'].min(),
            step=1000,
            marks=None,
            tooltip={"placement": "top", "always_visible": True, "template": "{value}"}
        ),
    ]),

    dcc.Store(id='analysis-mode', data='5s'),
])

# --- Callbacks ---

@app.callback(
    [Output('analysis-mode', 'data'),
     Output('btn-5s', 'className'),
     Output('btn-25s', 'className'),
     Output('feature-freq-label', 'children')],
    [Input('btn-5s', 'n_clicks'),
     Input('btn-25s', 'n_clicks')],
    [State('analysis-mode', 'data')]
)
def switch_mode(n5, n25, current_mode):
    ctx = callback_context
    if not ctx.triggered:
        return '5s', "toggle-btn active", "toggle-btn", "Displaying 5-Second Interval Features"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-5s':
        return '5s', "toggle-btn active", "toggle-btn", "Displaying 5-Second Interval Features"
    else:
        return '25s', "toggle-btn", "toggle-btn active", "Displaying 25-Second Interval Features"

@app.callback(
    [Output('main-density-graph', 'figure'),
     Output('confidence-graph', 'figure'),
     Output('feat-1', 'figure'),
     Output('feat-2', 'figure'),
     Output('feat-3', 'figure'),
     Output('timeline-container', 'children'),
     Output('current-time-display', 'children'),
     Output('prediction-badge', 'children')],
    [Input('time-slider', 'value'),
     Input('analysis-mode', 'data')]
)
def update_dashboard(slider_val, mode):
    current_time = pd.to_datetime(slider_val, unit='ms')
    WINDOW_SIZE_MINS = 15
    start_time = current_time - timedelta(minutes=WINDOW_SIZE_MINS / 2)
    end_time = current_time + timedelta(minutes=WINDOW_SIZE_MINS / 2)

    # Filter data
    mask_trades = (df_trades['time'] >= start_time) & (df_trades['time'] <= end_time)
    current_trades = df_trades[mask_trades]
    
    feat_df = df_features_5s if mode == '5s' else df_features_25s
    mask_feat = (feat_df.index >= start_time) & (feat_df.index <= end_time)
    current_feats = feat_df[mask_feat]

    # 1. Trade Density Graph
    fig_density = go.Figure()
    if not current_trades.empty:
        # Separate buys and sells for comparison
        buys = current_trades[current_trades['side'] == 'buy']
        sells = current_trades[current_trades['side'] == 'sell']
        
        fig_density.add_trace(go.Scattergl(
            x=sells['time'], y=sells['price'], mode='markers',
            marker=dict(color='#8b5cf6', size=current_trades['amount'].clip(0, 1) * 10 + 2, opacity=0.4),
            name='SELL'
        ))
        fig_density.add_trace(go.Scattergl(
            x=buys['time'], y=buys['price'], mode='markers',
            marker=dict(color='#fbbf24', size=current_trades['amount'].clip(0, 1) * 10 + 2, opacity=0.6),
            name='BUY'
        ))
        
        y_range = [current_trades['price'].min() * 0.998, current_trades['price'].max() * 1.002]
        fig_density.update_layout(yaxis_range=y_range)

    fig_density.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#ef4444")
    fig_density.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_range=[start_time, end_time]
    )

    # 2. Confidence Graph
    fig_conf = go.Figure()
    if not current_feats.empty:
        fig_conf.add_trace(go.Scatter(
            x=current_feats.index, y=current_feats['prediction_prob'],
            mode='lines', fill='tozeroy',
            line=dict(color='#06b6d4', width=2),
            fillcolor='rgba(6, 182, 212, 0.2)'
        ))
    
    fig_conf.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#ef4444")
    fig_conf.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0, 1.05],
        xaxis_range=[start_time, end_time]
    )

    # 3. Feature Graphs
    def create_feat_fig(col, color, title):
        fig = go.Figure()
        if not current_feats.empty:
            fig.add_trace(go.Scatter(x=current_feats.index, y=current_feats[col], mode='lines', line=dict(color=color, width=1.5)))
        fig.add_vline(x=current_time, line_width=1, line_dash="dash", line_color="#ef4444")
        fig.update_layout(
            template="plotly_dark",
            title=dict(text=title, font=dict(size=10, color='#9ca3af')),
            height=150,
            margin=dict(l=30, r=10, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_range=[start_time, end_time],
            xaxis=dict(showticklabels=False)
        )
        return fig

    f1 = create_feat_fig('std_rush_order', '#ef4444', 'RUSH ORDER VOLATILITY')
    f2 = create_feat_fig('avg_volume', '#3b82f6', 'AVERAGE VOLUME DELTA')
    f3 = create_feat_fig('std_price', '#10b981', 'PRICE VARIANCE')

    # 4. Timeline
    timeline_items = []
    current_unix = slider_val / 1000
    for ts, event in CONTEXT_EVENTS.items():
        is_past = current_unix >= ts
        is_recent = is_past and (current_unix - ts < 300) # within 5 mins
        
        classname = "event-card active" if is_recent else "event-card"
        if not is_past: classname += " future" # Optional styling for future events
        
        timeline_items.append(html.Div(className=classname, children=[
            html.Div(event['time'], className="event-time"),
            html.Div(event['text'], className="event-text", style={"color": "#fff" if is_past else "#4b5563"})
        ]))

    # 5. Prediction Badge
    latest_prob = 0
    if not current_feats.empty:
        # Get the prob closest to current_time
        latest_prob = current_feats.iloc[len(current_feats)//2]['prediction_prob']
    
    if latest_prob > 0.7:
        badge = html.Span("🚨 HIGH RISK", className="prediction-status status-pump")
    elif latest_prob > 0.4:
        badge = html.Span("⚠️ CAUTION", className="prediction-status status-pump", style={"backgroundColor": "rgba(251, 191, 36, 0.1)", "color": "#fbbf24", "borderColor": "#fbbf24"})
    else:
        badge = html.Span("✅ STABLE", className="prediction-status status-normal")

    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    return fig_density, fig_conf, f1, f2, f3, timeline_items, time_str, badge

if __name__ == '__main__':
    print("Starting V4 Market Sentinel...")
    app.run(debug=True, port=8051)

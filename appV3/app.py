import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Configuration & Data Loading ---
# Load the pre-trained model and scaler
try:
    rf_model = joblib.load('../random_forest_5s_model.joblib')
    scaler = joblib.load('../scaler_5s.joblib')
except FileNotFoundError:
    print("ERROR: Model files not found. Please ensure .joblib files are in the root directory.")
    # In a real scenario, you might exit or use a dummy model.
    rf_model = scaler = None

# Load the static, pre-downloaded trade data
try:
    df_trades = pd.read_csv('data/presentation_data.csv')
    df_trades['time'] = pd.to_datetime(df_trades['timestamp'], unit='ms')
except FileNotFoundError:
    print("ERROR: data/presentation_data.csv not found. Please create the data file.")
    df_trades = pd.DataFrame() # Empty df to prevent crashing

# --- Feature & Prediction Pre-computation (The "Time Machine") ---
# This is the key to a fast dashboard. We do all heavy lifting once at the start.
def compute_all_features(df):
    if df.empty:
        return pd.DataFrame()

    df_buy = df[df["side"] == "buy"].copy().set_index("time")
    
    # Resample to 5-second intervals, handling future deprecation warning for 'S'
    df_resampled = df_buy.resample("5s")
    
    # Calculate features
    rush_vol = (df_buy.groupby(df_buy.index).size() > 1).astype(int).resample("5s").sum()
    
    features = pd.DataFrame(index=df_resampled.mean(numeric_only=True).index)
    features["std_rush_order"] = rush_vol.rolling(700, min_periods=1).std().pct_change()
    features["avg_rush_order"] = rush_vol.rolling(700, min_periods=1).mean().pct_change()
    features["std_trades"] = df_resampled["price"].count().rolling(700, min_periods=1).std().pct_change()
    features["std_volume"] = df_resampled["amount"].sum().rolling(700, min_periods=1).std().pct_change()
    features["avg_volume"] = df_resampled["amount"].sum().rolling(700, min_periods=1).mean().pct_change()
    features["std_price"] = df_resampled["price"].mean().rolling(700, min_periods=1).std().pct_change()
    
    # Add time features
    features["hour_sin"] = np.sin(2 * np.pi * features.index.hour / 23)
    features["hour_cos"] = np.cos(2 * np.pi * features.index.hour / 23)
    features["minute_sin"] = np.sin(2 * np.pi * features.index.minute / 59)
    features["minute_cos"] = np.cos(2 * np.pi * features.index.minute / 59)

    # For the model, we need avg_price and avg_price_max, which had a different window
    features["avg_price"] = df_resampled["price"].mean().rolling(10, min_periods=1).mean().pct_change(fill_method=None)
    features["avg_price_max"] = df_resampled["price"].max().rolling(10, min_periods=1).mean().pct_change(fill_method=None)
    features["pump_index"] = 0

    # Explicitly handle inf values which can break sklearn's scaler
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(0)

df_features = compute_all_features(df_trades)

# Run model predictions for the entire dataset
def predict_all(df_feat):
    if df_feat.empty or rf_model is None or scaler is None:
        return pd.Series(0, index=df_feat.index, name="prediction_prob")

    feature_cols = [
        "pump_index", "std_rush_order", "avg_rush_order", "std_trades",
        "std_volume", "avg_volume", "std_price", "avg_price",
        "avg_price_max", "hour_sin", "hour_cos", "minute_sin", "minute_cos"
    ]
    
    # Ensure all columns exist
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    df_feat = df_feat[feature_cols]
    
    input_scaled = scaler.transform(df_feat)
    probs = rf_model.predict_proba(input_scaled)[:, 1]
    return pd.Series(probs, index=df_feat.index, name="prediction_prob")

df_features["prediction_prob"] = predict_all(df_features)


# --- Hardcoded Narrative Context ---
CONTEXT_EVENTS = {
    1677693600: {"time": "18:00 UTC", "text": "Market opens flat. BTC showing low volatility."},
    1677696300: {"time": "18:45 UTC", "text": "Minor dip in BTC price, altcoins follow."},
    1677697140: {"time": "18:59 UTC", "text": "Social media mentions for target asset show unusual increase."},
    1677697200: {"time": "19:00 UTC", "text": "PUMP EVENT: A sudden, massive influx of buy orders occurs in a coordinated manner."},
    1677697260: {"time": "19:01 UTC", "text": "Price peaks as initial pump participants begin to sell off their holdings."},
    1677698400: {"time": "19:20 UTC", "text": "Price stabilizes at a level higher than pre-pump, but significantly lower than the peak."}
}


# --- App Layout ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server 

app.layout = html.Div(className="three-col-container", children=[
    # --- LEFT COLUMN (FEATURES) ---
    html.Div(className="left-col", children=[
        html.H4("Model Feature Explorer"),
        dcc.Graph(id='feature-graph-1', figure=go.Figure().update_layout(template="plotly_dark")),
        dcc.Graph(id='feature-graph-2', figure=go.Figure().update_layout(template="plotly_dark")),
        dcc.Graph(id='feature-graph-3', figure=go.Figure().update_layout(template="plotly_dark")),
    ]),

    # --- CENTER COLUMN (MAIN EVENT & CONTROLS) ---
    html.Div(className="center-col", children=[
        html.H1("Pump & Dump Narrative Analysis"),
        dcc.Graph(id='trade-density-graph'),
        dcc.Graph(id='prediction-prob-graph'),
        dcc.Slider(
            id='time-slider',
            min=df_trades['timestamp'].min(),
            max=df_trades['timestamp'].max(),
            value=df_trades['timestamp'].min(),
            marks=None, # We can add marks later if needed
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ]),

    # --- RIGHT COLUMN (CONTEXT) ---
    html.Div(className="right-col", children=[
        html.H4("Market Narrative"),
        html.Div(id='context-panel')
    ]),
])


# --- Interactivity Callbacks ---
@app.callback(
    [Output('trade-density-graph', 'figure'),
     Output('prediction-prob-graph', 'figure'),
     Output('feature-graph-1', 'figure'),
     Output('feature-graph-2', 'figure'),
     Output('feature-graph-3', 'figure'),
     Output('context-panel', 'children')],
    [Input('time-slider', 'value')]
)
def update_graphs(slider_timestamp):
    # Convert slider value to datetime
    current_time = pd.to_datetime(slider_timestamp, unit='ms')
    WINDOW_SIZE_MINUTES = 10 # Display 10 minutes of data around the current time

    # Filter data based on slider's current_time and window size
    start_time = current_time - timedelta(minutes=WINDOW_SIZE_MINUTES / 2)
    end_time = current_time + timedelta(minutes=WINDOW_SIZE_MINUTES / 2)

    current_window_trades = df_trades[(df_trades['time'] >= start_time) & (df_trades['time'] <= end_time)]
    current_window_features = df_features[(df_features.index >= start_time) & (df_features.index <= end_time)]

    # 1. Trade Density Plot
    fig_density = go.Figure()
    if not current_window_trades.empty:
        fig_density.add_trace(go.Scattergl(
            x=current_window_trades['time'], y=current_window_trades['price'], mode='markers',
            marker=dict(color=current_window_trades['amount'], colorscale='Viridis', showscale=True, size=5, opacity=0.7),
            name='Trades'
        ))
        # Dynamic Y-axis for better comparison
        y_min = current_window_trades['price'].min() * 0.99
        y_max = current_window_trades['price'].max() * 1.01
        fig_density.update_yaxes(range=[y_min, y_max])
    
    fig_density.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="red")
    fig_density.update_layout(
        template="plotly_dark", 
        title="Trade Density (Hover for Details)", 
        xaxis_title="Time", 
        yaxis_title="Price",
        xaxis_range=[start_time, end_time] # Ensure x-axis matches window
    )

    # 2. Prediction Probability Plot
    fig_pred = go.Figure()
    if not current_window_features.empty:
        fig_pred.add_trace(go.Scatter(
            x=current_window_features.index, y=current_window_features['prediction_prob'], mode='lines',
            fill='tozeroy', line_color='cyan'
        ))
    fig_pred.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="red")
    fig_pred.update_layout(
        template="plotly_dark", 
        title="Model Prediction Confidence", 
        yaxis=dict(range=[0, 1]),
        xaxis_range=[start_time, end_time] # Ensure x-axis matches window
    )

    # 3. Feature Plots
    def create_feature_fig(feature_name, color):
        fig = go.Figure()
        if not current_window_features.empty:
            fig.add_trace(go.Scatter(x=current_window_features.index, y=current_window_features[feature_name], mode='lines', line_color=color))
        fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="red")
        fig.update_layout(template="plotly_dark", title=feature_name, height=200, margin=dict(t=30, b=30, l=10, r=10),
                          xaxis_range=[start_time, end_time]) # Ensure x-axis matches window
        return fig

    fig_feat1 = create_feature_fig('std_rush_order', '#F87171')
    fig_feat2 = create_feature_fig('avg_volume', '#38BDF8')
    fig_feat3 = create_feature_fig('std_price', '#34D399')
    
    # 4. Context Panel
    current_unix_time = int(slider_timestamp / 1000)
    context_children = []
    for event_time, event_data in CONTEXT_EVENTS.items():
        # Highlight if event time is within the current display window
        is_highlighted = (event_time * 1000 >= start_time.timestamp() * 1000) and \
                         (event_time * 1000 <= end_time.timestamp() * 1000)
        classname = "timeline-event highlighted" if is_highlighted else "timeline-event"
        context_children.append(html.Div(className=classname, children=[
            html.P(event_data['time'], className="timeline-timestamp"),
            html.P(event_data['text'])
        ]))

    return fig_density, fig_pred, fig_feat1, fig_feat2, fig_feat3, context_children


if __name__ == '__main__':
    app.run(debug=True)

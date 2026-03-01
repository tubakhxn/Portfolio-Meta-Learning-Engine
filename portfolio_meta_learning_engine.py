import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
import datetime

# =====================
# THEME & STYLES
# =====================
st.set_page_config(
    page_title="Portfolio Meta-Learning Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark hedge-fund aesthetic
st.markdown("""
    <style>
    body, .stApp {
        background-color: #181A20;
        color: #E0E0E0;
    }
    .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
        background-color: #181A20 !important;
        color: #E0E0E0 !important;
    }
    .stSidebar {
        background-color: #181A20 !important;
    }
    .metric-card {
        background: linear-gradient(90deg, #23272F 60%, #1A1D23 100%);
        border-radius: 12px;
        box-shadow: 0 0 16px #00FFC6;
        color: #00FFC6;
        padding: 24px;
        margin-bottom: 16px;
    }
    .neon {
        color: #00FFC6;
        text-shadow: 0 0 8px #00FFC6, 0 0 16px #00FFC6;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# SIDEBAR CONTROLS
# =====================
with st.sidebar:
    st.title("⚡ Meta-Learning Engine")
    strategy_set = st.multiselect(
        "Strategy Universe",
        ["Mean-Variance", "Risk Parity", "Minimum Variance", "Momentum Tilt", "Defensive Allocation"],
        default=["Mean-Variance", "Risk Parity", "Momentum Tilt"]
    )
    lookback = st.slider("Lookback Window (days)", 30, 252, 90, 10)
    risk_aversion = st.slider("Risk Aversion", 0.1, 5.0, 2.0, 0.1)
    regime_sensitivity = st.slider("Regime Sensitivity", 0.1, 2.0, 1.0, 0.1)
    st.markdown("---")
    st.markdown("<span class='neon'>Demo Mode: Using Sample Data</span>", unsafe_allow_html=True)

# =====================
# DATA PIPELINE
# =====================
def load_data(uploaded_file, tickers, start_date, end_date, fetch_data):
    # Always load sample data for demo
    sample_tickers = ["SPY", "TLT", "GLD"]  # Fewer assets for speed
    # Shorter date range for speed
    data = yf.download(sample_tickers, start="2023-01-01", end="2023-12-31")
    df = data['Close']
    df = df.dropna(how="all")
    return df

def compute_features(df, lookback):
    returns = np.log(df / df.shift(1)).dropna()
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    corr = returns.rolling(lookback).corr().dropna()
    trend = returns.rolling(lookback).mean()
    meanrev = -trend
    return returns, vol, corr, trend, meanrev

# =====================
# MARKET REGIME DETECTION
# =====================
def detect_regimes(vol, corr, trend, regime_sensitivity):
    # Volatility clustering via GMM
    scaler = StandardScaler()
    vol_scaled = scaler.fit_transform(vol.fillna(0))
    gmm_vol = GaussianMixture(n_components=2, random_state=42)
    vol_regime = gmm_vol.fit_predict(vol_scaled)
    # Correlation regime via PCA
    corr_matrix = corr.groupby(corr.index).mean().fillna(0)
    pca = PCA(n_components=1)
    corr_regime = pca.fit_transform(corr_matrix)
    corr_regime = (corr_regime > np.median(corr_regime)).astype(int)
    # Trend regime
    trend_strength = (trend.abs().mean(axis=1) > regime_sensitivity * trend.abs().mean().mean()).astype(int)
    # Combine regimes
    # Align regime arrays to vol.index length
    # Use only the last n rows to match vol.index
    n = len(vol.index)
    vol_regime = np.array(vol_regime)[-n:]
    corr_regime = np.array(corr_regime.flatten())[-n:]
    trend_strength = np.array(trend_strength)[-n:]
    regime_df = pd.DataFrame({
        "Volatility": vol_regime,
        "Correlation": corr_regime,
        "Trend": trend_strength
    }, index=vol.index)
    return regime_df

# =====================
# STRATEGY FUNCTIONS
# =====================
def mean_variance(returns, risk_aversion):
    mu = returns.mean()
    cov = returns.cov()
    n = len(mu)
    def obj(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -port_ret + risk_aversion * port_vol
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0,1)]*n
    w0 = np.ones(n)/n
    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x

def risk_parity(returns):
    cov = returns.cov()
    n = len(cov)
    def risk_contribution(w):
        port_var = np.dot(w, np.dot(cov, w))
        mrc = cov.dot(w)
        rc = w * mrc / port_var
        return rc
    def obj(w):
        rc = risk_contribution(w)
        return np.sum((rc - 1/n)**2)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0,1)]*n
    w0 = np.ones(n)/n
    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x

def min_variance(returns):
    cov = returns.cov()
    n = len(cov)
    def obj(w):
        return np.dot(w, np.dot(cov, w))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0,1)]*n
    w0 = np.ones(n)/n
    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x

def momentum_tilt(returns):
    mu = returns.mean()
    mom = mu / returns.std()
    w = mom / mom.sum()
    return w

def defensive_allocation(returns, vol):
    inv_vol = 1 / (vol.iloc[-1] + 1e-6)
    w = inv_vol / inv_vol.sum()
    return w

# =====================
# META-LEARNER
# =====================
def bayesian_strategy_selection(returns, strategies, regime_df):
    # Simulate Bayesian optimization for strategy selection
    # Reward: rolling risk-adjusted return
    n = len(returns)
    strategy_timeline = []
    confidence_scores = []
    allocations = []
    for i in range(lookback, n):
        window = returns.iloc[i-lookback:i]
        vol_window = window.rolling(lookback).std().iloc[-1]
        strat_scores = {}
        for strat in strategies:
            if strat == "Mean-Variance":
                w = mean_variance(window, risk_aversion)
            elif strat == "Risk Parity":
                w = risk_parity(window)
            elif strat == "Minimum Variance":
                w = min_variance(window)
            elif strat == "Momentum Tilt":
                w = momentum_tilt(window)
            elif strat == "Defensive Allocation":
                w = defensive_allocation(window, vol_window)
            else:
                w = np.ones(window.shape[1])/window.shape[1]
            port_ret = np.dot(w, window.mean())
            port_vol = np.sqrt(np.dot(w, np.dot(window.cov(), w)))
            sharpe = port_ret / (port_vol + 1e-6)
            strat_scores[strat] = sharpe
        # Bayesian selection: pick strategy with highest expected reward
        best_strat = max(strat_scores, key=strat_scores.get)
        confidence = norm.cdf(strat_scores[best_strat], np.mean(list(strat_scores.values())), np.std(list(strat_scores.values()))+1e-6)
        strategy_timeline.append(best_strat)
        confidence_scores.append(confidence)
        allocations.append(w)
    timeline_idx = returns.index[lookback:]
    return pd.Series(strategy_timeline, index=timeline_idx), pd.Series(confidence_scores, index=timeline_idx), pd.DataFrame(allocations, index=timeline_idx, columns=returns.columns)

# =====================
# ADAPTIVE REBALANCING
# =====================
def adaptive_rebalance(strategy_timeline, regime_df):
    # Variable rebalance frequency based on regime changes
    rebalance_points = [0]
    for i in range(1, len(strategy_timeline)):
        if strategy_timeline.iloc[i] != strategy_timeline.iloc[i-1] or not regime_df.iloc[i].equals(regime_df.iloc[i-1]):
            rebalance_points.append(i)
    return rebalance_points

# =====================
# PERFORMANCE METRICS
# =====================
def compute_performance(returns, allocations, rebalance_points):
    port_returns = []
    last_w = allocations.iloc[0].values
    for i in range(len(allocations)):
        if i in rebalance_points:
            last_w = allocations.iloc[i].values
        port_ret = np.dot(last_w, returns.iloc[i].values)
        port_returns.append(port_ret)
    port_returns = pd.Series(port_returns, index=allocations.index)
    equity_curve = (1 + port_returns).cumprod()
    rolling_sharpe = port_returns.rolling(30).mean() / (port_returns.rolling(30).std() + 1e-6) * np.sqrt(252)
    drawdown = equity_curve / equity_curve.cummax() - 1
    return equity_curve, drawdown, rolling_sharpe

# =====================
# MAIN APP LOGIC
    # --- Spectral Market Decomposition Engine ---
    st.markdown("<h1 class='neon'>Spectral Market Decomposition Engine</h1>", unsafe_allow_html=True)
    st.markdown("<b>Rolling Spectral Surface</b>", unsafe_allow_html=True)
    # Use portfolio returns for FFT
    port_ret = np.dot(allocations.values, returns.iloc[lookback:].values.T)
    port_ret = np.mean(port_ret, axis=0)
    window = 64
    step = 8
    freq_list = []
    amp_list = []
    time_list = []
    for i in range(0, len(port_ret)-window, step):
        seg = port_ret[i:i+window]
        fft = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(window, d=1)
        amps = np.abs(fft)
        freq_list.append(freqs)
        amp_list.append(amps)
        time_list.append(i)
    freq_arr = np.array(freq_list)
    amp_arr = np.array(amp_list)
    time_arr = np.array(time_list)
    # 3D surface plot
    surface_fig = go.Figure(data=[go.Surface(
        z=amp_arr.T,
        x=time_arr,
        y=freq_arr[0],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Amplitude', tickfont=dict(color='#00FFC6'), bgcolor='#181A20'),
    )])
    surface_fig.update_layout(
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Frequency',
            zaxis_title='Amplitude',
            xaxis=dict(backgroundcolor='#181A20', color='#00FFC6'),
            yaxis=dict(backgroundcolor='#181A20', color='#FF00FF'),
            zaxis=dict(backgroundcolor='#181A20', color='#FFAA00'),
        ),
        template='plotly_dark',
        plot_bgcolor='#181A20',
        paper_bgcolor='#181A20',
        font=dict(color='#00FFC6'),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        title='Rolling Spectral Surface'
    )
    st.plotly_chart(surface_fig, use_container_width=True)

    # Dominant frequency/amplitude in last window
    last_fft = amp_arr[-1]
    last_freqs = freq_arr[-1]
    dom_idx = np.argmax(last_fft)
    dom_freq = last_freqs[dom_idx]
    dom_amp = last_fft[dom_idx]
    st.metric("Dominant Frequency (Hz)", f"{dom_freq:.4f}")
    st.metric("Dominant Amplitude", f"{dom_amp:.4f}")

    # FFT spectrum for last window
    fft_fig = go.Figure()
    fft_fig.add_trace(go.Scatter(x=last_freqs, y=last_fft, mode='lines+markers', line=dict(color='#00FFC6', width=2), name='Amplitude'))
    fft_fig.add_trace(go.Scatter(x=[dom_freq], y=[dom_amp], mode='markers', marker=dict(size=12, color='#FF00FF'), name='Dominant Frequency'))
    fft_fig.update_layout(
        title='FFT Spectrum (Last Window)',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        template='plotly_dark',
        plot_bgcolor='#181A20',
        paper_bgcolor='#181A20',
        font=dict(color='#00FFC6'),
        height=350
    )
    st.plotly_chart(fft_fig, use_container_width=True)
    # 3D Portfolio Allocation Evolution
    st.markdown("<h2 class='neon'>3D Portfolio Allocation Evolution</h2>", unsafe_allow_html=True)
    # Use last 30 points for speed
    alloc_3d = allocations.tail(30)
    assets = alloc_3d.columns.tolist()
    dates = alloc_3d.index.strftime('%Y-%m-%d').tolist()
    fig3d = go.Figure()
    for i, asset in enumerate(assets):
        fig3d.add_trace(go.Scatter3d(
            x=list(range(len(dates))),
            y=[i]*len(dates),
            z=alloc_3d[asset],
            mode='lines+markers',
            line=dict(color=f'rgb({50+50*i},{255-50*i},{255})', width=6),
            marker=dict(size=6, color=f'rgb({50+50*i},{255-50*i},{255})', symbol='circle'),
            name=asset,
            text=dates
        ))
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title='Time', tickvals=list(range(len(dates))), ticktext=dates, backgroundcolor='#181A20', color='#00FFC6'),
            yaxis=dict(title='Asset', tickvals=list(range(len(assets))), ticktext=assets, backgroundcolor='#181A20', color='#FF00FF'),
            zaxis=dict(title='Allocation', backgroundcolor='#181A20', color='#FFAA00'),
        ),
        template='plotly_dark',
        plot_bgcolor='#181A20',
        paper_bgcolor='#181A20',
        font=dict(color='#00FFC6'),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        title='Portfolio Allocations Over Time (3D)'
    )
    st.plotly_chart(fig3d, use_container_width=True)
# =====================
st.title("Portfolio Meta-Learning Engine")
st.markdown("<span class='neon' style='font-size:36px;'>Portfolio Meta-Learning Engine</span>", unsafe_allow_html=True)
st.markdown("""
    <div style='font-size:22px; color:#00FFC6; margin-bottom:24px;'>
    Institutional-grade quant dashboard for adaptive portfolio construction.<br>
    <span style='color:#FF00FF;'>Meta-learning</span> selects optimal strategies by regime.<br>
    </div>
""", unsafe_allow_html=True)

# Load data
st.markdown("---")
st.markdown("---")
st.markdown("<div style='text-align:center; color:#00FFC6; font-size:18px;'>Built for institutional quant research. Powered by meta-learning.</div>", unsafe_allow_html=True)



try:
    df = load_data(None, None, None, None, None)
    if df.empty:
        raise ValueError("No data loaded.")
    returns, vol, corr, trend, meanrev = compute_features(df, lookback)
    regime_df = detect_regimes(vol, corr, trend, regime_sensitivity)
    strategy_timeline, confidence_scores, allocations = bayesian_strategy_selection(returns, strategy_set, regime_df)
    rebalance_points = adaptive_rebalance(strategy_timeline, regime_df)
    equity_curve, drawdown, rolling_sharpe = compute_performance(returns.iloc[lookback:], allocations, rebalance_points)

    # --- Spectral Market Decomposition Engine ---
    st.markdown("<h1 class='neon'>Spectral Market Decomposition Engine</h1>", unsafe_allow_html=True)
    st.markdown("<b>Rolling Spectral Surface</b>", unsafe_allow_html=True)
    # Use portfolio returns for FFT
    port_ret = np.dot(allocations.values, returns.iloc[lookback:].values.T)
    port_ret = np.mean(port_ret, axis=0)
    window = 64
    step = 8
    freq_list = []
    amp_list = []
    time_list = []
    for i in range(0, len(port_ret)-window, step):
        seg = port_ret[i:i+window]
        fft = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(window, d=1)
        amps = np.abs(fft)
        freq_list.append(freqs)
        amp_list.append(amps)
        time_list.append(i)
    freq_arr = np.array(freq_list)
    amp_arr = np.array(amp_list)
    time_arr = np.array(time_list)
    # 3D surface plot
    surface_fig = go.Figure(data=[go.Surface(
        z=amp_arr.T,
        x=time_arr,
        y=freq_arr[0],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Amplitude', tickfont=dict(color='#00FFC6'), bgcolor='#181A20'),
    )])
    surface_fig.update_layout(
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Frequency',
            zaxis_title='Amplitude',
            xaxis=dict(backgroundcolor='#181A20', color='#00FFC6'),
            yaxis=dict(backgroundcolor='#181A20', color='#FF00FF'),
            zaxis=dict(backgroundcolor='#181A20', color='#FFAA00'),
        ),
        template='plotly_dark',
        plot_bgcolor='#181A20',
        paper_bgcolor='#181A20',
        font=dict(color='#00FFC6'),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        title='Rolling Spectral Surface'
    )
    st.plotly_chart(surface_fig, use_container_width=True)

    # Dominant frequency/amplitude in last window
    last_fft = amp_arr[-1]
    last_freqs = freq_arr[-1]
    dom_idx = np.argmax(last_fft)
    dom_freq = last_freqs[dom_idx]
    dom_amp = last_fft[dom_idx]
    st.metric("Dominant Frequency (Hz)", f"{dom_freq:.4f}")
    st.metric("Dominant Amplitude", f"{dom_amp:.4f}")

    # FFT spectrum for last window
    fft_fig = go.Figure()
    fft_fig.add_trace(go.Scatter(x=last_freqs, y=last_fft, mode='lines+markers', line=dict(color='#00FFC6', width=2), name='Amplitude'))
    fft_fig.add_trace(go.Scatter(x=[dom_freq], y=[dom_amp], mode='markers', marker=dict(size=12, color='#FF00FF'), name='Dominant Frequency'))
    fft_fig.update_layout(
        title='FFT Spectrum (Last Window)',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        template='plotly_dark',
        plot_bgcolor='#181A20',
        paper_bgcolor='#181A20',
        font=dict(color='#00FFC6'),
        height=350
    )
    st.plotly_chart(fft_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    # Show a default graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1,2,3,4,5], y=[1,4,9,16,25], line=dict(color="#00FFC6", width=3)))
    fig.update_layout(title="Demo Graph", template="plotly_dark", plot_bgcolor="#181A20", paper_bgcolor="#181A20", font=dict(color="#00FFC6"), height=400)
    st.plotly_chart(fig, use_container_width=True)

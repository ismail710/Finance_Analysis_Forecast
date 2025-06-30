import streamlit as st

st.set_page_config(page_title="Finance Analysis", layout="wide")

# --- Navigation ---
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
nav_spacer1, nav_col1, nav_col2, nav_spacer2 = st.columns([2, 3, 3, 2])
with nav_col1:
    if st.button("📈 Analysis", use_container_width=True):
        st.switch_page("pages/1_AnalysisFinance.py")
with nav_col2:
    if st.button("🚀 Forecast", use_container_width=True):
        st.switch_page("pages/2_AdvancedForecast.py")

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import numpy as np
from scipy.stats import norm, shapiro, jarque_bera
import statsmodels.api as sm

# --- Shared State ---
ticker = st.session_state.get('ticker', 'AAPL')
start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
end_date = st.session_state.get('end_date', datetime(2023, 1, 1))

st.title("Financial Analysis Dashboard")

# Sidebar for user input
st.sidebar.header("Settings")
ma_window = st.sidebar.slider("Moving Average/Volatility Window (days)", min_value=5, max_value=60, value=20, step=1)

# Fetch stock data
ticker_obj = yf.Ticker(ticker)
stock_data = ticker_obj.history(start=start_date, end=end_date)

if stock_data.empty:
    st.warning("No data found for the selected ticker and date range.")
else:
    # Moving average and volatility
    stock_data['MA'] = stock_data['Close'].rolling(window=ma_window).mean()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=ma_window).std()

    # Plot closing price
    st.subheader(f"{ticker} Closing Price Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig1.update_layout(title=f'{ticker} Closing Price ({start_date} to {end_date})',
                       xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig1, use_container_width=True)

    # Daily returns
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    hist_data = stock_data['Daily Return'].dropna()
    mean, std = hist_data.mean(), hist_data.std()

    # Normal fit with correct scaling
    st.subheader(f"{ticker} Daily Return Distribution")
    nbins = 50
    counts, bins = np.histogram(hist_data, bins=nbins)
    bin_width = bins[1] - bins[0]
    x = np.linspace(hist_data.min(), hist_data.max(), 100)
    normal_fit = norm.pdf(x, mean, std) * len(hist_data) * bin_width

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=hist_data, nbinsx=nbins, marker_color='blue', opacity=0.7, name='Histogram'))
    fig2.add_trace(go.Scatter(x=x, y=normal_fit, mode='lines', name='Normal Fit', line=dict(color='red')))
    fig2.update_layout(title=f'{ticker} Daily Return Distribution with Normal Fit',
                       xaxis_title='Daily Return', yaxis_title='Frequency')
    st.plotly_chart(fig2, use_container_width=True)

    # Stats
    st.write("### Distribution Statistics")
    st.write(f"Mean: {mean:.4f}")
    st.write(f"Standard Deviation: {std:.4f}")
    # QQ-Plot for Daily Returns
    st.subheader(f"{ticker} QQ-Plot of Daily Returns")
    qq_fig = go.Figure()
    qq_data = sm.ProbPlot(hist_data, dist=norm, fit=True)  # Pass norm object instead of string
    qq_line = qq_data.theoretical_quantiles
    qq_points = qq_data.sample_quantiles
     # Normality Tests
    st.write("### Normality Tests")
    shapiro_stat, shapiro_p = shapiro(hist_data)
    jb_stat, jb_p = jarque_bera(hist_data)
    st.write(f"Shapiro-Wilk Test p-value: {shapiro_p:.4f}")
    st.write(f"Jarque-Bera Test p-value: {jb_p:.4f}")
    
    qq_fig.add_trace(go.Scatter(x=qq_line, y=qq_points, mode='markers', name='QQ-Plot Points'))
    qq_fig.add_trace(go.Scatter(x=qq_line, y=qq_line, mode='lines', name='45-degree Line', line=dict(color='red')))
    qq_fig.update_layout(title=f'{ticker} QQ-Plot of Daily Returns',
                         xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    st.plotly_chart(qq_fig, use_container_width=True)
    # Plot price with moving average
    st.subheader(f"{ticker} Price with {ma_window}-day Moving Average")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA'], mode='lines',
                              name=f'{ma_window}-day MA', line=dict(dash='dash')))
    fig4.update_layout(title=f'{ticker} Price with {ma_window}-day Moving Average',
                       xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig4, use_container_width=True)

   

    # Download
    st.download_button(
        label="Download CSV",
        data=stock_data.to_csv().encode('utf-8'),
        file_name=f'{ticker}_historical_data.csv',
        mime='text/csv'
    )

    # --- Candlestick Chart with Support Levels ---
    st.subheader(f"{ticker} Candlestick Chart with Support Levels")
    fig_candle = go.Figure(data=[
        go.Candlestick(x=stock_data.index,
                       open=stock_data['Open'],
                       high=stock_data['High'],
                       low=stock_data['Low'],
                       close=stock_data['Close'])
    ])
    fig_candle.update_layout(title=f'{ticker} Candlestick Chart ({start_date} to {end_date})',
                             xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig_candle, use_container_width=True)
    

# --- Comparison Section ---
st.subheader("📈 Stock Comparison: Performance, Risk, and Fundamentals")
compare_tickers = st.text_area("Enter additional stock tickers separated by commas", "MSFT, GOOGL, META")
compare_tickers = [ticker] + [t.strip() for t in compare_tickers.split(',') if t.strip()]
compare_start = st.date_input("Comparison Start Date", datetime(2020, 1, 1))
compare_end = st.date_input("Comparison End Date", datetime(2024, 12, 31))

if len(compare_tickers) >= 2:
    comp_data = pd.DataFrame()
    for t in compare_tickers:
        ticker_obj = yf.Ticker(t)
        data = ticker_obj.history(start=compare_start, end=compare_end)['Close']
        comp_data[t] = data

    # Normalized prices
    st.write("### Normalized Price Comparison")
    norm = comp_data / comp_data.iloc[0]
    fig = go.Figure()
    for t in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], mode='lines', name=t))
    fig.update_layout(title="Normalized Price Comparison", xaxis_title="Date", yaxis_title="Normalized Price")
    st.plotly_chart(fig, use_container_width=True)

    # Volatility
    returns = comp_data.pct_change().dropna()
    volatility = returns.std()
    st.write("### Volatility (Std Dev of Returns)")
    st.dataframe(volatility.rename("Volatility"))

    # Beta
    st.write("### Beta (vs S&P 500)")
    market_obj = yf.Ticker('^GSPC')
    market = market_obj.history(start=compare_start, end=compare_end)['Close']
    market_returns = market.pct_change().dropna()
    betas = {}
    for t in compare_tickers:
        if t in returns.columns:
            stock_series = returns[t]
            aligned = pd.concat([stock_series, market_returns], axis=1).dropna()
            aligned.columns = [t, 'Market']
            if aligned.shape[0] > 1:
                cov = aligned[t].cov(aligned['Market'])
                var = aligned['Market'].var()
                betas[t] = cov / var if var != 0 else float('nan')
            else:
                betas[t] = float('nan')
    st.dataframe(pd.Series(betas, name="Beta"))

    # Sharpe Ratio
    st.write("### Sharpe Ratio")
    risk_free_rate = 0.02 / 252
    sharpe_ratios = (returns.mean() - risk_free_rate) / returns.std()
    st.dataframe(sharpe_ratios.rename("Sharpe Ratio"))

    # Financial Ratios
    st.write("### Key Financial Ratios")
    pe, pb, roe, de, dy = [], [], [], [], []
    for t in compare_tickers:
        info = yf.Ticker(t).info
        pe.append(info.get('trailingPE'))
        pb.append(info.get('priceToBook'))
        roe.append(info.get('returnOnEquity'))
        de.append(info.get('debtToEquity'))
        dy.append(info.get('dividendYield'))
    ratios = pd.DataFrame({
        'P/E': pe,
        'P/B': pb,
        'ROE': roe,
        'Debt/Equity': de,
        'Dividend Yield': dy
    }, index=compare_tickers)
    st.dataframe(ratios)
else:
    st.info("Enter at least two stock tickers for detailed comparison.")

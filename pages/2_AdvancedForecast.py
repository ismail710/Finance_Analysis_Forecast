import streamlit as st
st.set_page_config(page_title="Finance Analysis", layout="wide")
from datetime import datetime

# --- Navigation ---
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
nav_spacer1, nav_col1, nav_col2, nav_spacer2 = st.columns([2, 3, 3, 2])
with nav_col1:
    if st.button("📈 Analysis", use_container_width=True):
        st.switch_page("pages/1_AnalysisFinance.py")
with nav_col2:
    if st.button("🚀 Forecast", use_container_width=True):
        st.switch_page("pages/2_AdvancedForecast.py")

# --- Shared State ---
ticker = st.session_state.get('ticker', 'AAPL')
start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
end_date = st.session_state.get('end_date', datetime(2023, 1, 1))

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import base64
from PIL import Image
import requests
from io import BytesIO
import io
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error # Import here for use in multiple models

# Import necessary libraries for models outside try blocks
try:
    from statsmodels.tsa.arima.model import ARIMA
    try:
        from pmdarima import auto_arima
    except ImportError:
        auto_arima = None
except ImportError:
    ARIMA = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from sklearn.preprocessing import MinMaxScaler as KerasMinMaxScaler # Rename to avoid conflict
except ImportError:
    Sequential = None
    Dense = None
    Dropout = None
    LSTM = None
    KerasMinMaxScaler = None

# Load custom CSS
with open('.streamlit/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar for user input (period/auto-run only)
st.sidebar.header("Settings")


default_start = datetime(2020, 1, 1)
default_end = datetime(2024, 12, 31)
col1, col2 = st.sidebar.columns(2)
with col1:
    period = st.selectbox(
        "📅 Timeframe",
        ( "1M", "6M"), # Corrected typo 5Y -> 1Y
        index=1,
        help="Select the time period for analysis"
    )
with col2:
    auto_run = st.checkbox("🔄 Auto-run", value=True, help="Automatically update when inputs change")
if not auto_run:
    button = st.sidebar.button("🚀 Analyze", use_container_width=True)
else:
    button = True

# Format market cap and enterprise value into something readable
def format_value(value):
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0
    while value and value >= 1000 and suffix_index < len(suffixes) - 1:
        value /= 1000
        suffix_index += 1
    return f"${value:.1f}{suffixes[suffix_index]}" if value else "N/A"

def safe_format(value, fmt="{:.2f}", fallback="N/A"):
    try:
        return fmt.format(value) if value is not None else fallback
    except (ValueError, TypeError):
        return fallback

# Add a function to create metric cards
def metric_card(title, value, delta=None):
    if delta:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color="normal"
        )
    else:
        st.metric(
            label=title,
            value=value
        )

# If button (or auto-run) is active
if button:
    if not ticker.strip():
        st.error("⚠️ Please provide a valid stock ticker.")
    else:
        # Fetch historical data for the selected ticker and period
         with st.spinner('📊 Fetching data...'):
                # Retrieve stock data
                stock = yf.Ticker(ticker)
                info = stock.info

                # Create header with company logo
                logo_url = info.get('logo_url')
                if logo_url:
                    try:
                        response = requests.get(logo_url)
                        img = Image.open(BytesIO(response.content))
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(img, width=100)
                        with col2:
                            st.title(f"{info.get('longName', ticker)}")
                    except:
                        st.title(f"{info.get('longName', ticker)}")
                else:
                    st.title(f"{info.get('longName', ticker)}")

                st.markdown("---")

                # Quick metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    metric_card("Current Price", 
                              safe_format(info.get('currentPrice'), fmt="${:.2f}"),
                              f"{safe_format(info.get('regularMarketChangePercent')*100 if info.get('regularMarketChangePercent') else None, fmt='{:+.2f}%')}")
                with col2:
                    metric_card("Market Cap", format_value(info.get('marketCap')))
                with col3:
                    metric_card("P/E Ratio", safe_format(info.get('forwardPE')))
                with col4:
                    metric_card("52W Range", f"${safe_format(info.get('fiftyTwoWeekLow'))} - ${safe_format(info.get('fiftyTwoWeekHigh'))}")

                # Plot historical stock price data
                period_map = {
                    "1D": ("1d", "1h"),
                    "5D": ("5d", "1d"),
                    "1M": ("1mo", "1d"),
                    "6M": ("6mo", "1wk"),
                    "YTD": ("ytd", "1mo"),
                    "1Y": ("1y", "1mo"),
                    "5Y": ("5y", "3mo"),
                }
                selected_period, interval = period_map.get(period, ("1mo", "1d"))
                
                # Use start and end date for stock information
                history = stock.history(start=start_date, end=end_date, interval=interval)

                # --- Common Target Column Selection ---
                if not history.empty:
                    available_cols = [col for col in history.columns if history[col].dtype != 'O']
                    target_col = st.selectbox('Select column to forecast (applies to all models)', available_cols, index=available_cols.index('Close') if 'Close' in available_cols else 0, key='target_col_common')
                else:
                    target_col = 'Close'

                chart_data = pd.DataFrame(history[target_col])
                st.line_chart(chart_data)

                col1, col2, col3 = st.columns(3)

                # Stock Info
                stock_info = [
                    ("Stock Info", "Value"),
                    ("Country", info.get('country', 'N/A')),
                    ("Sector", info.get('sector', 'N/A')),
                    ("Industry", info.get('industry', 'N/A')),
                    ("Market Cap", format_value(info.get('marketCap'))),
                    ("Enterprise Value", format_value(info.get('enterpriseValue'))),
                    ("Employees", info.get('fullTimeEmployees', 'N/A'))
                ]
                df = pd.DataFrame(stock_info[1:], columns=stock_info[0]).astype(str)
                col1.dataframe(df, width=400, hide_index=True)

                # Price Info
                price_info = [
                    ("Price Info", "Value"),
                    ("Current Price", safe_format(info.get('currentPrice'), fmt="${:.2f}")),
                    ("Previous Close", safe_format(info.get('previousClose'), fmt="${:.2f}")),
                    ("Day High", safe_format(info.get('dayHigh'), fmt="${:.2f}")),
                    ("Day Low", safe_format(info.get('dayLow'), fmt="${:.2f}")),
                    ("52 Week High", safe_format(info.get('fiftyTwoWeekHigh'), fmt="${:.2f}")),
                    ("52 Week Low", safe_format(info.get('fiftyTwoWeekLow'), fmt="${:.2f}")),
                    ("Price Change", safe_format(info.get('regularMarketChange'), fmt="${:.2f}")),
                    ("% Change", safe_format(info.get('regularMarketChangePercent') * 100, fmt="{:+.2f}%") if info.get('regularMarketChangePercent') is not None else "N/A")
                ]
                df = pd.DataFrame(price_info[1:], columns=price_info[0]).astype(str)
                col2.dataframe(df, width=400, hide_index=True)

                # Business Metrics
                biz_metrics = [
                    ("Business Metrics", "Value"),
                    ("EPS (FWD)", safe_format(info.get('forwardEps'))),
                    ("P/E (FWD)", safe_format(info.get('forwardPE'))),
                    ("PEG Ratio", safe_format(info.get('pegRatio'))),
                    ("Div Rate (FWD)", safe_format(info.get('dividendRate'), fmt="${:.2f}")),
                    ("Div Yield (FWD)", safe_format(info.get('dividendYield') * 100, fmt="{:.2f}%") if info.get('dividendYield') else 'N/A'),
                    ("Recommendation", info.get('recommendationKey', 'N/A').capitalize())
                ]
                df = pd.DataFrame(biz_metrics[1:], columns=biz_metrics[0]).astype(str)
                col3.dataframe(df, width=400, hide_index=True)

                # --- ARIMA Forecast Section ---
                st.subheader(f"ARIMA {target_col} Forecast")
                if ARIMA and mean_squared_error:
                    try:
                        # Config
                        p = st.number_input('AR (p): Autoregressive order', 0, 10, value=5, key='p_input_arima')
                        d = st.number_input('I (d): Differencing order', 0, 2, value=1, key='d_input_arima')
                        q = st.number_input('MA (q): Moving average order', 0, 10, value=0, key='q_input_arima')
                        use_auto = st.checkbox('Use auto_arima', value=True, key='use_auto_input_arima')
                        forecast_days = st.slider('Forecast horizon (days, ARIMA)', 5, min(30, len(history[target_col].dropna())-1), value=10, key='forecast_days_arima')
                        future_days = st.slider('Future days to predict (ARIMA)', 1, 30, 5, key='future_days_arima')
                        close_prices = history[target_col].dropna().values if target_col in history.columns else []

                        if len(close_prices) < forecast_days + 1:
                             st.warning(f"Not enough data ({len(close_prices)} points) for ARIMA forecast with {forecast_days} forecast days.")
                        else:
                            train, test = close_prices[:-forecast_days], close_prices[-forecast_days:]
                            if use_auto and auto_arima:
                                with st.spinner('Fitting auto_arima...'):
                                    arima_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
                                    order = arima_model.order
                            else:
                                order = (p, d, q)
                            
                            model_arima = ARIMA(train, order=order)
                            model_arima_fit = model_arima.fit()
                            forecast_obj = model_arima_fit.get_forecast(steps=forecast_days)
                            forecast_arima = forecast_obj.predicted_mean
                            conf_int = forecast_obj.conf_int(alpha=0.05)
                            mse_arima = mean_squared_error(test, forecast_arima)
                            
                            # ARIMA Future
                            future_forecast_obj = model_arima_fit.get_forecast(steps=forecast_days + future_days)
                            full_pred_arima = future_forecast_obj.predicted_mean
                            
                            # Plot
                            x_train = history.index[:len(train)]
                            x_test = history.index[len(train):len(train) + len(test)]
                            last_date = history.index[-1]
                            future_index = pd.date_range(start=last_date, periods=future_days+1, freq='B')[1:]
                            
                            fig_arima = go.Figure()
                            fig_arima.add_trace(go.Scatter(x=x_train, y=train, mode='lines', name='Train Actual'))
                            fig_arima.add_trace(go.Scatter(x=x_test, y=test, mode='lines+markers', name='Test Actual'))
                            fig_arima.add_trace(go.Scatter(x=x_test, y=forecast_arima, mode='lines+markers', name=f'ARIMA {forecast_days}-Day Forecast'))
                            fig_arima.add_trace(go.Scatter(x=future_index, y=full_pred_arima[-future_days:], mode='lines+markers', name=f'ARIMA Future {future_days}-Day Forecast', line=dict(dash='dash', color='orange')))
                            fig_arima.update_layout(title=f"ARIMA Forecast for {ticker} ({target_col})", xaxis_title="Date", yaxis_title=target_col, legend_title="Legend", template="plotly_white")
                            st.plotly_chart(fig_arima, use_container_width=True)
                            st.markdown(f"**ARIMA MSE:** {mse_arima:.2f} **Order:** {order}")
                            
                            # Next-Day Forecast as info
                            try:
                                next_forecast = model_arima_fit.get_forecast(steps=forecast_days+1).predicted_mean[-1]
                                next_date = last_date + pd.Timedelta(days=1)
                                st.info(f"ARIMA Predicted {target_col} for {next_date.date()}: ${next_forecast:.2f}")
                            except Exception as e:
                                st.warning(f"ARIMA next-day forecast error: {e}")
                            
                            # Sample predictions table
                            st.markdown(f"**Sample ARIMA Predictions ({target_col}, last 10):**")
                            try:
                                arima_df = pd.DataFrame({
                                    'Date': x_test,
                                    f'Actual_{target_col}': test,
                                    f'Predicted_{target_col}': forecast_arima
                                }).set_index('Date')
                                st.dataframe(arima_df.tail(10))
                            except Exception:
                                pass

                    except Exception as e:
                        st.warning(f"ARIMA forecast unavailable: {e}")
                else:
                    st.warning("ARIMA or required libraries not available.")

                # --- XGBoost Forecast Section ---
                st.subheader(f"XGBoost {target_col} Forecast")
                if xgb and mean_squared_error:
                    try:
                        # Use target_col for XGBoost data and graph labels
                        df_xgb = pd.DataFrame({target_col: history[target_col].dropna()})
                        n_lags = st.number_input('Number of lag days (XGBoost)', min_value=1, max_value=15, value=5, step=1, key='n_lags_xgb')
                        xgb_forecast_days = st.slider('Forecast horizon (days, XGBoost)', 5, min(30, len(df_xgb[target_col])-1), value=10, key='forecast_days_xgb')
                        xgb_future_days = st.slider('Future days to predict (XGBoost)', 1, 30, 5, key='future_days_xgb')
                        
                        if len(df_xgb) < n_lags + xgb_forecast_days:
                             st.warning(f"Not enough data for XGBoost forecast with {n_lags} lags and {xgb_forecast_days} forecast days.")
                        else:
                            for i in range(1, n_lags+1):
                                df_xgb[f'lag_{i}'] = df_xgb[target_col].shift(i)
                            df_xgb = df_xgb.dropna()

                            X = df_xgb.drop(target_col, axis=1).values
                            y = df_xgb[target_col].values
                            X_train, X_test = X[:-xgb_forecast_days], X[-xgb_forecast_days:]
                            y_train, y_test = y[:-xgb_forecast_days], y[-xgb_forecast_days:]
                            
                            model_xgb = xgb.XGBRegressor(n_estimators=500, objective='reg:squarederror', n_jobs=-1)
                            model_xgb.fit(X_train, y_train)
                            y_pred_xgb = model_xgb.predict(X_test)
                            mse_xgb = mean_squared_error(y_test, y_pred_xgb)
                            
                            # XGBoost Future
                            last_vals = list(df_xgb[target_col].values[-n_lags:])
                            preds_xgb = []
                            for i in range(xgb_future_days):
                                input_arr = np.array(last_vals[-n_lags:]).reshape(1, -1)
                                pred = model_xgb.predict(input_arr)[0]
                                preds_xgb.append(pred)
                                last_vals.append(pred)

                            x_train_idx = df_xgb.index[:len(y_train)]
                            x_test_idx = df_xgb.index[len(y_train):len(y_train) + len(y_test)]
                            last_date = df_xgb.index[-1]
                            future_dates = pd.date_range(start=last_date, periods=xgb_future_days+1, freq='B')[1:]

                            fig_xgb = go.Figure()
                            fig_xgb.add_trace(go.Scatter(x=x_train_idx, y=y_train, mode='lines', name='Train Actual'))
                            fig_xgb.add_trace(go.Scatter(x=x_test_idx, y=y_test, mode='lines+markers', name='Test Actual'))
                            fig_xgb.add_trace(go.Scatter(x=x_test_idx, y=y_pred_xgb, mode='lines+markers', name=f'XGBoost {xgb_forecast_days}-Day Forecast'))
                            fig_xgb.add_trace(go.Scatter(x=future_dates, y=preds_xgb, mode='lines+markers', name=f'XGBoost Future {xgb_future_days}-Day Forecast', line=dict(dash='dash', color='orange')))
                            fig_xgb.update_layout(title=f"XGBoost Forecast for {ticker} ({target_col})", xaxis_title="Date", yaxis_title=target_col, legend_title="Legend", template="plotly_white")
                            st.plotly_chart(fig_xgb, use_container_width=True)
                            st.markdown(f"**XGBoost MSE:** {mse_xgb:.2f} **n_lags:** {n_lags}, **n_estimators:** 500")
                            
                            # Next-Day Forecast as info
                            try:
                                last_row = df_xgb.drop(target_col, axis=1).iloc[-1].values.reshape(1, -1)
                                next_pred = model_xgb.predict(last_row)[0]
                                next_date = last_date + pd.Timedelta(days=1)
                                st.info(f"XGBoost Predicted {target_col} for {next_date.date()}: ${next_pred:.2f}")
                            except Exception as e:
                                st.warning(f"XGBoost next-day forecast error: {e}")

                            # Sample predictions table
                            st.markdown(f"**Sample XGBoost Predictions ({target_col}, last 10):**")
                            try:
                                xgb_df_pred = pd.DataFrame({
                                    'Date': x_test_idx,
                                    f'Actual_{target_col}': y_test,
                                    f'Predicted_{target_col}': y_pred_xgb
                                }).set_index('Date')
                                st.dataframe(xgb_df_pred.tail(10))
                            except Exception:
                                pass

                    except Exception as e:
                        st.warning(f"XGBoost forecast unavailable: {e}")
                else:
                    st.warning("XGBoost or required libraries not available.")

                # --- LSTM Forecast Section ---
                st.subheader(f"LSTM {target_col} Forecast")
                if Sequential and Dense and Dropout and LSTM and KerasMinMaxScaler:
                    try:
                        # Use target_col for LSTM data and graph labels
                        lstm_data = history[target_col].dropna().values

                        if len(lstm_data) < 30:
                             st.warning(f"Not enough data ({len(lstm_data)} points) for LSTM forecast. Need at least 30 points.")
                        else:
                            time_step = st.slider('Time step (LSTM)', min_value=5, max_value=30, value=10, step=1, key='time_step_lstm')
                            lstm_forecast_days = st.slider('Forecast horizon (days, LSTM)', 5, min(30, len(lstm_data) - time_step - 1), value=10, key='forecast_days_lstm')
                            lstm_future_days = st.slider('Future days to predict (LSTM)', 1, 30, 5, key='future_days_lstm')

                            if len(lstm_data) <= time_step + lstm_forecast_days:
                                 st.warning(f"Not enough data for LSTM forecast with time step {time_step} and forecast horizon {lstm_forecast_days}.")
                            else:
                                # Split data
                                train_data = lstm_data[:-lstm_forecast_days]
                                test_data = lstm_data[-lstm_forecast_days:]

                                # Scale data (fit on train, transform train and test)
                                scaler = KerasMinMaxScaler()
                                scaled_train = scaler.fit_transform(train_data.reshape(-1, 1))
                                scaled_test = scaler.transform(test_data.reshape(-1, 1))

                                # Function to create sequences
                                def create_dataset(dataset, time_step=1):
                                    X, Y = [], []
                                    for i in range(len(dataset) - time_step):
                                        a = dataset[i:(i + time_step), 0]
                                        X.append(a)
                                        Y.append(dataset[i + time_step, 0])
                                    return np.array(X), np.array(Y)

                                # Prepare training data
                                X_train, y_train = create_dataset(scaled_train, time_step)
                                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

                                # Prepare test data for prediction
                                # To predict the test set, we need sequences starting from the end of the training data
                                combined_train_test_scaled = np.vstack([scaled_train, scaled_test])
                                # We need time_step + len(scaled_test) points to create sequences for the test set
                                test_input_data = combined_train_test_scaled[-(len(scaled_test) + time_step):]
                                X_test, y_test_dummy = create_dataset(test_input_data, time_step)
                                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                                # Build LSTM model
                                model = Sequential()
                                model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
                                model.add(Dropout(0.2))
                                model.add(LSTM(units=50))
                                model.add(Dropout(0.2))
                                model.add(Dense(units=1))
                                model.compile(loss='mse', optimizer='adam')

                                # Fit model
                                model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

                                # Predict on test data
                                scaled_y_pred_lstm = model.predict(X_test)
                                y_pred_lstm = scaler.inverse_transform(scaled_y_pred_lstm).flatten()

                                # Calculate MSE (compare inverse transformed predictions with original test data)
                                mse_lstm = mean_squared_error(test_data, y_pred_lstm)

                                # Generate future predictions
                                future_preds_lstm = []
                                # Start future predictions from the last sequence of the combined data
                                last_batch = combined_train_test_scaled[-time_step:].reshape((1, time_step, 1))
                                for i in range(lstm_future_days):
                                    next_pred_scaled = model.predict(last_batch)
                                    future_preds_lstm.append(next_pred_scaled[0, 0])
                                    # Append the new prediction to the sequence and drop the oldest value
                                    last_batch = np.append(last_batch[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)

                                future_preds_lstm = scaler.inverse_transform(np.array(future_preds_lstm).reshape(-1, 1)).flatten()

                                # Plot
                                train_dates = history.index[:len(train_data)]
                                test_dates = history.index[len(train_data):len(train_data) + len(test_data)]
                                last_test_date = test_dates[-1] if not test_dates.empty else train_dates[-1]
                                future_dates = pd.date_range(start=last_test_date, periods=lstm_future_days + 1, freq='B')[1:]

                                fig_lstm = go.Figure()
                                fig_lstm.add_trace(go.Scatter(x=train_dates, y=train_data, mode='lines', name='Train Actual'))
                                fig_lstm.add_trace(go.Scatter(x=test_dates, y=test_data, mode='lines+markers', name='Test Actual'))
                                fig_lstm.add_trace(go.Scatter(x=test_dates, y=y_pred_lstm, mode='lines+markers', name=f'LSTM {lstm_forecast_days}-Day Forecast'))
                                fig_lstm.add_trace(go.Scatter(x=future_dates, y=future_preds_lstm, mode='lines+markers', name=f'LSTM Future {lstm_future_days}-Day Forecast', line=dict(dash='dash', color='orange')))
                                fig_lstm.update_layout(title=f"LSTM Forecast for {ticker} ({target_col})", xaxis_title="Date", yaxis_title=target_col, legend_title="Legend", template="plotly_white")
                                st.plotly_chart(fig_lstm, use_container_width=True)

                                st.markdown(f"**LSTM MSE:** {mse_lstm:.2f}")

                                # Next-Day Forecast as info
                                try:
                                    # To predict the next day, use the last 'time_step' points from the combined train+test data
                                    combined_scaled_data = np.vstack([scaled_train, scaled_test])
                                    next_day_input = combined_scaled_data[-time_step:].reshape((1, time_step, 1))
                                    next_day_scaled_pred = model.predict(next_day_input)
                                    next_day_pred = scaler.inverse_transform(next_day_scaled_pred).flatten()[0]
                                    next_date = history.index[-1] + pd.Timedelta(days=1)
                                    st.info(f"LSTM Predicted {target_col} for {next_date.date()}: ${next_day_pred:.2f}")
                                except Exception as e:
                                    st.warning(f"LSTM next-day forecast error: {e}")

                                # Sample predictions table
                                st.markdown(f"**Sample LSTM Predictions ({target_col}, last 10):**")
                                try:
                                    lstm_df_pred = pd.DataFrame({
                                        'Date': test_dates,
                                        f'Actual_{target_col}': test_data,
                                        f'Predicted_{target_col}': y_pred_lstm
                                    }).set_index('Date')
                                    st.dataframe(lstm_df_pred.tail(10))
                                except Exception:
                                    pass

                    except Exception as e:
                        st.warning(f"LSTM forecast unavailable: {e}")
                else:
                    st.warning("Keras or required layers/libraries for LSTM not available.")
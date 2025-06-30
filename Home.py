import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Financial Dashboard Home",
    page_icon="📊",
    layout="wide"
)

# --- Shared State Initialization ---
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'AAPL'
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime(2022, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.today()



# Ticker and Date Range Inputs (Shared)
st.text_input("Enter stock ticker", value=st.session_state.ticker, key='ticker')
st.date_input("Start Date", value=st.session_state.start_date, key='start_date')
st.date_input("End Date", value=st.session_state.end_date, key='end_date')


# --- Navigation ---
st.markdown("<div class='centered-nav-container'>", unsafe_allow_html=True)
nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    if st.button("📈 Go to Analysis"):
        st.switch_page("pages/1_AnalysisFinance.py")
with nav_col2:
    if st.button("🚀 Go to Forecast"):
        st.switch_page("pages/2_AdvancedForecast.py")
st.markdown("</div>", unsafe_allow_html=True)
# --- Dashboard Description ---
st.markdown("""
This dashboard lets you:
- Perform 📈 financial analysis
- Run 🔮 forecasting using ARIMA, XGBoost, and LSTM

Use the ticker input and date range above to apply to both pages.
""")

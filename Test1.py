import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.set_page_config(page_title="📈 Global Market Tracker", layout="wide")

st.title("🌍 Global Market Dashboard")

# --- Stock Options ---
markets = {
    "NIFTY 50 (India)": "^NSEI",
    "NASDAQ (US)": "^IXIC",
    "DOW JONES (US)": "^DJI"
}

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
market_name = st.sidebar.selectbox("Select Market Index", list(markets.keys()))
symbol = markets[market_name]

# Stock ticker input
custom_symbol = st.sidebar.text_input(
    "🔍 Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS, AAPL, MSFT):",
    ""
)

if custom_symbol:
    st.sidebar.success(f"Tracking custom stock: {custom_symbol}")
    selected_symbol = custom_symbol
else:
    selected_symbol = symbol

# Time period selector
time_options = {
    "1 Day": "1d",
    "1 Week": "7d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "4 Years": "4y",
    "5 Years": "5y"
}
selected_time = st.sidebar.selectbox("Select Time Range", list(time_options.keys()))

# --- Fetch Data ---
st.info(f"Fetching data for **{selected_symbol}** ({market_name})...")

try:
    data = yf.download(selected_symbol, period=time_options[selected_time], interval="1d")

    if data.empty:
        st.error("⚠️ No data available for this symbol or time range.")
    else:
        data = data.reset_index()
        data["Close Price"] = data["Close"]

        # --- Summary Table ---
        st.subheader("📊 Summary Statistics")
        st.dataframe(data.describe())

        # --- Line Chart ---
        st.subheader("📈 Price Trend")
        st.line_chart(data.set_index("Date")["Close Price"])

        # --- Performance Summary ---
        st.subheader("💡 Performance Summary")

        latest_close = float(data["Close Price"].iloc[-1])
        prev_close = float(data["Close Price"].iloc[-2]) if len(data) > 1 else latest_close
        change = latest_close - prev_close
        pct_change = (change / prev_close * 100) if prev_close != 0 else 0

        total_days = (data["Date"].iloc[-1] - data["Date"].iloc[0]).days
        total_years = total_days / 365 if total_days > 0 else 1
        cagr = (((latest_close / float(data["Close Price"].iloc[0])) ** (1 / total_years)) - 1) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stock Symbol", selected_symbol)
        col2.metric("Latest Close", f"{latest_close:,.2f}")
        col3.metric("Daily % Change", f"{pct_change:+.2f}%")
        col4.metric("CAGR (since start)", f"{cagr:.2f}%")

        # --- Indicator for Market Trend ---
        st.subheader("📉 Market Indicator")
        if pct_change > 0:
            st.success("📈 The market is showing an **uptrend** today.")
        elif pct_change < 0:
            st.error("📉 The market is showing a **downtrend** today.")
        else:
            st.warning("⚖️ The market is stable today.")

        st.caption("💬 *Values auto-update daily using Yahoo Finance (INR for Indian stocks, USD for US markets).*")

except Exception as e:
    st.error(f"❌ Error fetching data: {str(e)}")

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
st.info(f"Fetching data for **{market_name} ({symbol})**...")

try:
    data = yf.download(symbol, period=time_options[selected_time], interval="1d")
    if data.empty:
        st.error("⚠️ No data available for this range.")
    else:
        # Clean Data
        data.reset_index(inplace=True)
        data.rename(columns={"Close": "Close Price"}, inplace=True)

        # --- Summary Stats ---
        st.subheader("📊 Summary Statistics")
        st.dataframe(data.describe())

        # --- Chart ---
        st.subheader("📈 Price Movement Chart")
        st.line_chart(data.set_index("Date")["Close Price"])

        # --- Performance Summary ---
        st.subheader("💡 Performance Summary")

        latest_close = float(data["Close Price"].iloc[-1])
        prev_close = float(data["Close Price"].iloc[-2]) if len(data) > 1 else latest_close
        change = latest_close - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

        # Handle zero division safely
        total_days = (data["Date"].iloc[-1] - data["Date"].iloc[0]).days
        total_years = total_days / 365 if total_days > 0 else 1

        cagr = (((data["Close Price"].iloc[-1] / data["Close Price"].iloc[0]) ** (1 / total_years)) - 1) * 100

        st.metric("Latest Close", f"{latest_close:,.2f}")
        st.metric("Daily Change", f"{change:+.2f}")
        st.metric("Daily % Change", f"{pct_change:+.2f}%")
        st.metric("CAGR (since start)", f"{cagr:.2f}%")

        # --- Notes ---
        st.caption("💬 *All values are in local market currency (INR for NSEI, USD for NASDAQ/DOW JONES)*")

except Exception as e:
    st.error(f"❌ Error fetching data: {e}")

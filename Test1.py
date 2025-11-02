import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np

st.set_page_config(page_title="📊 Real-Time Stock Dashboard", layout="wide")

st.title("📈 Real-Time Stock & Index Dashboard")

# Sidebar input section
st.sidebar.header("🔍 Customize Your View")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. ^NSEI, RELIANCE.NS, ^BSESN)", "^NSEI")

# Time range selection
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "2 Years", "3 Years", "4 Years", "5 Years"]
)

# Calculate date range
end_date = dt.date.today()
if time_range == "1 Day":
    start_date = end_date - dt.timedelta(days=1)
elif time_range == "1 Week":
    start_date = end_date - dt.timedelta(weeks=1)
elif time_range == "1 Month":
    start_date = end_date - dt.timedelta(days=30)
elif time_range == "6 Months":
    start_date = end_date - dt.timedelta(days=182)
elif time_range == "1 Year":
    start_date = end_date - dt.timedelta(days=365)
elif time_range == "2 Years":
    start_date = end_date - dt.timedelta(days=730)
elif time_range == "3 Years":
    start_date = end_date - dt.timedelta(days=1095)
elif time_range == "4 Years":
    start_date = end_date - dt.timedelta(days=1460)
else:
    start_date = end_date - dt.timedelta(days=1825)

st.sidebar.write(f"🗓 Showing data from **{start_date}** to **{end_date}**")

# Fetch data
st.info("⏳ Fetching latest stock data...")
data = yf.download(symbol, start=start_date, end=end_date, progress=False, interval="1d")

if data.empty:
    st.error("⚠️ No data found for this symbol. Try another one (e.g., ^NSEI, ^BSESN, RELIANCE.NS).")
else:
    st.success("✅ Data fetched successfully!")

    # Basic stock info
    stock_info = yf.Ticker(symbol).info
    st.subheader(f"🏦 {stock_info.get('longName', symbol)} ({symbol})")
    st.caption(f"Exchange: {stock_info.get('exchange', 'N/A')} | Currency: {stock_info.get('currency', 'INR')}")

    # Calculate performance indicators
    latest_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[-2] if len(data) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

    if change > 0:
        st.metric("📈 Today's Change", f"+{change:.2f}", f"{pct_change:.2f}% ↑")
    elif change < 0:
        st.metric("📉 Today's Change", f"{change:.2f}", f"{pct_change:.2f}% ↓")
    else:
        st.metric("⏸ No Change", "0.00", "0.00%")

    # Convert volume to crores for Indian stocks
    if symbol.endswith(".NS") or symbol.startswith("^NSE"):
        data["Volume (Cr)"] = data["Volume"] / 10_000_000
    else:
        data["Volume (Cr)"] = data["Volume"]

    # Rename columns
    data = data.rename(columns={
        "Open": "Open Price",
        "High": "High Price",
        "Low": "Low Price",
        "Close": "Close Price",
        "Adj Close": "Adjusted Close",
    })

    # Chart section
    st.subheader(f"📊 {symbol} Price Chart ({time_range})")
    st.line_chart(data["Close Price"], use_container_width=True)

    # Performance summary
    st.subheader("📈 Performance Summary")

    total_years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = ((data["Close Price"].iloc[-1] / data["Close Price"].iloc[0]) ** (1/total_years) - 1) * 100
    volatility = data["Close Price"].pct_change().std() * np.sqrt(252) * 100

    st.write(f"**CAGR (5Y Avg Return):** {cagr:.2f}%")
    st.write(f"**Annualized Volatility:** {volatility:.2f}%")

    # Recent daily data
    st.subheader("📅 Latest Daily Data")
    st.dataframe(data.tail(10))

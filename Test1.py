import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.set_page_config(page_title="Global Index Dashboard", layout="wide")

st.title("🌍 Global Market Index Dashboard (2020–2025)")
st.markdown("**Live-updating dashboard for NIFTY 50, Dow Jones, and NASDAQ — with real-time data from Yahoo Finance.**")

# --- Sidebar Controls ---
st.sidebar.header("📅 Time Range Selector")

range_option = st.sidebar.selectbox(
    "Select Time Range",
    ["5 Years", "1 Year", "6 Months", "1 Month", "1 Week", "1 Day"]
)

# --- Determine Date Range ---
today = datetime.date.today()
if range_option == "5 Years":
    start_date = today - datetime.timedelta(days=5 * 365)
elif range_option == "1 Year":
    start_date = today - datetime.timedelta(days=365)
elif range_option == "6 Months":
    start_date = today - datetime.timedelta(days=180)
elif range_option == "1 Month":
    start_date = today - datetime.timedelta(days=30)
elif range_option == "1 Week":
    start_date = today - datetime.timedelta(days=7)
else:
    start_date = today - datetime.timedelta(days=1)

# --- Index List ---
indices = {
    "^NSEI": "NIFTY 50 🇮🇳",
    "^DJI": "Dow Jones 🇺🇸",
    "^IXIC": "NASDAQ 🇺🇸"
}

# --- Fetch Data ---
st.sidebar.write(f"Fetching data from **{start_date}** to **{today}**")
st.caption("🔁 Data refreshes automatically each time you reload this page (real-time market updates).")

for symbol, name in indices.items():
    st.subheader(f"{name} — {range_option} Chart")
    try:
        data = yf.download(symbol, start=start_date, end=today)
        if not data.empty:
            st.line_chart(data["Close"])
            st.write("**Summary Statistics:**")
            st.dataframe(data.describe())
        else:
            st.warning(f"⚠️ No data found for {name}.")
    except Exception as e:
        st.error(f"Error fetching data for {name}: {e}")

# --- Sidebar Info ---
st.sidebar.info("💡 Tip: Use the sidebar to switch time ranges (5Y / 1Y / 6M / 1M / 1W / 1D).")

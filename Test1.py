# data_fetcher.py
"""
Simplified data fetcher for financial data.
Works perfectly on Android (Pydroid 3).
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import requests_cache
from datetime import datetime

# Enable cache to speed up repeat API calls
requests_cache.install_cache("fetch_cache", expire_after=60 * 60 * 6)  # 6 hours

# -------------------------
# 1. Fetch stock / index data using Yahoo Finance
# -------------------------
def get_price_series_yf(ticker: str, start: str = "2018-01-01", end: str = None, interval: str = "1d"):
    """
    Fetch price data for a stock/index (NIFTY, NASDAQ, etc.)
    Example: df = get_price_series_yf("^NSEI")
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, threads=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# -------------------------
# 2. Fetch Indian Mutual Fund NAVs (from AMFI)
# -------------------------
def get_amfi_navs():
    """
    Fetches Indian mutual fund NAV data from AMFI official source.
    Returns full DataFrame with columns like SchemeName, NAV, Date.
    """
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.text.splitlines()
        rows = []
        for line in data[1:]:
            parts = line.split(";")
            if len(parts) >= 6:
                rows.append(parts[:6])
        df = pd.DataFrame(rows, columns=["SchemeCode", "RTACode", "ISIN", "SchemeName", "NAV", "Date"])
        df["NAV"] = pd.to_numeric(df["NAV"], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["NAV", "Date"])
        return df
    except Exception as e:
        print(f"Error fetching AMFI data: {e}")
        return pd.DataFrame()

# -------------------------
# 3. Compute CAGR (return) and volatility
# -------------------------
def compute_cagr_and_vol(df: pd.DataFrame):
    """
    Compute CAGR and annualized volatility for the given price DataFrame.
    Returns: (CAGR, Volatility)
    """
    if df is None or df.empty:
        return None, None

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        print("No Close column found")
        return None, None

    close = df[col].dropna()
    if len(close) < 10:
        return None, None

    total_years = (close.index[-1] - close.index[0]).days / 365.25
    if total_years <= 0:
        return None, None

    cagr = (close.iloc[-1] / close.iloc[0]) ** (1.0 / total_years) - 1.0
    vol = close.pct_change().dropna().std() * np.sqrt(252)
    return round(cagr * 100, 2), round(vol * 100, 2)

# -------------------------
# 4. Example: test function
# -------------------------
if __name__ == "__main__":
    print("Fetching NIFTY 50 data...")
    df_nifty = get_price_series_yf("^NSEI", start="2020-01-01")
    print(df_nifty.tail())

    cagr, vol = compute_cagr_and_vol(df_nifty)
    print(f"CAGR: {cagr}% | Volatility: {vol}%")

    # Uncomment below to test mutual fund data
    # print("Fetching AMFI NAVs...")
    # df_amfi = get_amfi_navs()
    # print(df_amfi.head())

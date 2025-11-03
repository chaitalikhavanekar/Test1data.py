# dashboard_full.py
"""
Streamlit app - Global Market Tracker + India Business Planning Intelligence Platform
Includes:
- Global Market Dashboard (existing Yahoo Finance features kept unchanged)
- India Economic Dashboard (MOSPI-friendly)
- Government Spending Tracker (IPMD/PAIMANA, data.gov.in state transfers, eProcure awards)
- Live Economy Updates (newsletter-style)
Notes: Prefer public endpoints; when not available, upload CSV exports.
"""

import os
import io
import re
import time
import math
import requests
import requests_cache
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
# --- MOSPI API Setup ---
import requests

API_KEY = "579b464db66ec23bdd0000011174991952a945c65c2d95c58efe0f0"  # 🟢 paste your key here
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"  # example: CPI data

def fetch_mospi_data():
    """Fetch real-time data from MOSPI/Data.gov.in API"""
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?format=json&api-key={API_KEY}&limit=1000"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        return pd.DataFrame(records)
    else:
        st.error("Failed to fetch data from MOSPI API")
        return pd.DataFrame()
# Load .env if present
load_dotenv()

# Simple requests cache
requests_cache.install_cache("gov_cache", expire_after=3600)

# Page config
st.set_page_config(page_title="📈 Global + India Business Planning Dashboard", layout="wide")

# ---------- Helper utilities ----------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def five_years_ago_date():
    return (dt.datetime.now() - dt.timedelta(days=365*5)).date()

@st.cache_data(ttl=3600)
def try_fetch_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def try_fetch_csv(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    content = r.content
    return pd.read_csv(io.BytesIO(content))

def parse_date_col(df: pd.DataFrame):
    for c in df.columns:
        if re.search(r"date|year|period", c, re.I):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    return c
            except Exception:
                continue
    # fallback to first col
    try:
        parsed = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if parsed.notna().sum() > 0:
            return df.columns[0]
    except Exception:
        return None

def parse_value_col(df: pd.DataFrame):
    for c in df.columns:
        if re.search(r"value|index|rate|amount|figure|estimate|value_in", c, re.I):
            try:
                tmp = pd.to_numeric(df[c], errors="coerce")
                if tmp.notna().sum() > 0:
                    return c
            except Exception:
                continue
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    return None

DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")

# ---------- Sidebar: page selector ----------
st.sidebar.title("📂 Select Dashboard")
page = st.sidebar.radio("Go to", [
    "🌍 Global Market Dashboard",
    "🇮🇳 India Economic Dashboard",
    "🏗️ Government Spending Tracker",
    "📰 Live Economy Updates"
])

# -------------------------
# PAGE 1: Global Market Dashboard (existing code preserved)
# -------------------------
if page == "🌍 Global Market Dashboard":
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

# -------------------------
# PAGE 2: India Economic Dashboard
# -------------------------
elif page == "🇮🇳 India Economic Dashboard":
    st.title("🇮🇳 India Economic Dashboard (MOSPI-friendly)")
    st.caption("Shows macro indicators (5-year history). If you have a DATA_GOV_API_KEY it will attempt to fetch automatically; otherwise upload CSVs.")

    st.sidebar.header("🇮🇳 India Dashboard Controls")
    view_mode = st.sidebar.selectbox("View mode", ["Dashboard (KPI & Charts)", "State Map & Compare", "Indicator Uploads & Manual"])
    years = st.sidebar.selectbox("Timeframe (years)", [1,3,5], index=2)
    start_date = dt.date.today() - dt.timedelta(days=365*years)

    # Placeholder dataset identifiers (user can upload CSVs or set DATA_GOV_API_KEY)
    # If you have API resource IDs you can set them here or paste datasets via upload UI
    st.info("You can either let the app try public endpoints (best-effort), set DATA_GOV_API_KEY env var to let the app call data.gov.in, or upload CSVs for indicators.")

    def load_indicator_from_data_gov(resource_id: str, limit=2000):
        if not DATA_GOV_API_KEY:
            st.warning("DATA_GOV_API_KEY not set. Upload CSV or set the key to fetch from data.gov.in.")
            return None
        url = f"https://api.data.gov.in/resource/{resource_id}?api-key={DATA_GOV_API_KEY}&format=json&limit={limit}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            payload = r.json()
            recs = payload.get("records", [])
            df = pd.DataFrame(recs)
            return df
        except Exception as e:
            st.warning(f"Error fetching resource {resource_id}: {e}")
            return None

    # UI: allow users to upload CSVs for key indicators (fast path)
    st.subheader("Quick: Upload indicator CSVs (optional)")
    uploaded_cpi = st.file_uploader("Upload CPI CSV (if you have it)", type=["csv"])
    uploaded_gdp = st.file_uploader("Upload GDP CSV (if you have it)", type=["csv"])
    uploaded_unemp = st.file_uploader("Upload Unemployment CSV (if you have it)", type=["csv"])
    uploaded_iip = st.file_uploader("Upload IIP CSV (if you have it)", type=["csv"])

    def normalize_indicator_df(df: pd.DataFrame):
        df = df.copy()
        date_col = parse_date_col(df)
        val_col = parse_value_col(df)
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        if val_col:
            df["value"] = pd.to_numeric(df[val_col], errors="coerce")
        return df.dropna(subset=["date","value"]).sort_values("date")

    indicators = {}
    # Load uploaded first (priority)
    if uploaded_cpi:
        try:
            df = pd.read_csv(uploaded_cpi)
            indicators["CPI"] = normalize_indicator_df(df)
            st.success("CPI uploaded and parsed.")
        except Exception as e:
            st.warning(f"Unable to parse CPI CSV: {e}")
    if uploaded_gdp:
        try:
            df = pd.read_csv(uploaded_gdp)
            indicators["GDP"] = normalize_indicator_df(df)
            st.success("GDP uploaded and parsed.")
        except Exception as e:
            st.warning(f"Unable to parse GDP CSV: {e}")
    if uploaded_unemp:
        try:
            df = pd.read_csv(uploaded_unemp)
            indicators["UNEMPLOYMENT"] = normalize_indicator_df(df)
            st.success("Unemployment uploaded and parsed.")
        except Exception as e:
            st.warning(f"Unable to parse unemployment CSV: {e}")
    if uploaded_iip:
        try:
            df = pd.read_csv(uploaded_iip)
            indicators["IIP"] = normalize_indicator_df(df)
            st.success("IIP uploaded and parsed.")
        except Exception as e:
            st.warning(f"Unable to parse IIP CSV: {e}")

    # If no uploads and user has API key, attempt fetching common MOSPI datasets (user may still need to provide resource IDs)
    if not indicators and DATA_GOV_API_KEY:
        st.info("No uploads detected — you may paste resource IDs for CPI/GDP/etc. into the inputs below to fetch from data.gov.in.")
        res_cpi = st.text_input("Paste data.gov.in resource id for CPI (optional)", "")
        res_gdp = st.text_input("Paste resource id for GDP (optional)", "")
        res_unemp = st.text_input("Paste resource id for Unemployment (optional)", "")
        res_iip = st.text_input("Paste resource id for IIP (optional)", "")
        if st.button("Fetch indicators from data.gov.in"):
            for label,resid in [("CPI",res_cpi),("GDP",res_gdp),("UNEMPLOYMENT",res_unemp),("IIP",res_iip)]:
                if resid:
                    df = load_indicator_from_data_gov(resid, limit=5000)
                    if df is not None:
                        indicators[label] = normalize_indicator_df(df)
                        st.success(f"Fetched {label} ({len(indicators[label])} rows).")
                    else:
                        st.warning(f"Could not fetch {label}.")

    # Show indicators if any
    if view_mode == "Dashboard (KPI & Charts)":
        st.header("Dashboard: Key Macroeconomic Indicators (India)")

        if not indicators:
            st.info("No indicator data loaded. Upload CSVs or provide data.gov.in resource IDs (if you have them).")
        else:
            # Keep only last `years`
            for key,df in indicators.items():
                df_recent = df[df["date"].dt.date >= start_date]
                indicators[key] = df_recent

            # KPIs
            cols = st.columns(len(indicators) if indicators else 1)
            for i,(k,df) in enumerate(indicators.items()):
                latest = df.iloc[-1]["value"] if len(df)>0 else None
                prev = df.iloc[-2]["value"] if len(df)>1 else latest
                if latest is not None:
                    pct = (latest - prev) / prev * 100 if prev and prev!=0 else 0
                    cols[i].metric(k, f"{latest:,.2f}", f"{pct:+.2f}%")
            st.markdown("---")

            # Charts
            for k,df in indicators.items():
                st.subheader(f"{k} — last {years} years")
                fig = px.line(df, x="date", y="value", title=f"{k} trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(f"📥 Download {k} CSV", data=to_csv_bytes(df), file_name=f"{k.lower()}_{years}yrs.csv")

            # Analytics quick box
            st.subheader("Analytics Insights")
            if "CPI" in indicators and "GDP" in indicators:
                merged = pd.merge(indicators["CPI"].rename(columns={"value":"CPI"}), indicators["GDP"].rename(columns={"value":"GDP"}), on="date", how="inner")
                if len(merged) > 3:
                    corr = merged["CPI"].corr(merged["GDP"])
                    st.write(f"Correlation (CPI vs GDP): **{corr:.2f}**")
                    st.line_chart(merged.set_index("date")[["CPI","GDP"]])
            st.divider()

    elif view_mode == "State Map & Compare":
        st.header("🗺️ State Map & Compare")
        st.write("Upload a state-wise CSV or use SDG/NIF CSV with state-level numbers. Columns expected: state, year/date, value")

        upload_state = st.file_uploader("Upload state-level CSV (columns: state, year/date, value)", type=["csv"])
        if upload_state:
            try:
                sdf = pd.read_csv(upload_state)
                # normalize
                col_state = [c for c in sdf.columns if re.search(r"state|region|name", c, re.I)]
                col_date = parse_date_col(sdf)
                col_val = parse_value_col(sdf)
                if not col_state:
                    st.error("Couldn't find a state column. Please ensure you have a 'state' column.")
                else:
                    state_col = col_state[0]
                    sdf["state_norm"] = sdf[state_col].astype(str).str.strip()
                    if col_date:
                        sdf["date"] = pd.to_datetime(sdf[col_date], errors="coerce")
                    if col_val:
                        sdf["value"] = pd.to_numeric(sdf[col_val], errors="coerce")
                    # choose year
                    years_list = sorted(sdf["date"].dropna().dt.year.unique()) if "date" in sdf else []
                    sel_year = st.selectbox("Select year", options=years_list) if years_list else None
                    if sel_year:
                        sub = sdf[pd.to_datetime(sdf["date"]).dt.year == int(sel_year)]
                    else:
                        sub = sdf
                    # aggregate by state
                    agg = sub.groupby("state_norm")["value"].mean().reset_index()
                    # Try choropleth
                    geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
                    try:
                        geo = try_fetch_json(geojson_url)
                        fig = px.choropleth(agg, geojson=geo, locations="state_norm", color="value",
                                            featureidkey="properties.ST_NM", projection="mercator",
                                            title=f"State values {sel_year if sel_year else ''}")
                        fig.update_geos(fitbounds="locations", visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.warning("Choropleth failed — showing bar chart fallback.")
                        fig = px.bar(agg.sort_values("value", ascending=False), x="state_norm", y="value")
                        st.plotly_chart(fig, use_container_width=True)
                    st.download_button("📥 Download state-level CSV", data=to_csv_bytes(agg), file_name="state_values.csv")
            except Exception as e:
                st.error(f"Error parsing uploaded state CSV: {e}")

    elif view_mode == "Indicator Uploads & Manual":
        st.header("Manual Indicators / Uploads")
        st.info("Upload any other datasets you want the dashboard to use. The app will try to auto-detect date & value columns.")
        up = st.file_uploader("Upload generic indicator CSV", accept_multiple_files=True, type=["csv"])
        if up:
            for f in up:
                try:
                    df = pd.read_csv(f)
                    dfn = normalize_indicator_df(df)
                    st.subheader(f.name)
                    st.line_chart(dfn.set_index("date")["value"])
                    st.download_button(f"📥 Download {f.name}", data=to_csv_bytes(dfn), file_name=f"{f.name}_normalized.csv")
                except Exception as e:
                    st.warning(f"Could not parse {f.name}: {e}")

# -------------------------
# PAGE 3: Government Spending Tracker
# -------------------------
elif page == "🏗️ Government Spending Tracker":
    st.title("🏗️ Government Spending Tracker — Central Projects, State Transfers, Tenders")
    st.markdown("""
    This module attempts to gather three signals:
    1. **IPMD / PAIMANA (MoSPI)** — central projects ₹150 Cr and above (project list & progress).
    2. **State-wise transfers & scheme expenditure** — from data.gov.in catalog (Centrally Sponsored Schemes, central transfers).
    3. **Awarded tenders (eProcure / CPPP)** — recent procurement awards (proxy for near-real-time committed spend).
    """)
    st.sidebar.header("Spending Tracker Controls")
    st.sidebar.write("If an automatic fetch fails, upload the CSV export from the portal (PAIMANA / data.gov.in / eProcure).")

    # -- IPMD / PAIMANA fetcher (best-effort) --
    st.subheader("1) IPMD / PAIMANA Projects (Central projects >= ₹150 Cr)")
    ipmd_manual = st.file_uploader("Upload PAIMANA/OCMS CSV export (if available)", type=["csv"])
    load_ipmd_btn = st.button("Fetch public IPMD/PAIMANA (best-effort)")
    ipmd_df = None
    if load_ipmd_btn:
        # best-effort attempt at public endpoints — many PAIMANA instances do not expose a stable public API
        st.info("Attempting to fetch PAIMANA/OCMS public export (best-effort). If this fails, please upload CSV from PAIMANA.")
        try:
            # try known public export (may not exist)
            candidate_csvs = [
                "https://ipmd.mospi.gov.in/ExportProjects.csv",  # hypothetical
                "https://ipm.mospi.gov.in/ReportProjectExport",  # hypothetical
            ]
            for u in candidate_csvs:
                try:
                    ipmd_df = try_fetch_csv(u)
                    if not ipmd_df.empty:
                        break
                except Exception:
                    ipmd_df = None
            if ipmd_df is None:
                st.warning("Public PAIMANA export not detected. Please upload a CSV export from PAIMANA.")
            else:
                st.success(f"Loaded {len(ipmd_df)} projects from public export.")
        except Exception as e:
            st.warning(f"IPMD fetch attempt failed: {e}")
            ipmd_df = None

    if ipmd_manual is not None:
        try:
            ipmd_df = pd.read_csv(ipmd_manual)
            st.success(f"Uploaded IPMD CSV ({len(ipmd_df)} rows).")
        except Exception as e:
            st.error(f"Could not parse uploaded IPMD CSV: {e}")

    if ipmd_df is not None:
        # normalize and show summary
        ipmd_df.columns = [c.lower().strip().replace(" ", "_") for c in ipmd_df.columns]
        # guess state column
        state_cols = [c for c in ipmd_df.columns if "state" in c]
        cost_cols = [c for c in ipmd_df.columns if re.search(r"cost|approved|project_cost|est_cost", c)]
        state_col = state_cols[0] if state_cols else None
        cost_col = cost_cols[0] if cost_cols else None

        if state_col:
            counts = ipmd_df[state_col].value_counts().reset_index()
            counts.columns = ["state","projects"]
            st.subheader("Projects by state (top 20)")
            st.dataframe(counts.head(20))
            st.download_button("📥 Download IPMD projects CSV", data=to_csv_bytes(ipmd_df), file_name="ipmd_projects.csv")
        else:
            st.dataframe(ipmd_df.head(50))
            st.download_button("📥 Download IPMD projects CSV", data=to_csv_bytes(ipmd_df), file_name="ipmd_projects.csv")

        if cost_col:
            # try numeric
            ipmd_df["_cost_num"] = pd.to_numeric(ipmd_df[cost_col].astype(str).str.replace(r"[^\d\.]","",regex=True), errors="coerce")
            total_cost = ipmd_df["_cost_num"].sum(skipna=True)
            st.metric("Total approved cost (sum, ₹)", f"{total_cost:,.0f}")

    st.markdown("---")

    # -- State-wise transfers / CSS via data.gov.in --
    st.subheader("2) State-wise transfers & scheme expenditure (data.gov.in)")
    st.write("If you have a data.gov.in resource UUID for a state-wise transfers dataset, paste it below. Otherwise upload a CSV export from data.gov.in or the ministry website.")
    css_resource_id = st.text_input("Paste data.gov.in resource id for state-wise transfers (optional)", "")
    css_manual = st.file_uploader("Or upload state-wise transfers CSV", type=["csv"])
    css_df = None
    if css_resource_id:
        if not DATA_GOV_API_KEY:
            st.warning("Set DATA_GOV_API_KEY environment variable to let the app fetch data.gov.in resources. Or upload CSV.")
        else:
            try:
                url = f"https://api.data.gov.in/resource/{css_resource_id}?api-key={DATA_GOV_API_KEY}&format=json&limit=5000"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                payload = r.json()
                recs = payload.get("records", [])
                css_df = pd.DataFrame(recs)
                st.success(f"Fetched {len(css_df)} rows from data.gov.in resource.")
            except Exception as e:
                st.warning(f"Could not fetch resource: {e}")
                css_df = None

    if css_manual is not None:
        try:
            css_df = pd.read_csv(css_manual)
            st.success(f"Uploaded CSS/state transfers CSV ({len(css_df)} rows).")
        except Exception as e:
            st.error(f"Could not parse uploaded CSS CSV: {e}")

    if css_df is not None:
        # try to detect state & amount/year columns
        css_df.columns = [c.lower().strip().replace(" ", "_") for c in css_df.columns]
        state_cols = [c for c in css_df.columns if "state" in c]
        amount_cols = [c for c in css_df.columns if re.search(r"amount|transfer|release|expenditure|exp", c)]
        date_cols = [c for c in css_df.columns if re.search(r"year|date|period", c)]
        state_col = state_cols[0] if state_cols else None
        amount_col = amount_cols[0] if amount_cols else None
        date_col = date_cols[0] if date_cols else None

        if state_col and amount_col:
            css_df["_amt_num"] = pd.to_numeric(css_df[amount_col].astype(str).str.replace(r"[^\d\.]","",regex=True), errors="coerce")
            agg = css_df.groupby(state_col)["_amt_num"].sum().reset_index().rename(columns={state_col:"state","_amt_num":"total_amount"})
            st.subheader("State-wise total (sum of amount column)")
            st.dataframe(agg.sort_values("total_amount", ascending=False).head(30))
            st.download_button("📥 Download state transfer summary CSV", data=to_csv_bytes(agg), file_name="state_transfers_summary.csv")
        else:
            st.warning("Couldn't detect state/amount columns automatically. Please ensure CSV has clear state and amount columns.")
            st.dataframe(css_df.head(20))
            st.download_button("📥 Download raw CSV", data=to_csv_bytes(css_df), file_name="css_raw.csv")

    st.markdown("---")

    # -- eProcure awarded tenders (central) --
    st.subheader("3) Awarded Tenders (eProcure / CPPP — recent awards)")
    st.write("This shows recent awarded tenders from the central portal as a proxy for near-real-time commitments. If fetching is blocked, upload CSV/MIS export from eProcure or state eProcure portals.")
    eproc_state = st.text_input("Optional: filter awards by state name (e.g., Maharashtra)", "")
    eproc_manual = st.file_uploader("Upload eProcure award CSV/MIS (optional)", type=["csv"])
    fetch_eproc_btn = st.button("Fetch recent awarded tenders (best-effort)")

    awards_df = None
    if fetch_eproc_btn:
        try:
            awards_url = "https://eprocure.gov.in/eprocure/app?page=WebAwards&service=page"
            r = requests.get(awards_url, timeout=30)
            r.raise_for_status()
            # parse HTML tables
            tables = pd.read_html(r.text)
            if tables:
                awards_df = max(tables, key=lambda t: t.shape[0])
                awards_df.columns = [str(c).strip().lower().replace(" ", "_") for c in awards_df.columns]
                st.success(f"Parsed awards table with {len(awards_df)} rows (best-effort).")
            else:
                st.warning("No award tables parsed from central awards page.")
        except Exception as e:
            st.warning(f"Could not auto-fetch eProcure awards: {e}")
            awards_df = None

    if eproc_manual is not None:
        try:
            awards_df = pd.read_csv(eproc_manual)
            st.success(f"Uploaded awards CSV ({len(awards_df)} rows).")
        except Exception as e:
            st.error(f"Could not parse uploaded awards CSV: {e}")

    if awards_df is not None:
        if eproc_state:
            state_cols = [c for c in awards_df.columns if "state" in c]
            if state_cols:
                awards_df = awards_df[awards_df[state_cols[0]].astype(str).str.contains(eproc_state, case=False, na=False)]
        st.dataframe(awards_df.head(200))
        st.download_button("📥 Download awards CSV", data=to_csv_bytes(awards_df), file_name="eprocure_awards.csv")

    st.markdown("---")
    st.info("Tip: For consistent state-level aggregations, canonicalize state names across datasets (I can provide a mapping table if you want).")

# -------------------------
# PAGE 4: Live Economy Updates (newsletter)
# -------------------------
elif page == "📰 Live Economy Updates":
    st.title("📰 Live Economy Updates — Newsletter & Automatic Summary")
    st.write("This section compiles short summaries you can use for a newsletter. It pulls what it can (India via MOSPI/data.gov.in if available) and lets you add manual notes for US/Europe.")
    st.sidebar.header("Newsletter Controls")
    refresh = st.sidebar.button("Refresh feeds (clear cache)")

    if refresh:
        try:
            requests_cache.clear()
            st.success("Cache cleared.")
        except Exception:
            st.warning("Could not clear cache.")

    # India summary from last indicators (re-use India dashboard indicators if uploaded)
    st.subheader("India — Macro snapshot (auto)")
    # Attempt to fetch CPI/GDP indicators from previously uploaded or try public simple sources
    # Quick approach: allow user to upload a one-row CSV with key metrics OR we try small public endpoints
    india_manual = st.file_uploader("Upload a 1-row CSV with keys (gdp,cpi,unemp,iip) for newsletter (optional)", type=["csv"])
    if india_manual:
        try:
            df = pd.read_csv(india_manual)
            st.write(df.to_dict(orient="records")[0])
            st.success("Loaded manual India summary.")
        except Exception as e:
            st.warning(f"Could not parse newsletter CSV: {e}")
    else:
        st.info("No manual India summary provided. If you provided indicator CSVs in India dashboard, use that to craft the summary there.")

    st.subheader("US & Europe — Quick notes (manual)")
    st.text_area("US economy notes (paste / write):", value="", height=120)
    st.text_area("Europe economy notes (paste / write):", value="", height=120)

    st.markdown("### Export newsletter")
    newsletter_title = st.text_input("Newsletter title", value=f"Economic Snapshot — {dt.date.today().isoformat()}")
    newsletter_body = st.text_area("Newsletter body (you can copy the summaries above):", height=240)
    if st.button("Generate newsletter (.txt)"):
        content = f"{newsletter_title}\n\n{newsletter_body}"
        st.download_button("📥 Download newsletter (txt)", data=content.encode("utf-8"), file_name=f"newsletter_{dt.date.today()}.txt")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Notes & Next steps:")
st.sidebar.markdown("""
- This app prefers public endpoints. If an automatic fetch fails, please export CSV from the portal (PAIMANA, data.gov.in, eProcure) and upload it — the app will parse and display it.
- For fully automated production, we recommend setting up a backend ETL (periodic jobs) that saves normalized results to Postgres, and let Streamlit query that DB (faster and scalable).
- If you'd like, I can:
  1) Search data.gov.in for exact resource IDs for CPI/GDP/IPMD/Sdg and pre-fill them for you.  
  2) Build the ETL script to pull/persist data nightly to Postgres and give the Streamlit app endpoints to query.
""")

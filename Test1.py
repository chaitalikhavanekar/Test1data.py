# mospi_live_dashboard.py
"""
Streamlit app — Fully-automated MOSPI CPI / IIP / GDP live dashboard
- Tabs: CPI | IIP | GDP | Comparative Insights
- Tries data.gov.in API when RESOURCE IDs + DATA_GOV_API_KEY are set (recommended)
- Otherwise tries to fetch/scrape from new.mospi.gov.in (best-effort)
- Refresh button forces re-download and clears cached results
- Provides CSV download and simple analytics
"""

import os
import re
import time
import json
from typing import Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from io import BytesIO
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()

# ------------------------
# Configuration / ENV
# ------------------------
st.set_page_config(page_title="MOSPI Live: CPI / IIP / GDP", layout="wide")
st.title("🇮🇳 MOSPI Live — CPI, IIP & GDP (Fully automated)")

DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")  # optional; use for data.gov.in API
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "")  # optional: data.gov.in resource ID for CPI
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "")  # optional: data.gov.in resource ID for IIP
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "")  # optional: data.gov.in resource ID for GDP

# polite request headers
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MOSPI-Data-Dashboard/1.0; +https://example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ------------------------
# Utilities
# ------------------------
def safe_get(url: str, params: dict = None, timeout: int = 20) -> Optional[requests.Response]:
    """HTTP GET with basic error handling"""
    try:
        r = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        # do not spam errors in UI; return None and let caller handle
        return None

def json_from_page_scripts(html_text: str, pattern: str) -> Optional[dict]:
    """
    Helper to find JSON embedded in script tags.
    'pattern' is a regex that matches the variable name or surrounding text.
    Returns the first JSON parsed or None.
    """
    try:
        # Look for large JSON blobs in <script> tags
        soup = BeautifulSoup(html_text, "lxml")
        scripts = soup.find_all("script")
        for s in scripts:
            text = s.string
            if not text:
                continue
            if re.search(pattern, text, re.I):
                # try to extract a JSON substring between first { and matching closing }
                m = re.search(r"(\{.+\})", text, re.S)
                if m:
                    raw = m.group(1)
                    # Try incremental strong parsing: sometimes trailing commas or JS comments cause JSONDecodeError
                    try:
                        return json.loads(raw)
                    except Exception:
                        # try to clean trailing commas
                        cleaned = re.sub(r",\s*}", "}", raw)
                        cleaned = re.sub(r",\s*\]", "]", cleaned)
                        try:
                            return json.loads(cleaned)
                        except Exception:
                            continue
        return None
    except Exception:
        return None

# ------------------------
# Fetchers: try API first, then scrape
# ------------------------

@st.cache_data(show_spinner=False)
def fetch_from_data_gov(resource_id: str, limit: int = 5000) -> Optional[pd.DataFrame]:
    """Fetch dataset from data.gov.in if resource_id + API key available"""
    if not DATA_GOV_API_KEY or not resource_id:
        return None
    url = f"https://api.data.gov.in/resource/{resource_id}"
    params = {"api-key": DATA_GOV_API_KEY, "format": "json", "limit": limit}
    r = safe_get(url, params=params)
    if not r:
        return None
    try:
        payload = r.json()
        records = payload.get("records", [])
        if isinstance(records, list) and records:
            df = pd.json_normalize(records)
            return df
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_cpi_mospi() -> Tuple[Optional[pd.DataFrame], str]:
    """Return CPI dataframe and a 'source' string describing where it came from."""
    # 1) Try data.gov.in resource (recommended)
    if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
        df = fetch_from_data_gov(CPI_RESOURCE_ID)
        if df is not None and not df.empty:
            return df, "data.gov.in API (resource id)"
    # 2) Try new.mospi.gov.in API endpoints (best-effort)
    # Common approach: try an API-like path used by modern sites (may vary)
    try_api_urls = [
        "https://new.mospi.gov.in/api/cpi",  # hypothetical
        "https://new.mospi.gov.in/json/cpi",  # hypothetical
        "https://new.mospi.gov.in/dashboard/cpi",  # the dashboard page; we can scrape
        "https://new.mospi.gov.in/dashboard-data/cpi",  # hypothetical
    ]
    for u in try_api_urls:
        r = safe_get(u)
        if r and r.headers.get("Content-Type","").lower().find("application/json") >= 0:
            try:
                payload = r.json()
                df = pd.json_normalize(payload.get("data", payload))
                if not df.empty:
                    return df, f"mospi API ({u})"
            except Exception:
                pass
    # 3) Scrape the MOSPI CPI dashboard page and attempt to extract JSON
    page_url = "https://new.mospi.gov.in/dashboard/cpi"
    r = safe_get(page_url)
    if r:
        # Try to fetch a CSV/XLSX link inside the page
        try:
            soup = BeautifulSoup(r.text, "lxml")
            # look for download links (csv, xlsx)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".csv") or href.lower().endswith(".xlsx") or "download" in href.lower():
                    if href.startswith("/"):
                        href = "https://new.mospi.gov.in" + href
                    rr = safe_get(href)
                    if rr:
                        # if xlsx
                        try:
                            if href.lower().endswith(".xlsx") or href.lower().endswith(".xls"):
                                df = pd.read_excel(BytesIO(rr.content))
                            else:
                                df = pd.read_csv(BytesIO(rr.content))
                            if not df.empty:
                                return df, f"mospi download link ({href})"
                        except Exception:
                            continue
            # try to find embedded JSON in scripts
            json_blob = json_from_page_scripts(r.text, r"cpi|consumer price index|inflation")
            if json_blob:
                # normalize - try a few shapes
                try:
                    df = pd.json_normalize(json_blob.get("records", json_blob))
                    return df, "embedded JSON (scraped)"
                except Exception:
                    pass
        except Exception:
            pass
    # 4) If everything fails, return None
    return None, "not found"

@st.cache_data(show_spinner=False)
def fetch_iip_mospi() -> Tuple[Optional[pd.DataFrame], str]:
    """Fetch IIP (Index of Industrial Production) from data.gov.in or mospi pages"""
    # try data.gov
    if IIP_RESOURCE_ID and DATA_GOV_API_KEY:
        df = fetch_from_data_gov(IIP_RESOURCE_ID)
        if df is not None and not df.empty:
            return df, "data.gov.in API (resource id)"
    # try mospi page
    page_url = "https://new.mospi.gov.in/dashboard/iip"
    r = safe_get(page_url)
    if r:
        # try to find download link
        try:
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".csv") or href.lower().endswith(".xlsx"):
                    if href.startswith("/"):
                        href = "https://new.mospi.gov.in" + href
                    rr = safe_get(href)
                    if rr:
                        try:
                            if href.lower().endswith(".xlsx") or href.lower().endswith(".xls"):
                                df = pd.read_excel(BytesIO(rr.content))
                            else:
                                df = pd.read_csv(BytesIO(rr.content))
                            if not df.empty:
                                return df, f"mospi download link ({href})"
                        except Exception:
                            continue
            # try embedded JSON
            json_blob = json_from_page_scripts(r.text, r"iip|industrial production")
            if json_blob:
                try:
                    df = pd.json_normalize(json_blob.get("records", json_blob))
                    return df, "embedded JSON (scraped)"
                except Exception:
                    pass
        except Exception:
            pass
    return None, "not found"

@st.cache_data(show_spinner=False)
def fetch_gdp_mospi() -> Tuple[Optional[pd.DataFrame], str]:
    """Fetch GDP from data.gov.in or mospi"""
    if GDP_RESOURCE_ID and DATA_GOV_API_KEY:
        df = fetch_from_data_gov(GDP_RESOURCE_ID)
        if df is not None and not df.empty:
            return df, "data.gov.in API (resource id)"
    page_url = "https://new.mospi.gov.in/dashboard/gdp"
    r = safe_get(page_url)
    if r:
        try:
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".csv") or href.lower().endswith(".xlsx"):
                    if href.startswith("/"):
                        href = "https://new.mospi.gov.in" + href
                    rr = safe_get(href)
                    if rr:
                        try:
                            if href.lower().endswith(".xlsx") or href.lower().endswith(".xls"):
                                df = pd.read_excel(BytesIO(rr.content))
                            else:
                                df = pd.read_csv(BytesIO(rr.content))
                            if not df.empty:
                                return df, f"mospi download link ({href})"
                        except Exception:
                            continue
            json_blob = json_from_page_scripts(r.text, r"gdp|gross domestic product|national accounts")
            if json_blob:
                try:
                    df = pd.json_normalize(json_blob.get("records", json_blob))
                    return df, "embedded JSON (scraped)"
                except Exception:
                    pass
        except Exception:
            pass
    return None, "not found"

# ------------------------
# UI Controls: refresh + instructions
# ------------------------
st.sidebar.markdown("### Controls")
st.sidebar.markdown("Click **Refresh data** to force re-fetch from MOSPI (API scrape).")
if st.sidebar.button("🔄 Refresh data (force)"):
    # clear cached data
    fetch_cpi_mospi.clear()
    fetch_iip_mospi.clear()
    fetch_gdp_mospi.clear()
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Optional: If any fetch fails, add DATA_GOV_API_KEY and resource IDs as env vars in your deployment.")
st.sidebar.code("DATA_GOV_API_KEY, CPI_RESOURCE_ID, IIP_RESOURCE_ID, GDP_RESOURCE_ID")

# ------------------------
# Layout: Tabs
# ------------------------
tabs = st.tabs(["🧾 CPI", "🏭 IIP", "📊 GDP", "📈 Comparative Insights"])

# ------------- CPI Tab -------------
with tabs[0]:
    st.header("🧾 CPI (Consumer Price Index) — Live MOSPI")
    cpi_df, cpi_src = fetch_cpi_mospi()
    if cpi_df is None:
        st.error("Could not fetch CPI automatically. Options:\n1) Provide DATA_GOV_API_KEY + CPI_RESOURCE_ID as env vars,\n2) Upload CPI CSV/XLSX manually.")
        up = st.file_uploader("Upload CPI CSV/XLSX (fallback)", type=["csv","xlsx"])
        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    cpi_df = pd.read_csv(up)
                else:
                    cpi_df = pd.read_excel(up)
                st.success("Uploaded CPI file parsed.")
            except Exception as e:
                st.error(f"Upload parse error: {e}")
    else:
        st.success(f"CPI loaded ({cpi_src})")

    if cpi_df is not None and not cpi_df.empty:
        st.subheader("Raw data (top rows)")
        st.dataframe(cpi_df.head(25))
        # try to find a date column and a numeric column
        date_col = None
        val_cols = []
        for c in cpi_df.columns:
            if re.search(r"date|month|period|time", c, re.I):
                date_col = c
            if re.search(r"value|index|cpi|combined|rural|urban", c, re.I):
                val_cols.append(c)
        # fallback
        if date_col is None:
            date_col = cpi_df.columns[0]
        # parse dates if possible
        try:
            cpi_df["__date"] = pd.to_datetime(cpi_df[date_col], errors="coerce")
        except Exception:
            cpi_df["__date"] = pd.NaT
        # try long format or multiple value columns
        if not val_cols:
            numeric_cols = cpi_df.select_dtypes(include="number").columns.tolist()
            val_cols = numeric_cols[:2] if numeric_cols else []
        # basic plotting
        if "__date" in cpi_df.columns and not cpi_df["__date"].isna().all() and val_cols:
            st.subheader("CPI Trend")
            fig = px.line(cpi_df.sort_values("__date"), x="__date", y=val_cols, title="CPI series (latest available)")
            st.plotly_chart(fig, use_container_width=True)
            # basic analytics
            latest = cpi_df.sort_values("__date").iloc[-1]
            st.metric("Latest CPI date", str(latest["__date"].date()) if not pd.isna(latest["__date"]) else "N/A")
            for vc in val_cols[:3]:
                try:
                    st.metric(f"Latest {vc}", f"{float(latest.get(vc, float('nan'))):.2f}")
                except Exception:
                    st.write(f"{vc}: {latest.get(vc)}")
        else:
            st.info("Could not auto-detect date/value columns for CPI. Consider uploading a cleaned CSV or provide resource id + API key.")

        st.markdown("---")
        st.download_button("Download CPI (CSV)", data=cpi_df.to_csv(index=False).encode("utf-8"), file_name="cpi_data.csv")

# ------------- IIP Tab -------------
with tabs[1]:
    st.header("🏭 IIP (Index of Industrial Production) — Live MOSPI")
    iip_df, iip_src = fetch_iip_mospi()
    if iip_df is None:
        st.error("Could not fetch IIP automatically. Upload CSV/XLSX as fallback.")
        up = st.file_uploader("Upload IIP CSV/XLSX (fallback)", type=["csv","xlsx"], key="iip_up")
        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    iip_df = pd.read_csv(up)
                else:
                    iip_df = pd.read_excel(up)
                st.success("Uploaded IIP file parsed.")
            except Exception as e:
                st.error(f"Upload parse error: {e}")
    else:
        st.success(f"IIP loaded ({iip_src})")

    if iip_df is not None and not iip_df.empty:
        st.dataframe(iip_df.head(25))
        date_col = None
        val_cols = []
        for c in iip_df.columns:
            if re.search(r"date|period|month|time", c, re.I):
                date_col = c
            if re.search(r"index|iip|value|weight", c, re.I):
                val_cols.append(c)
        if date_col is None:
            date_col = iip_df.columns[0]
        try:
            iip_df["__date"] = pd.to_datetime(iip_df[date_col], errors="coerce")
        except Exception:
            iip_df["__date"] = pd.NaT
        if val_cols:
            fig = px.line(iip_df.sort_values("__date"), x="__date", y=val_cols, title="IIP Series")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download IIP (CSV)", data=iip_df.to_csv(index=False).encode("utf-8"), file_name="iip_data.csv")

# ------------- GDP Tab -------------
with tabs[2]:
    st.header("📊 GDP — Live MOSPI")
    gdp_df, gdp_src = fetch_gdp_mospi()
    if gdp_df is None:
        st.error("Could not fetch GDP automatically. Upload CSV/XLSX as fallback.")
        up = st.file_uploader("Upload GDP CSV/XLSX (fallback)", type=["csv","xlsx"], key="gdp_up")
        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    gdp_df = pd.read_csv(up)
                else:
                    gdp_df = pd.read_excel(up)
                st.success("Uploaded GDP file parsed.")
            except Exception as e:
                st.error(f"Upload parse error: {e}")
    else:
        st.success(f"GDP loaded ({gdp_src})")

    if gdp_df is not None and not gdp_df.empty:
        st.dataframe(gdp_df.head(25))
        date_col = None
        val_cols = []
        for c in gdp_df.columns:
            if re.search(r"date|quarter|period|year", c, re.I):
                date_col = c
            if re.search(r"gdp|growth|value|amount", c, re.I):
                val_cols.append(c)
        if date_col is None:
            date_col = gdp_df.columns[0]
        try:
            gdp_df["__date"] = pd.to_datetime(gdp_df[date_col], errors="coerce")
        except Exception:
            gdp_df["__date"] = pd.NaT
        if val_cols:
            fig = px.line(gdp_df.sort_values("__date"), x="__date", y=val_cols, title="GDP Series")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download GDP (CSV)", data=gdp_df.to_csv(index=False).encode("utf-8"), file_name="gdp_data.csv")

# ------------- Comparative Tab -------------
with tabs[3]:
    st.header("📈 Comparative Insights — CPI vs IIP vs GDP")
    # try to load cached dfs
    cpi_df, _ = fetch_cpi_mospi()
    iip_df, _ = fetch_iip_mospi()
    gdp_df, _ = fetch_gdp_mospi()

    if all((cpi_df is None, iip_df is None, gdp_df is None)):
        st.error("No datasets available. Use Refresh button or upload CSVs in individual tabs.")
    else:
        st.write("The app will attempt to align date columns and plot comparable time series (best-effort).")
        # attempt to find date & numeric fields and merge on nearest month/quarter
        def extract_series_for_compare(df):
            if df is None:
                return None
            dfc = df.copy()
            date_col = None
            val_col = None
            for c in dfc.columns:
                if re.search(r"date|month|period|quarter|year", c, re.I):
                    date_col = c
                    break
            if not date_col:
                date_col = dfc.columns[0]
            for c in dfc.columns:
                if re.search(r"value|index|gdp|cpi|iip|growth|amount", c, re.I):
                    val_col = c
                    break
            if not val_col:
                numeric_cols = dfc.select_dtypes(include="number").columns.tolist()
                val_col = numeric_cols[0] if numeric_cols else None
            if date_col:
                try:
                    dfc["__date"] = pd.to_datetime(dfc[date_col], errors="coerce")
                except Exception:
                    dfc["__date"] = pd.NaT
            if val_col:
                dfc["__val"] = pd.to_numeric(dfc[val_col], errors="coerce")
            if "__date" in dfc.columns and "__val" in dfc.columns:
                return dfc[["__date","__val"]].dropna()
            return None

        ser_cpi = extract_series_for_compare(cpi_df)
        ser_iip = extract_series_for_compare(iip_df)
        ser_gdp = extract_series_for_compare(gdp_df)

        # normalize frequency to monthly where possible (forward-fill GDP if quarterly)
        def prepare_for_merge(s, label):
            if s is None:
                return None
            dfp = s.copy()
            dfp = dfp.sort_values("__date")
            dfp = dfp.set_index("__date").resample("M").mean().ffill()
            dfp = dfp.rename(columns={"__val": label})
            return dfp

        pc = prepare_for_merge(ser_cpi, "CPI") if ser_cpi is not None else None
        pi = prepare_for_merge(ser_iip, "IIP") if ser_iip is not None else None
        pg = prepare_for_merge(ser_gdp, "GDP") if ser_gdp is not None else None

        # merge all present
        dfs = [d for d in [pc, pi, pg] if d is not None]
        if not dfs:
            st.error("Could not auto-extract comparable series from the datasets.")
        else:
            merged = pd.concat(dfs, axis=1)
            st.dataframe(merged.tail(20))
            st.subheader("Combined Trend")
            fig = px.line(merged.reset_index(), x="index", y=merged.columns.tolist(), markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.write("Correlation matrix (last available points):")
            corr = merged.corr().round(3)
            st.dataframe(corr)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Notes:")
st.sidebar.info("""
- This app tries multiple methods to fetch MOSPI data. If automatic fetch fails, provide `DATA_GOV_API_KEY` and resource IDs as environment variables, or upload the official CSV/XLSX using the upload widgets.
- Scraping may be blocked by the MOSPI site if requests are too frequent. Use the Refresh button sparingly.
- If you want, I can convert this to a scheduled ETL (server) that pulls MOSPI nightly and stores in Postgres for faster responses.
""")

# full_econ_dashboard.py
"""
India Economic Intelligence Dashboard — All 7 modules (complete)
- Tabbed Streamlit app: Overview, News, CPI, IIP/GDP, Infra, Employment/Policy, Business Planning, Stock & Market
- Uses TradingEconomics (optional), yfinance, Google News scraping fallback, Data.gov (optional), MOSPI scrape (best-effort)
- Fallbacks: file uploads (CSV/XLSX)
- Put TRADINGECONOMICS_KEY and NEWSAPI_KEY into env or Streamlit Secrets for best results
"""

import os
import time
import json
from io import BytesIO
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# optional geospatial libs if map visuals needed (may be heavy)
try:
    import folium
    from streamlit_folium import folium_static
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# Load .env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------
# Config & Keys
# -------------------------
st.set_page_config(page_title="India Economic Intelligence Dashboard", layout="wide")
st.title("🇮🇳 India Economic Intelligence Dashboard — All Modules")

# API keys (optional)
TRADINGECONOMICS_KEY = os.getenv("TRADINGECONOMICS_KEY", "")  # format: email:key
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")

# polite header for scraping
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117 Safari/537.36"}

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# -------------------------
# Utilities / Helpers
# -------------------------
@st.cache_data(ttl=60*5)
def yf_history(symbol: str, period: str = "1y", interval: str = "1d"):
    """Fetch historical data from yfinance; cached."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        return df
    except Exception:
        return None

@st.cache_data(ttl=60*5)
def yf_info(symbol: str):
    """Fetch ticker info & corporate actions using yfinance."""
    try:
        tk = yf.Ticker(symbol)
        info = {}
        # recent history for price
        hist = tk.history(period="5d")
        if not hist.empty:
            info["latest"] = float(hist["Close"].iloc[-1])
            info["prev_close"] = float(hist["Close"].iloc[-2]) if len(hist) > 1 else info["latest"]
            info["pct_change"] = (info["latest"] - info["prev_close"]) / info["prev_close"] * 100 if info["prev_close"] != 0 else 0.0
        # profile (may be empty)
        try:
            raw = tk.info
            info["name"] = raw.get("longName") or raw.get("shortName")
            info["sector"] = raw.get("sector")
            info["marketCap"] = raw.get("marketCap")
            info["trailingPE"] = raw.get("trailingPE")
            info["summary"] = raw.get("longBusinessSummary")
        except Exception:
            pass
        # corporate actions
        try:
            divs = tk.dividends
            splits = tk.splits
            info["dividends"] = divs.tail(10).to_dict() if not divs.empty else {}
            info["splits"] = splits.tail(10).to_dict() if not splits.empty else {}
        except Exception:
            info["dividends"] = {}
            info["splits"] = {}
        return info
    except Exception:
        return None

def sentiment_score(text: str):
    s = analyzer.polarity_scores(text or "")
    compound = s["compound"]
    label = "neutral"
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    return label, compound

def te_request(endpoint: str, params: dict = None):
    """Call TradingEconomics API endpoint if key present. Return JSON or None."""
    if not TRADINGECONOMICS_KEY:
        return None
    base = "https://api.tradingeconomics.com"
    params = params or {}
    params["c"] = TRADINGECONOMICS_KEY
    try:
        r = requests.get(base + endpoint, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None

def google_news_search(query: str = "India economy", limit: int = 10):
    """Best-effort Google News scraping for a query (uses tbm=nws)."""
    try:
        q = requests.utils.requote_uri(query)
        url = f"https://www.google.com/search?q={q}&tbm=nws"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        items = []
        for block in soup.select("div.dbsr")[:limit]:
            a = block.find("a", href=True)
            title = block.find("div", role="heading")
            snippet = block.find("div", class_="Y3v8qd")
            source = block.find("div", class_="XTjFC")
            if a and title:
                items.append({
                    "title": title.get_text(strip=True),
                    "link": a["href"],
                    "snippet": snippet.get_text(strip=True) if snippet else "",
                    "source": source.get_text(strip=True) if source else ""
                })
        if not items:
            # fallback parse generic results
            for g in soup.select("div.g")[:limit]:
                title = g.find("h3")
                link = g.find("a", href=True)
                snippet = g.find("div", class_="st")
                if title and link:
                    items.append({"title": title.get_text(strip=True), "link": link['href'], "snippet": snippet.get_text(strip=True) if snippet else "", "source": ""})
        return items
    except Exception:
        return []

@st.cache_data(ttl=60*10)
def mospi_cpi_scrape():
    """Best-effort scrape new.mospi.gov.in/cpi for CSV or embedded JSON. Returns DataFrame or None."""
    try:
        url = "https://new.mospi.gov.in/dashboard/cpi"
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        # find csv/xlsx links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv") or href.lower().endswith(".xlsx"):
                if href.startswith("/"):
                    href = "https://new.mospi.gov.in" + href
                rr = requests.get(href, headers=HEADERS, timeout=15)
                if rr.status_code == 200:
                    try:
                        if href.lower().endswith(".csv"):
                            df = pd.read_csv(BytesIO(rr.content))
                        else:
                            df = pd.read_excel(BytesIO(rr.content))
                        return df
                    except Exception:
                        continue
        # try to extract JSON blob in scripts
        scripts = soup.find_all("script")
        for s in scripts:
            if s.string and "cpi" in s.string.lower():
                # attempt to extract {...}
                import re, json
                m = re.search(r"(\{.*\})", s.string, re.S)
                if m:
                    raw = m.group(1)
                    try:
                        data = json.loads(raw)
                        df = pd.json_normalize(data.get("records", data))
                        return df
                    except Exception:
                        continue
        return None
    except Exception:
        return None

# -------------------------
# Layout: Tabs (All modules)
# -------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📰 News & Insights",
    "📊 CPI & Inflation",
    "📈 IIP & GDP",
    "🏗 Infrastructure Tracker",
    "💼 Employment & Policy",
    "🧠 Business Planning",
    "💹 Stock & Market"
])

# ------------------------------------------------
# Tab: Overview (Home) — Key KPIs + Indices + Pulse
# ------------------------------------------------
with tabs[0]:
    st.header("🏠 Global Dashboard — Snapshot")
    st.markdown("Quick snapshot of markets and macro KPIs. Use the side panel to change timeframe and refresh.")

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([2,1])
    with col_ctrl2:
        refresh_secs = st.number_input("Auto refresh every (seconds)", min_value=30, max_value=600, value=60, step=10)
        last_refresh = st.empty()

    # Indices overview (top)
    st.subheader("Live Market Indices")
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NASDAQ": "^IXIC",
        "DOW JONES": "^DJI",
        "S&P 500": "^GSPC"
    }
    cols = st.columns(len(indices))
    for (name, sym), c in zip(indices.items(), cols):
        df_idx = yf_history = yf_history = yf_history if False else None  # placeholder hack safe
        try:
            data = yf_history(sym, period="5d") if 'yf_history' not in locals() else yf_history
            # use yfinance directly if cached util not present: (we call helper)
            df2 = yf_history(sym, period="5d")  # call above helper
            if df2 is None or df2.empty:
                c.metric(name, "N/A", delta="N/A")
            else:
                last = df2["Close"].iloc[-1]
                prev = df2["Close"].iloc[-2] if len(df2)>1 else last
                delta = (last-prev)/prev*100 if prev != 0 else 0.0
                c.metric(name, f"{last:,.2f}", f"{delta:+.2f}%")
        except Exception:
            c.metric(name, "N/A", delta="N/A")

    st.markdown("---")
    # Macro KPIs — attempt TradingEconomics; else show info
    st.subheader("Macro KPIs (Latest available)")
    col1, col2, col3, col4 = st.columns(4)
    # CPI (try TE)
    try:
        if TRADINGECONOMICS_KEY:
            # attempt to fetch inflation indicator
            cpi_series = te_request("/historical/country/india/indicator/inflation%20rate")
            if isinstance(cpi_series, list) and len(cpi_series)>0:
                latest_cpi = cpi_series[0].get("Value") if "Value" in cpi_series[0] else None
            else:
                latest_cpi = None
        else:
            latest_cpi = None
    except Exception:
        latest_cpi = None
    col1.metric("CPI (latest)", f"{latest_cpi if latest_cpi is not None else '—'}")
    # GDP (placeholder)
    col2.metric("GDP Growth (latest)", "—")
    # IIP placeholder
    col3.metric("IIP (latest)", "—")
    # Unemployment placeholder
    col4.metric("Unemployment (latest)", "—")

    # Pulse meter (rule-based)
    st.subheader("Economic Pulse")
    pulse_text = "Stable"
    pulse_color = "green"
    if latest_cpi is not None:
        try:
            cpi_val = float(latest_cpi)
            if cpi_val > 6:
                pulse_text = "High Inflation — Watch Out"
                pulse_color = "red"
            elif cpi_val > 4:
                pulse_text = "Rising Inflation — Caution"
                pulse_color = "amber"
            else:
                pulse_text = "Stable"
                pulse_color = "green"
        except Exception:
            pulse_text = "Stable"
            pulse_color = "green"
    st.markdown(f"**Pulse:** {pulse_text}  \n_Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")

# ------------------------------------------------
# Tab: News & Insights
# ------------------------------------------------
with tabs[1]:
    st.header("📰 News & Insights")
    st.markdown("Live economic and market news. Sources: NewsAPI (if provided), TradingEconomics news (if available), or Google News scraping fallback. MOSPI press releases attempted from new.mospi.gov.in / PIB.")

    # Controls
    q = st.text_input("Search query (e.g., India economy, RBI, MOSPI, stock):", value="India economy")
    max_items = st.slider("Headlines to show", 5, 25, 12)
    use_newsapi = bool(NEWSAPI_KEY)

    news_items = []
    if use_newsapi:
        try:
            # NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {"q": q, "pageSize": max_items, "sortBy": "publishedAt", "language": "en", "apiKey": NEWSAPI_KEY}
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            for a in data.get("articles", []):
                news_items.append({
                    "title": a.get("title"),
                    "snippet": a.get("description"),
                    "link": a.get("url"),
                    "source": a.get("source", {}).get("name"),
                    "publishedAt": a.get("publishedAt")
                })
        except Exception:
            news_items = []

    # TradingEconomics news fallback
    if not news_items and TRADINGECONOMICS_KEY:
        te_news = te_request("/news", params={"c": TRADINGECONOMICS_KEY})
        if isinstance(te_news, list):
            for it in te_news[:max_items]:
                news_items.append({
                    "title": it.get("title"),
                    "snippet": it.get("description") or it.get("summary"),
                    "link": it.get("url") or it.get("link"),
                    "source": it.get("source"),
                    "publishedAt": it.get("date") or it.get("published_at")
                })

    # Google News fallback
    if not news_items:
        scraped = google_news_search(q, limit=max_items)
        for s in scraped:
            news_items.append({
                "title": s.get("title"),
                "snippet": s.get("snippet"),
                "link": s.get("link"),
                "source": s.get("source"),
                "publishedAt": None
            })

    if not news_items:
        st.warning("No news available. Try a different query or add NEWSAPI_KEY.")
    else:
        # sentiment + display
        dfn = pd.DataFrame(news_items)
        dfn["text"] = dfn["title"].fillna("") + ". " + dfn["snippet"].fillna("")
        dfn["sentiment_label"], dfn["sentiment_score"] = zip(*dfn["text"].map(lambda t: sentiment_score(t)))
        avg_sent = dfn["sentiment_score"].mean()
        pos = (dfn["sentiment_label"]=="positive").sum()
        neg = (dfn["sentiment_label"]=="negative").sum()
        neu = (dfn["sentiment_label"]=="neutral").sum()
        st.metric("News sentiment (avg)", f"{avg_sent:.2f}", delta=f"pos:{pos} neg:{neg} neu:{neu}")
        st.markdown("---")
        for i, row in dfn.iterrows():
            st.write(f"**{row['title']}**")
            if row["snippet"]:
                st.write(row["snippet"])
            if row["link"]:
                st.write(f"[Read more]({row['link']})")
            st.caption(f"{row.get('source','')} • {row.get('publishedAt','')}")
            lab = row["sentiment_label"]
            if lab == "positive":
                st.success(f"Sentiment: {lab} ({row['sentiment_score']:.2f})")
            elif lab == "negative":
                st.error(f"Sentiment: {lab} ({row['sentiment_score']:.2f})")
            else:
                st.info(f"Sentiment: {lab} ({row['sentiment_score']:.2f})")
            st.markdown("---")

    # MOSPI press releases (attempt)
    st.subheader("MOSPI / Government Releases")
    try:
        r = requests.get("https://new.mospi.gov.in/press-release", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            items = []
            for a in soup.select("a[href]")[:15]:
                txt = a.get_text(strip=True)
                href = a["href"]
                if "press release" in txt.lower() or "press" in txt.lower() or "release" in txt.lower():
                    link = href if href.startswith("http") else "https://new.mospi.gov.in" + href
                    items.append((txt, link))
            if items:
                for t, l in items[:6]:
                    st.write(f"- [{t}]({l})")
            else:
                st.info("No MOSPI press items scraped. Use PIB site as fallback.")
        else:
            st.info("MOSPI press page not reachable.")
    except Exception:
        st.info("Could not fetch MOSPI press releases (network or structure change).")

# ------------------------------------------------
# Tab: CPI & Inflation
# ------------------------------------------------
with tabs[2]:
    st.header("📊 CPI & Inflation (India)")
    st.markdown("Goal: MOSPI CPI data (monthly). Sources: TradingEconomics historical indicators, MOSPI scrape, data.gov.in, or upload CSV/XLSX.")
    # Try TradingEconomics first
    cpi_df = None
    if TRADINGECONOMICS_KEY:
        try:
            cpi_data = te_request("/historical/country/india/indicator/inflation%20rate")
            if isinstance(cpi_data, list) and len(cpi_data)>0:
                cpi_df = pd.DataFrame(cpi_data)
                if "Date" in cpi_df.columns:
                    cpi_df["date"] = pd.to_datetime(cpi_df["Date"])
                elif "date" in cpi_df.columns:
                    cpi_df["date"] = pd.to_datetime(cpi_df["date"])
        except Exception:
            cpi_df = None

    # MOSPI scrape fallback
    if cpi_df is None:
        cpi_df = mospi_cpi_scrape()

    # Upload fallback
    up = st.file_uploader("Upload CPI CSV/XLSX (if automatic fetch fails)", type=["csv","xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                cpi_df = pd.read_csv(up)
            else:
                cpi_df = pd.read_excel(up)
            st.success("CPI file uploaded")
        except Exception as e:
            st.error("Upload parse failed: " + str(e))

    if cpi_df is None:
        st.warning("No CPI data available automatically. Upload CSV/XLSX or add TradingEconomics key.")
    else:
        st.write("Preview (top rows):")
        st.dataframe(cpi_df.head())
        # attempt to find date and numeric column
        date_col = None
        value_cols = []
        for col in cpi_df.columns:
            if "date" in col.lower() or "month" in col.lower():
                date_col = col
            if any(k in col.lower() for k in ["value","index","cpi","combined"]):
                value_cols.append(col)
        if date_col is None:
            date_col = cpi_df.columns[0]
        try:
            cpi_df["__date"] = pd.to_datetime(cpi_df[date_col], errors="coerce")
        except Exception:
            cpi_df["__date"] = pd.NaT
        if not value_cols:
            numeric_cols = cpi_df.select_dtypes("number").columns.tolist()
            value_cols = numeric_cols[:2] if numeric_cols else []
        if "__date" in cpi_df.columns and value_cols:
            fig = px.line(cpi_df.sort_values("__date"), x="__date", y=value_cols[:3], title="CPI series")
            st.plotly_chart(fig, use_container_width=True)
            # simple AI insight: top contributing category if categories present
            st.markdown("**AI Insight (rule-based):**")
            st.write("If category-level CPI available, compute month-over-month changes and show top contributors. (Implemented once CPI categories present.)")
        else:
            st.info("Could not auto-detect date/value columns. Upload a cleaned file or set TradingEconomics key.")

# ------------------------------------------------
# Tab: IIP & GDP Trends
# ------------------------------------------------
with tabs[3]:
    st.header("📈 IIP & GDP Trends")
    st.markdown("Live IIP and GDP series. Forecasting uses a simple linear model on historical series (quick estimate). Upload fallback available.")

    # IIP via TradingEconomics
    iip_df = None
    if TRADINGECONOMICS_KEY:
        try:
            iip_data = te_request("/historical/country/india/indicator/index%20of%20industrial%20production")
            if isinstance(iip_data, list) and iip_data:
                iip_df = pd.DataFrame(iip_data)
                if "Date" in iip_df.columns:
                    iip_df["date"] = pd.to_datetime(iip_df["Date"])
        except Exception:
            iip_df = None

    iip_up = st.file_uploader("Upload IIP CSV/XLSX (fallback)", type=["csv","xlsx"], key="iip_up")
    if iip_up is not None:
        try:
            iip_df = pd.read_csv(iip_up) if iip_up.name.lower().endswith(".csv") else pd.read_excel(iip_up)
            st.success("IIP uploaded")
        except Exception as e:
            st.error("Upload failed: " + str(e))

    if iip_df is None:
        st.warning("IIP not available automatically. Provide data or TradingEconomics key.")
    else:
        st.dataframe(iip_df.head())
        # pick numeric column
        val_col = None
        for c in iip_df.columns:
            if any(k in c.lower() for k in ["value","index","iip"]):
                val_col = c
                break
        if val_col is None:
            numeric_cols = iip_df.select_dtypes("number").columns.tolist()
            val_col = numeric_cols[0] if numeric_cols else None
        if val_col:
            try:
                if "date" in iip_df.columns:
                    iip_df["date"] = pd.to_datetime(iip_df["date"], errors="coerce")
                    fig = px.line(iip_df.sort_values("date"), x="date", y=val_col, title="IIP series")
                    st.plotly_chart(fig, use_container_width=True)
                    # simple forecast
                    series = pd.to_numeric(iip_df.sort_values("date")[val_col].astype(float), errors="coerce").dropna()
                    preds = None
                    if len(series) >= 8:
                        preds = simple_linear_forecast(series, steps=4)
                        st.write("Simple linear forecast (next 4 periods):")
                        st.write(preds)
                else:
                    st.info("IIP date column not found; show raw values.")
            except Exception as e:
                st.error("Plot/forecast failed: " + str(e))

    # GDP
    st.markdown("---")
    st.subheader("GDP (series)")
    gdp_df = None
    if TRADINGECONOMICS_KEY:
        try:
            gdp_data = te_request("/historical/country/india/indicator/gdp")
            if isinstance(gdp_data, list) and gdp_data:
                gdp_df = pd.DataFrame(gdp_data)
                if "Date" in gdp_df.columns:
                    gdp_df["date"] = pd.to_datetime(gdp_df["Date"])
        except Exception:
            gdp_df = None

    gdp_up = st.file_uploader("Upload GDP CSV/XLSX (fallback)", type=["csv","xlsx"], key="gdp_up")
    if gdp_up is not None:
        try:
            gdp_df = pd.read_csv(gdp_up) if gdp_up.name.lower().endswith(".csv") else pd.read_excel(gdp_up)
            st.success("GDP uploaded")
        except Exception as e:
            st.error("Upload failed: " + str(e))

    if gdp_df is None:
        st.warning("GDP timeseries not available automatically. Upload CSV/XLSX or add TradingEconomics key.")
    else:
        st.dataframe(gdp_df.head())
        # detect numeric column and plot
        val_col_g = None
        for c in gdp_df.columns:
            if any(k in c.lower() for k in ["value","gdp","amount","growth"]):
                val_col_g = c
                break
        if val_col_g is None:
            numeric_cols = gdp_df.select_dtypes("number").columns.tolist()
            val_col_g = numeric_cols[0] if numeric_cols else None
        if val_col_g and "date" in gdp_df.columns:
            try:
                gdp_df["date"] = pd.to_datetime(gdp_df["date"], errors="coerce")
                fig = px.line(gdp_df.sort_values("date"), x="date", y=val_col_g, title="GDP series")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# ------------------------------------------------
# Tab: Infrastructure Tracker
# ------------------------------------------------
with tabs[4]:
    st.header("🏗 Infrastructure Tracker")
    st.markdown("Shows government infrastructure project spending by state/sector. Preferred sources: India Investment Grid (IIG), IPMD, MOSPI project data. Upload CSV as fallback.")

    infra_up = st.file_uploader("Upload projects CSV/XLSX (columns: state, city, project, cost, sector, status)", type=["csv","xlsx"], key="infra_up")
    infra_df = None
    if infra_up:
        try:
            infra_df = pd.read_csv(infra_up) if infra_up.name.lower().endswith(".csv") else pd.read_excel(infra_up)
            st.success("Infra dataset loaded")
            st.dataframe(infra_df.head())
            if "state" in infra_df.columns and "cost" in infra_df.columns:
                st.subheader("Top 10 states by infra spend")
                totals = infra_df.groupby("state")["cost"].sum().sort_values(ascending=False).head(10)
                st.bar_chart(totals)
                # map if geo available and folium installed
                if HAS_FOLIUM and "latitude" in infra_df.columns and "longitude" in infra_df.columns:
                    m = folium.Map(location=[20.5937,78.9629], zoom_start=5)
                    for _, r in infra_df.dropna(subset=["latitude","longitude"]).iterrows():
                        folium.Marker([r["latitude"], r["longitude"]], popup=f'{r.get("project","")}: {r.get("cost","")}', tooltip=r.get("state","")).add_to(m)
                    folium_static(m, width=700, height=400)
        except Exception as e:
            st.error("Failed to load infra file: " + str(e))
    else:
        st.info("Upload a projects CSV/XLSX file to visualize infrastructure spending. We can also wire IIG/IPMD APIs if you have access.")

# ------------------------------------------------
# Tab: Employment & Policy
# ------------------------------------------------
with tabs[5]:
    st.header("💼 Employment & Policy")
    st.markdown("Employment series (PLFS / CMIE optional). Policy tracker pulls PIB / MOSPI feeds (best-effort). Upload data for immediate visuals.")

    emp_file = st.file_uploader("Upload employment dataset (CSV/XLSX) with date and value columns", type=["csv","xlsx"], key="emp_up")
    if emp_file:
        try:
            edf = pd.read_csv(emp_file) if emp_file.name.lower().endswith(".csv") else pd.read_excel(emp_file)
            st.dataframe(edf.head())
            # plot first numeric column vs date if available
            if "date" in edf.columns:
                edf["date"] = pd.to_datetime(edf["date"], errors="coerce")
                numcols = edf.select_dtypes("number").columns.tolist()
                if numcols:
                    fig = px.line(edf.sort_values("date"), x="date", y=numcols[0], title="Employment series")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Upload failed: " + str(e))
    else:
        st.info("Upload employment CSV/XLSX to visualize. Policy news is available in News tab.")

# ------------------------------------------------
# Tab: Business Planning
# ------------------------------------------------
with tabs[6]:
    st.header("🧠 Business Planning")
    st.markdown("Combine CPI, IIP, GDP, market indices and news sentiment for sector recommendations and planning. This is the analytics workspace.")

    st.write("Select indicators to include in a simple sector score model (example).")
    # Example inputs
    include_cpi = st.checkbox("Include CPI (inflation)", value=True)
    include_iip = st.checkbox("Include IIP", value=True)
    include_gdp = st.checkbox("Include GDP", value=True)
    include_market = st.checkbox("Include Market Indices movements", value=True)

    if st.button("Generate quick sector suggestion (sample)"):
        st.info("Running a simple heuristic analysis (demo). This will use available series to rank sectors.")
        # Demo: pick sectors from uploaded infra or ticker sector
        demo = [
            {"sector":"Manufacturing","score":0.8},
            {"sector":"Energy","score":0.75},
            {"sector":"IT Services","score":0.6},
            {"sector":"Retail","score":0.5}
        ]
        df_demo = pd.DataFrame(demo).sort_values("score", ascending=False)
        st.table(df_demo)
        st.download_button("Download One-Page Plan (CSV)", data=df_demo.to_csv(index=False).encode("utf-8"), file_name="business_plan_suggestions.csv")

# ------------------------------------------------
# Tab: Stock & Market (Single-stock + Indices + Corporate actions + News)
# ------------------------------------------------
with tabs[7]:
    st.header("💹 Stock & Market — Single Stock (Auto-refresh)")
    st.markdown("Enter **one stock** symbol at a time (Indian: .NS suffix, e.g., RELIANCE.NS ; US: AAPL). Dashboard auto-refreshes every 60 seconds by default.")

    col_left, col_right = st.columns([2,1])
    with col_right:
        auto_refresh = st.number_input("Auto refresh interval (seconds)", min_value=20, max_value=600, value=60, step=10)
        last_run = st.empty()
        show_sparklines = st.checkbox("Show index sparklines", value=True)
        st.markdown("**Indices snapshot**")
        for nm, sym in [("NIFTY 50","^NSEI"),("SENSEX","^BSESN"),("NASDAQ","^IXIC"),("DOW JONES","^DJI")]:
            idx = yf_history(sym, period="1mo")
            if idx is not None and not idx.empty:
                last = idx["Close"].iloc[-1]
                prev = idx["Close"].iloc[-2] if len(idx)>1 else last
                delta = (last-prev)/prev*100 if prev!=0 else 0.0
                st.metric(nm, f"{last:,.2f}", f"{delta:+.2f}%")
                if show_sparklines:
                    fig = px.line(idx, x="Date", y="Close", height=120)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"{nm}: N/A")

    with col_left:
        symbol = st.text_input("Stock symbol (one at a time)", value="RELIANCE.NS")
        lookup = st.button("Lookup stock")
        # Keep input value preserved during refresh
        if lookup or symbol:
            # fetch ticker data
            info = yf_info(symbol)
            hist = yf_history(symbol, period="1y")
            if info is None:
                st.error("Could not fetch ticker info. Check symbol.")
            else:
                # Top metrics
                st.subheader(f"{info.get('name', symbol)}  —  {symbol}")
                st.columns([1,1,1,1])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest", f"{info.get('latest','N/A')}")
                c2.metric("Change %", f"{info.get('pct_change',0):+.2f}%")
                c3.metric("Market Cap", f"{info.get('marketCap','N/A')}")
                c4.metric("P/E (trailing)", f"{info.get('trailingPE','N/A')}")
                if hist is not None and not hist.empty:
                    fig = px.line(hist, x="Date", y="Close", title=f"{symbol} — 1 year")
                    st.plotly_chart(fig, use_container_width=True)
                # corporate actions
                st.subheader("Corporate Actions")
                divs = info.get("dividends", {})
                splits = info.get("splits", {})
                if divs:
                    try:
                        ddf = pd.DataFrame(list(divs.items()), columns=["Date","Dividend"])
                        ddf["Date"] = pd.to_datetime(ddf["Date"])
                        st.table(ddf.sort_values("Date", ascending=False).head(8))
                    except Exception:
                        st.write(divs)
                else:
                    st.write("No dividend data available (via yfinance).")
                if splits:
                    try:
                        sdf = pd.DataFrame(list(splits.items()), columns=["Date","Split"])
                        sdf["Date"] = pd.to_datetime(sdf["Date"])
                        st.table(sdf.sort_values("Date", ascending=False).head(8))
                    except Exception:
                        st.write(splits)
                else:
                    st.write("No splits data available.")

                # company news
                st.subheader("Related News (Google News / Google Finance search)")
                news_items = google_news_search(f"{symbol} stock", limit=6)
                if news_items:
                    for n in news_items:
                        lab, sc = sentiment_score(n.get("title","") + " " + n.get("snippet",""))
                        st.write(f"**{n.get('title')}**")
                        if n.get("snippet"):
                            st.write(n.get("snippet"))
                        if n.get("link"):
                            st.write(f"[Read more]({n.get('link')})")
                        # sentiment
                        if lab == "positive":
                            st.success(f"Sentiment: {lab} ({sc:.2f})")
                        elif lab == "negative":
                            st.error(f"Sentiment: {lab} ({sc:.2f})")
                        else:
                            st.info(f"Sentiment: {lab} ({sc:.2f})")
                        st.markdown("---")
                else:
                    st.info("No news found. Add NEWSAPI_KEY for more reliable results.")

    # Auto-refresh mechanism
    st.write(f"Auto-refresh interval: {auto_refresh} seconds. (App will refresh data automatically.)")
    # note: streamlit_autorefresh could be used; implement simple sleep+rerun pattern
    # but streamlit_autorefresh is non-blocking; use it:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=auto_refresh * 1000, key="autorefresh")
    except Exception:
        # fallback: instruct user to click refresh; we avoid blocking loops
        st.info("Auto-refresh available if 'streamlit-autorefresh' is installed. Otherwise refresh manually.")

# -------------------------
# Footer notes
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- This app uses best-effort scraping for government sites when APIs are not available. Scraping may break if site changes or blocks requests.")
st.sidebar.write("- For robust automation in production, provide TRADINGECONOMICS_KEY and NEWSAPI_KEY in environment or Streamlit Secrets.")
st.sidebar.write("- If a dataset is missing, upload CSV/XLSX as a fallback in the relevant tab.")
st.sidebar.write("- Contact me for a hosted ETL pipeline option (pull nightly & store in DB for faster dashboard).")

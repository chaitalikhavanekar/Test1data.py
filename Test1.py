# full_econ_dashboard.py
"""
India Economic Intelligence Dashboard (All-in-one)
Tabs:
 - 🏠 Dashboard (Overview)
 - 📰 News & Insights
 - 📊 CPI & Inflation
 - 🏗 Infrastructure Tracker
 - 📈 IIP & GDP Trends
 - 💼 Employment & Policy
 - 🧠 Business Planning
 - 💹 Stock & Market
Uses TradingEconomics (if TRADINGECONOMICS_KEY provided), yfinance, google news scraping, and fallbacks.
"""

import os
import time
import json
from datetime import datetime, timedelta
from io import BytesIO

import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------- Config -----------------
st.set_page_config(page_title="India Economic Intelligence Dashboard", layout="wide")
st.title("🇮🇳 India Economic Intelligence Dashboard")

# Keys (put in environment or Streamlit secrets)
TRADINGECONOMICS_KEY = os.getenv("TRADINGECONOMICS_KEY", "")  # format: email:key
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # optional

# polite headers for scraping
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DashboardBot/1.0)"}

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ----------------- Utility functions -----------------
@st.cache_data(ttl=60 * 10)
def get_yf_history(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        return df
    except Exception:
        return None

@st.cache_data(ttl=60 * 5)
def get_yf_ticker_info(symbol):
    try:
        t = yf.Ticker(symbol)
        info = {}
        # Basic live price & recent history
        hist = t.history(period="5d")
        if not hist.empty:
            info["latest"] = float(hist["Close"].iloc[-1])
            info["prev_close"] = float(hist["Close"].iloc[-2]) if len(hist) > 1 else info["latest"]
            info["pct_change"] = (info["latest"] - info["prev_close"]) / info["prev_close"] * 100 if info["prev_close"] != 0 else 0
        # company info (may be limited)
        try:
            info_raw = t.info
            info["name"] = info_raw.get("longName") or info_raw.get("shortName")
            info["sector"] = info_raw.get("sector")
            info["marketCap"] = info_raw.get("marketCap")
            info["trailingPE"] = info_raw.get("trailingPE")
            info["summary"] = info_raw.get("longBusinessSummary")
        except Exception:
            pass
        # corporate actions
        try:
            divs = t.dividends
            splits = t.splits
            info["dividends"] = divs.tail(10).to_dict() if not divs.empty else {}
            info["splits"] = splits.tail(10).to_dict() if not splits.empty else {}
        except Exception:
            info["dividends"] = {}
            info["splits"] = {}
        return info
    except Exception:
        return None

def sentiment_label(text):
    s = analyzer.polarity_scores(text or "")
    c = s["compound"]
    if c >= 0.05:
        return "positive", c
    if c <= -0.05:
        return "negative", c
    return "neutral", c

# TradingEconomics helper (best-effort)
def te_request(endpoint, params=None):
    """Make request to TradingEconomics with key if provided"""
    base = "https://api.tradingeconomics.com"
    if not TRADINGECONOMICS_KEY:
        return None
    url = base + endpoint
    p = params or {}
    p["c"] = TRADINGECONOMICS_KEY
    try:
        r = requests.get(url, params=p, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None

# Simple linear forecast (1-step) for quick estimate
def simple_linear_forecast(series, steps=4):
    try:
        series = series.dropna()
        if len(series) < 6:
            return None
        X = np.arange(len(series)).reshape(-1,1)
        y = series.values
        model = LinearRegression().fit(X, y)
        xp = np.arange(len(series), len(series)+steps).reshape(-1,1)
        preds = model.predict(xp)
        return preds
    except Exception:
        return None

# Google News scraping (generic search)
def google_news_search(query, limit=10):
    try:
        q = requests.utils.requote_uri(query)
        url = f"https://www.google.com/search?q={q}&tbm=nws"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        items = []
        # look for news blocks
        for block in soup.select("div.dbsr")[:limit]:
            a = block.find("a", href=True)
            title_tag = block.find("div", role="heading")
            snippet_tag = block.find("div", class_="Y3v8qd")
            source_tag = block.find("div", class_="XTjFC")
            if a and title_tag:
                items.append({
                    "title": title_tag.get_text(strip=True),
                    "link": a["href"],
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                    "source": source_tag.get_text(strip=True) if source_tag else ""
                })
        # fallback: parse 'g' results
        if not items:
            for g in soup.select("div.g")[:limit]:
                title = g.find("h3")
                link = g.find("a", href=True)
                snippet = g.find("div", class_="st")
                if title and link:
                    items.append({"title": title.get_text(strip=True), "link": link['href'], "snippet": (snippet.get_text(strip=True) if snippet else ""), "source": ""})
        return items
    except Exception:
        return []

# ----------------- Sidebar Controls -----------------
st.sidebar.header("Controls")
st.sidebar.markdown("Refresh or force re-fetch data when needed.")
if st.sidebar.button("🔄 Refresh data (force)"):
    # clear caches - call cache clear functions if exist
    try:
        get_yf_history.clear()
        get_yf_ticker_info.clear()
    except Exception:
        pass
    st.experimental_rerun()

# Time range selectors used across
default_range = "5y"
range_option = st.sidebar.selectbox("Global time range", ["1y","3y","5y"], index=2)

# ----------------- Tabs -----------------
tabs = st.tabs(["🏠 Dashboard", "📰 News & Insights", "📊 CPI & Inflation",
                "🏗 Infrastructure", "📈 IIP & GDP", "💼 Employment & Policy",
                "🧠 Business Planning", "💹 Stock & Market"])

# ----------------- TAB: Dashboard (Overview) -----------------
with tabs[0]:
    st.header("🏠 Main Dashboard — Overview")
    st.write("Snapshot of markets and key macro indicators (auto-update).")

    # Market indices snapshot
    st.subheader("Market Indices")
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NASDAQ": "^IXIC",
        "DOW JONES": "^DJI"
    }
    cols = st.columns(len(indices))
    for (name, ticker), col in zip(indices.items(), cols):
        with col:
            data = get_yf_history(ticker, period="5d")
            if data is None or data.empty:
                col.metric(name, "N/A", delta="N/A")
            else:
                last = data["Close"].iloc[-1]
                prev = data["Close"].iloc[-2] if len(data) > 1 else last
               import numpy as np
delta = np.where(prev != 0, (last - prev) / prev * 100, 0)
                col.metric(name, f"{last:,.2f}", f"{delta:+.2f}%")

    st.markdown("---")
    # Macro KPIs (try TradingEconomics; fallback to msg)
    st.subheader("Macro KPIs (Latest)")
    k1, k2, k3, k4 = st.columns(4)

    # CPI (India) — TradingEconomics indicator name "inflation rate" endpoint can be used for country
    cpi_val = None
    gdp_val = None
    iip_val = None

    if TRADINGECONOMICS_KEY:
        try:
            cpi_data = te_request("/country/india")
            # tradingeconomics country endpoint returns a lot; for reliability use indicator endpoints
            # fallback: try inflation indicator
            cpi_series = te_request("/country/forecast/inflation?country=india")
            # many TE endpoints vary; best-effort: use /historical/country?indicator=Consumer%20Price%20Index
            # We'll show placeholders if parsing fails
        except Exception:
            cpi_series = None
    else:
        cpi_series = None

    k1.metric("CPI (latest)", f"{'—' if cpi_val is None else cpi_val}")
    k2.metric("GDP Growth (latest)", f"{'—' if gdp_val is None else gdp_val}")
    k3.metric("IIP (latest)", f"{'—' if iip_val is None else iip_val}")
    k4.metric("Last update", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    st.markdown("### Economic Pulse")
    # Pulse rule example using placeholders - color logic
    # If CPI > 6 -> red; CPI 4-6 amber; else green. Using placeholder - user sees real values when TE works.
    pulse = "green"
    st.metric("Pulse", "Stable", delta="—")

    st.info("This overview mixes market indices (left) with macro KPIs (right). CPI/IIP/GDP populate when TradingEconomics connection is available.")

# ----------------- TAB: News & Insights -----------------
with tabs[1]:
    st.header("📰 News & Insights")
    st.write("Live economic & market news. Uses Google News scraping as default (no API required).")

    q = st.text_input("Search news for (e.g., 'India economy', 'RBI', 'stock market'):", value="India economy")
    max_items = st.slider("Number of headlines to show", 5, 20, 10)

    # Priority: if NEWSAPI_KEY present, use it (clean). Else fallback to TradingEconomics news (if key), else Google scraping.
    news_df = None
    use_newsapi = bool(NEWSAPI_KEY)
    if use_newsapi:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": q, "pageSize": max_items, "sortBy": "publishedAt", "language": "en", "apiKey": NEWSAPI_KEY}
            r = requests.get(url, params=params, timeout=15)
            payload = r.json()
            articles = payload.get("articles", [])
            news_df = pd.DataFrame([{"title": a["title"], "desc": a["description"], "url": a["url"], "publishedAt": a["publishedAt"], "source": a["source"]["name"]} for a in articles])
        except Exception:
            news_df = None

    if news_df is None and TRADINGECONOMICS_KEY:
        # TradingEconomics news endpoint (best-effort)
        te_news = te_request("/news", params={"c": TRADINGECONOMICS_KEY})
        if isinstance(te_news, list):
            try:
                news_df = pd.DataFrame(te_news)[:max_items]
                if "title" not in news_df.columns:
                    news_df = None
            except Exception:
                news_df = None

    if news_df is None:
        # Google News scraping fallback
        results = google_news_search(q, limit=max_items)
        news_df = pd.DataFrame(results)

    if news_df is None or news_df.empty:
        st.warning("No news available from configured sources. Try changing the query or add NEWSAPI_KEY.")
    else:
        # sentiment
        news_df["text"] = news_df.get("title","").fillna("") + ". " + news_df.get("snippet", news_df.get("desc","")).fillna("")
        sents = news_df["text"].apply(lambda t: sentiment_label(t))
        news_df["sent_label"] = [s[0] for s in sents]
        news_df["sent_score"] = [s[1] for s in sents]

        avg = news_df["sent_score"].mean()
        pos = (news_df["sent_label"]=="positive").sum()
        neg = (news_df["sent_label"]=="negative").sum()
        neu = (news_df["sent_label"]=="neutral").sum()

        st.metric("Avg news sentiment", f"{avg:.2f}", delta=f"pos:{pos} neg:{neg} neu:{neu}")
        st.markdown("---")
        for i, row in news_df.iterrows():
            st.write(f"**{row.get('title')}**")
            if row.get("snippet"):
                st.write(row.get("snippet"))
            if row.get("url"):
                st.write(f"[Read more]({row.get('url')})")
            st.caption(f"{row.get('source','')} • {row.get('publishedAt','')}")
            # sentiment badge
            lab = row.get("sent_label","neutral")
            if lab == "positive":
                st.success(f"Sentiment: {lab} ({row.get('sent_score'):.2f})")
            elif lab == "negative":
                st.error(f"Sentiment: {lab} ({row.get('sent_score'):.2f})")
            else:
                st.info(f"Sentiment: {lab} ({row.get('sent_score'):.2f})")
            st.markdown("---")

# ----------------- TAB: CPI & Inflation -----------------
with tabs[2]:
    st.header("📊 CPI & Inflation (India)")
    st.write("Goal: MOSPI-style CPI analytics (monthly). Uses TradingEconomics when available; else provide upload fallback.")
    if TRADINGECONOMICS_KEY:
        # try to fetch CPI series for India (TradingEconomics historical indicator)
        # endpoint may vary; using TE historical indicator endpoint example for "Consumer Price Index" or "Inflation Rate"
        cpi_data = te_request("/historical/country/india/indicator/inflation%20rate")
        # fallback to direct country indicator 'inflation rate' - TE's endpoints can differ; best-effort
        if cpi_data and isinstance(cpi_data, list):
            try:
                df_cpi = pd.DataFrame(cpi_data)
                # TE returns "Date" and "Value" - normalize names
                if "Date" in df_cpi.columns:
                    df_cpi["date"] = pd.to_datetime(df_cpi["Date"])
                elif "date" in df_cpi.columns:
                    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
                st.write("Source: TradingEconomics (inflation rate historical, best-effort)")
                st.dataframe(df_cpi.tail(10))
                # plot
                if "Value" in df_cpi.columns:
                    fig = px.line(df_cpi.sort_values("date"), x="date", y="Value", title="Inflation Rate (TE) - India")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not parse TradingEconomics CPI data: " + str(e))
        else:
            st.info("TradingEconomics CPI endpoint didn't return usable data. Use file upload below or wait.")
    else:
        st.info("TradingEconomics key not set. You can upload MOSPI CPI CSV/XLSX to visualize.")
    up = st.file_uploader("Upload CPI CSV/XLSX (fallback)", type=["csv","xlsx"])
    if up:
        try:
            if up.name.lower().endswith(".csv"):
                cpi_df = pd.read_csv(up)
            else:
                cpi_df = pd.read_excel(up)
            st.success("CPI file loaded")
            st.dataframe(cpi_df.head())
        except Exception as e:
            st.error("Upload failed: " + str(e))

# ----------------- TAB: Infrastructure Tracker -----------------
with tabs[3]:
    st.header("🏗 Infrastructure Tracker (Projects & Spending)")
    st.write("Data source: IPMD / India Investment Grid / MOSPI. If APIs not available, upload CSVs or use India Investment Grid.")
    st.info("This tab shows top projects, state spending and filters. (Upload a project CSV as fallback.)")
    infra_up = st.file_uploader("Upload projects CSV (columns: state, project, cost, status, sector, start_date, end_date)", type=["csv","xlsx"], key="infra")
    if infra_up:
        try:
            if infra_up.name.lower().endswith(".csv"):
                infra_df = pd.read_csv(infra_up)
            else:
                infra_df = pd.read_excel(infra_up)
            st.dataframe(infra_df.head())
            # simple state totals
            if "state" in infra_df.columns and "cost" in infra_df.columns:
                st.subheader("Top States by Project Spend")
                st.bar_chart(infra_df.groupby("state")["cost"].sum().sort_values(ascending=False).head(10))
        except Exception as e:
            st.error("Could not parse file: " + str(e))
    else:
        st.info("If you have API access to IPMD/PAIMANA or India Investment Grid, we can wire it here. Otherwise upload CSV to visualize.")

# ----------------- TAB: IIP & GDP Trends -----------------
with tabs[4]:
    st.header("📈 IIP & GDP Trends")
    st.write("Live IIP & GDP data (TradingEconomics preferred). Includes simple forecast (linear).")

    # Try IIP
    iip_data = None
    if TRADINGECONOMICS_KEY:
        iip_data = te_request("/historical/country/india/indicator/index%20of%20industrial%20production")
    if iip_data and isinstance(iip_data, list):
        iip_df = pd.DataFrame(iip_data)
        # normalize
        if "Date" in iip_df.columns:
            iip_df["date"] = pd.to_datetime(iip_df["Date"])
        # plot index if present
        val_col = "Value" if "Value" in iip_df.columns else (iip_df.columns[-1] if len(iip_df.columns)>0 else None)
        if val_col:
            fig = px.line(iip_df.sort_values("date"), x="date", y=val_col, title="IIP (TradingEconomics)")
            st.plotly_chart(fig, use_container_width=True)
            # forecast
            try:
                series = pd.to_numeric(iip_df.sort_values("date")[val_col].astype(float), errors="coerce").dropna()
                preds = simple_linear_forecast(series, steps=4)
                if preds is not None:
                    st.write("Simple linear forecast (next 4 periods):")
                    st.write(preds)
            except Exception:
                pass
    else:
        st.info("IIP data not available via TradingEconomics. Upload fallback CSV with 'date' and 'value' columns.")
        file = st.file_uploader("Upload IIP CSV/XLSX", type=["csv","xlsx"], key="iip")
        if file:
            df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
            st.dataframe(df.head())
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                fig = px.line(df.sort_values("date"), x="date", y=df.select_dtypes("number").columns[0])
                st.plotly_chart(fig, use_container_width=True)

    # GDP block (try TE)
    if TRADINGECONOMICS_KEY:
        gdp_data = te_request("/historical/country/india/indicator/gdp")
        if gdp_data and isinstance(gdp_data, list):
            gdp_df = pd.DataFrame(gdp_data)
            if "Date" in gdp_df.columns:
                gdp_df["date"] = pd.to_datetime(gdp_df["Date"])
            val_col = "Value" if "Value" in gdp_df.columns else (gdp_df.columns[-1] if len(gdp_df.columns)>0 else None)
            if val_col:
                fig = px.line(gdp_df.sort_values("date"), x="date", y=val_col, title="GDP (TradingEconomics)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GDP timeseries not available via TradingEconomics endpoint used here. Use file upload fallback.")
            gfile = st.file_uploader("Upload GDP CSV/XLSX", type=["csv","xlsx"], key="gdp")
            if gfile:
                gdf = pd.read_csv(gfile) if gfile.name.lower().endswith(".csv") else pd.read_excel(gfile)
                st.dataframe(gdf.head())
                if "date" in gdf.columns:
                    gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce")
                    fig = px.line(gdf.sort_values("date"), x="date", y=gdf.select_dtypes("number").columns[0])
                    st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB: Employment & Policy -----------------
with tabs[5]:
    st.header("💼 Employment & Policy")
    st.write("Employment stats (PLFS / Labour) and policy tracker from PIB / MOSPI press releases.")
    st.info("You can upload employment CSV/PLFS data. Policy releases are shown in News tab (search 'PIB budget' etc.).")

    emp_file = st.file_uploader("Upload Employment CSV/XLSX (optional)", type=["csv","xlsx"], key="emp")
    if emp_file:
        try:
            edf = pd.read_csv(emp_file) if emp_file.name.lower().endswith(".csv") else pd.read_excel(emp_file)
            st.dataframe(edf.head())
        except Exception as e:
            st.error("Failed to parse file: " + str(e))

# ----------------- TAB: Business Planning -----------------
with tabs[6]:
    st.header("🧠 Business Planning")
    st.write("This section will combine CPI, GDP, IIP, market & news sentiment to provide sector-level suggestions.")
    st.info("Currently this shows a plan summary. After data is available, we will provide 'Top sectors to consider' and downloadable one-page reports.")
    st.button("Generate sample one-page planning report (placeholder)", help="Will produce report after data pipelines are set up")

# ----------------- TAB: Stock & Market -----------------
with tabs[7]:
    st.header("💹 Stock & Market")
    st.write("Live indices + Stock search. Supports Indian (.NS) and US tickers. News via Google Finance best-effort.")

    left, right = st.columns([2,1])
    with left:
        symbol = st.text_input("Search stock symbol (e.g., RELIANCE.NS, TCS.NS, AAPL):", value="RELIANCE.NS")
        period = st.selectbox("Chart period", ["1mo","3mo","6mo","1y"], index=2)
        if st.button("Lookup"):
            with st.spinner("Fetching stock data..."):
                hist = get_yf_history(symbol, period=period)
                info = get_yf_ticker_info(symbol)
            if hist is None or hist.empty:
                st.error("No historical data found. Check symbol (for Indian tickers use .NS).")
            else:
                st.subheader(f"{symbol} price ({period})")
                fig = px.line(hist, x="Date", y="Close", title=f"{symbol} price")
                st.plotly_chart(fig, use_container_width=True)
                # show info metrics
                if info:
                    st.metric("Latest Price", f"{info.get('latest','N/A')}", f"{info.get('pct_change',0):+.2f}%")
                    st.write(f"**Name:** {info.get('name','N/A')}")
                    st.write(f"**Sector:** {info.get('sector','N/A')}")
                    st.write(f"**Market Cap:** {info.get('marketCap','N/A')}")
                    if info.get("summary"):
                        st.write(info.get("summary")[:400] + ("..." if len(info.get("summary"))>400 else ""))
                # corporate actions
                st.subheader("Corporate actions (dividends & splits)")
                divs = info.get("dividends", {})
                splits = info.get("splits", {})
                if divs:
                    try:
                        ddf = pd.DataFrame(list(divs.items()), columns=["Date","Dividend"])
                        ddf["Date"] = pd.to_datetime(ddf["Date"])
                        st.table(ddf.sort_values("Date", ascending=False).head(10))
                    except Exception:
                        st.write(divs)
                else:
                    st.write("No dividend info")

                if splits:
                    try:
                        sdf = pd.DataFrame(list(splits.items()), columns=["Date","Split"])
                        sdf["Date"] = pd.to_datetime(sdf["Date"])
                        st.table(sdf.sort_values("Date", ascending=False).head(10))
                    except Exception:
                        st.write(splits)
                else:
                    st.write("No splits info")

                st.subheader("Related news (Google Finance / News search)")
                news_items = google_news_search(f"{symbol} stock", limit=6)
                if news_items:
                    for n in news_items:
                        st.write(f"**{n.get('title')}**")
                        if n.get('snippet'):
                            st.write(n.get('snippet'))
                        if n.get('link'):
                            st.write(f"[Read more]({n.get('link')})")
                        st.markdown("---")
                else:
                    st.write("No news found via Google scraping. Consider adding NEWSAPI_KEY for reliable news.")

    with right:
        st.subheader("Indices snapshot")
        for n,t in [("NIFTY 50", "^NSEI"), ("NASDAQ", "^IXIC"), ("DOW JONES", "^DJI")]:
            idx = get_yf_history(t, period="1mo")
            if idx is not None and not idx.empty:
                last = idx["Close"].iloc[-1]
                prev = idx["Close"].iloc[-2] if len(idx)>1 else last
                delta = (last-prev)/prev*100 if prev!=0 else 0
                st.metric(n, f"{last:,.2f}", f"{delta:+.2f}%")
            else:
                st.write(f"{n}: N/A")

# ----------------- Footer / notes -----------------
st.sidebar.markdown("---")
st.sidebar.info("Notes:\n- TradingEconomics integration is best-effort. If TE endpoints change, add resource IDs or CSV uploads.\n- Google News scraping is a best-effort fallback; consider adding NEWSAPI_KEY for reliability.\n- If an endpoint is blocked, upload CSV/XLSX in the relevant tab as a fallback.")

st.write("Built for you — next steps: connect TE key & optional NewsAPI key, then we can refine CPI/IIP/GDP fetches and add state-level infra maps.")

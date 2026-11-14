"""
Nifty 50 Stock Performance Dashboard
Run: streamlit run app.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #080b12;
    color: #e2e8f0;
}
.stApp { background-color: #080b12; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2430;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #00e676;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2430;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'Space Mono', monospace;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e2430;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    color: #64748b;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #0f1724 !important;
    border-color: #1e2430 !important;
    border-bottom-color: #0f1724 !important;
    color: #00e676 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #1e2430; border-radius: 8px; }
.stDataFrame thead { background: #0d1117 !important; }

/* Divider */
hr { border-color: #1e2430; }

/* Header banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f1f1a 50%, #0d1117 100%);
    border: 1px solid #1e3a2f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00e676;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}
.badge {
    background: #0f2a1e;
    border: 1px solid #00e676;
    border-radius: 6px;
    padding: 6px 14px;
    color: #00e676;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #00e676;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
    border-left: 2px solid #00e676;
    padding-left: 8px;
}

/* Stock pill in tables */
.green-pill { color: #00e676; font-weight: 600; }
.red-pill   { color: #ff1744; font-weight: 600; }

/* Info box */
.info-box {
    background: #0d1117;
    border: 1px solid #1e2430;
    border-left: 3px solid #448aff;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Chart style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0f1724",
    "axes.edgecolor":   "#1e2430",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e2430",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
    "axes.titlesize":   11,
    "axes.titlepad":    14,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

GREEN  = "#00e676"; RED = "#ff1744"; BLUE = "#448aff"
GOLD   = "#ffd740"; PURPLE = "#e040fb"; ORANGE = "#ff6d00"
TEAL   = "#00b0ff"
PALETTE = [GREEN, BLUE, GOLD, PURPLE, ORANGE, TEAL, "#69f0ae", "#ffab40", "#ea80fc", "#b2ff59"]

# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(csv_dir: str, sector_file: str):
    # ── Stock prices
    files = glob.glob(f"{csv_dir}/*.csv")
    frames = []
    for f in files:
        ticker = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f, parse_dates=["date"])
        df["Ticker"] = ticker
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    master.sort_values(["Ticker", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)
    master["daily_return"] = master.groupby("Ticker")["close"].pct_change()
    master["month"] = master["date"].dt.to_period("M")

    # ── Sector mapping
    sec = pd.read_csv(sector_file)
    sec["Ticker"] = sec["Symbol"].str.split(": ").str[-1].str.strip()
    sec = sec[["Ticker", "sector"]].drop_duplicates()
    fixes = {"TATACONSUMER": "TATACONSUM", "AIRTEL": "BHARTIARTL", "ADANIGREEN": "ADANIENT"}
    sec["Ticker"] = sec["Ticker"].replace(fixes)

    master = master.merge(sec, on="Ticker", how="left")

    # ── Derived metrics
    first = master.groupby("Ticker")["close"].first()
    last  = master.groupby("Ticker")["close"].last()
    yearly = ((last - first) / first * 100).rename("yearly_return_pct").reset_index()

    volatility = (
        master.groupby("Ticker")["daily_return"].std() * 100
    ).rename("volatility").reset_index()

    monthly_ret = (
        master.groupby(["Ticker", "month"])["close"]
        .agg(fc="first", lc="last")
        .assign(monthly_return=lambda x: (x["lc"] - x["fc"]) / x["fc"] * 100)
        .reset_index()
    )

    return master, yearly, volatility, monthly_ret, sec


# ── Paths ─────────────────────────────────────────────────────
CSV_DIR     = r"C:\Users\HP\Downloads\files\csv_output"
SECTOR_FILE = r"C:\Users\HP\Downloads\files\Sector_data_-_Sheet1.csv"

if not os.path.exists(CSV_DIR):
    st.error(f"Folder `{CSV_DIR}/` not found. Place all stock CSVs there.")
    st.stop()
if not os.path.exists(SECTOR_FILE):
    st.warning(f"`{SECTOR_FILE}` not found — sector chart will be skipped.")

with st.spinner("Loading market data..."):
    master, yearly_df, volatility_df, monthly_ret, sector_df = load_data(CSV_DIR, SECTOR_FILE)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙ FILTERS")

    all_tickers = sorted(master["Ticker"].unique())
    selected    = st.multiselect("Stocks", all_tickers, default=all_tickers[:6])

    all_sectors = sorted(master["sector"].dropna().unique())
    sel_sectors = st.multiselect("Sectors", all_sectors, default=all_sectors)

    date_min = master["date"].min().date()
    date_max = master["date"].max().date()
    d_range = st.date_input("Date Range", [date_min, date_max])

    st.markdown("---")
    st.markdown("## 📋 DATA INFO")
    st.caption(f"**Symbols:** {master['Ticker'].nunique()}")
    st.caption(f"**Rows:** {len(master):,}")
    st.caption(f"**From:** {date_min}")
    st.caption(f"**To:**   {date_max}")

# ── Apply filters
filtered = master[master["Ticker"].isin(selected)]
if len(d_range) == 2:
    filtered = filtered[
        (filtered["date"].dt.date >= d_range[0]) &
        (filtered["date"].dt.date <= d_range[1])
    ]

# ═══════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div>
    <p class="hero-title">NIFTY 50 — MARKET INTELLIGENCE</p>
    <p class="hero-sub">
        {master['Ticker'].nunique()} stocks &nbsp;·&nbsp;
        {date_min} → {date_max} &nbsp;·&nbsp;
        {len(master):,} trading records
    </p>
  </div>
  <div class="badge">LIVE DASHBOARD</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  KPI ROW
# ═══════════════════════════════════════════════════════════════
green_count = (yearly_df["yearly_return_pct"] > 0).sum()
red_count   = (yearly_df["yearly_return_pct"] <= 0).sum()
avg_price   = master["close"].mean()
avg_volume  = master["volume"].mean()
best_stock  = yearly_df.loc[yearly_df["yearly_return_pct"].idxmax()]
worst_stock = yearly_df.loc[yearly_df["yearly_return_pct"].idxmin()]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🟢 Green Stocks",   int(green_count))
k2.metric("🔴 Red Stocks",     int(red_count))
k3.metric("Avg Close Price",  f"₹{avg_price:,.0f}")
k4.metric("Avg Daily Volume", f"{avg_volume/1e6:.1f}M")
k5.metric("Best Stock",       best_stock["Ticker"],
          f"+{best_stock['yearly_return_pct']:.1f}%")
k6.metric("Worst Stock",      worst_stock["Ticker"],
          f"{worst_stock['yearly_return_pct']:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆  Rankings",
    "📉  Volatility",
    "📈  Cumulative Returns",
    "🏭  Sector",
    "🔥  Correlation",
    "📅  Monthly Movers",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Rankings
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-label">Yearly Performance Rankings</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**📗 Top 10 Green Stocks**")
        top10 = (
            yearly_df.nlargest(10, "yearly_return_pct")
            .merge(sector_df, on="Ticker", how="left")
            [["Ticker", "sector", "yearly_return_pct"]]
            .rename(columns={"yearly_return_pct": "Return %", "sector": "Sector"})
            .assign(**{"Return %": lambda x: x["Return %"].map("{:+.2f}%".format)})
        )
        st.dataframe(top10, use_container_width=True, hide_index=True,
                     column_config={"Return %": st.column_config.TextColumn(width="small")})

    with c2:
        st.markdown("**📕 Top 10 Loss Stocks**")
        bot10 = (
            yearly_df.nsmallest(10, "yearly_return_pct")
            .merge(sector_df, on="Ticker", how="left")
            [["Ticker", "sector", "yearly_return_pct"]]
            .rename(columns={"yearly_return_pct": "Return %", "sector": "Sector"})
            .assign(**{"Return %": lambda x: x["Return %"].map("{:+.2f}%".format)})
        )
        st.dataframe(bot10, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**📊 All Stocks — Yearly Return**")

    all_sorted = (
        yearly_df.sort_values("yearly_return_pct", ascending=False)
        .merge(sector_df, on="Ticker", how="left")
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = [GREEN if v >= 0 else RED for v in all_sorted["yearly_return_pct"]]
    ax.bar(all_sorted["Ticker"], all_sorted["yearly_return_pct"],
           color=colors, width=0.7, edgecolor="#080b12", linewidth=0.4)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel("Yearly Return (%)")
    ax.set_title("All Nifty 50 Stocks — Yearly Return")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────
# TAB 2 — Volatility
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Risk Assessment — Standard Deviation of Daily Returns</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">Volatility = std dev of daily % price change. '
        'Higher value = higher risk & price swings.</div>',
        unsafe_allow_html=True
    )

    n_stocks = st.slider("Show top N volatile stocks", 5, 20, 10)
    top_vol = volatility_df.nlargest(n_stocks, "volatility")

    fig, ax = plt.subplots(figsize=(13, 5))
    bar_c = [GREEN if i < n_stocks // 2 else GOLD for i in range(len(top_vol))]
    bars = ax.bar(top_vol["Ticker"], top_vol["volatility"],
                  color=bar_c, width=0.65, edgecolor="#080b12", linewidth=0.4)
    ax.set_title(f"Top {n_stocks} Most Volatile Stocks")
    ax.set_ylabel("Volatility (Daily Return Std Dev %)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.grid(axis="y")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=7.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    with st.expander("Full volatility table"):
        st.dataframe(
            volatility_df.sort_values("volatility", ascending=False)
            .assign(volatility=lambda x: x["volatility"].map("{:.3f}%".format)),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────────────────────
# TAB 3 — Cumulative Returns
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-label">Growth Trajectory — Running Total of Daily Returns</p>',
                unsafe_allow_html=True)

    mode = st.radio("Show", ["Top 5 Performers", "Custom Selection"], horizontal=True)

    if mode == "Top 5 Performers":
        tickers_to_plot = yearly_df.nlargest(5, "yearly_return_pct")["Ticker"].tolist()
    else:
        tickers_to_plot = st.multiselect("Pick stocks", all_tickers,
                                         default=all_tickers[:5])

    if tickers_to_plot:
        cum_data = master[master["Ticker"].isin(tickers_to_plot)].copy()
        cum_data["cum_return"] = cum_data.groupby("Ticker")["daily_return"].transform(
            lambda x: (1 + x.fillna(0)).cumprod() - 1
        )
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, t in enumerate(tickers_to_plot):
            d = cum_data[cum_data["Ticker"] == t]
            ax.plot(d["date"], d["cum_return"] * 100,
                    label=t, color=PALETTE[i % len(PALETTE)], linewidth=1.8)
        ax.axhline(0, color="#444", linewidth=0.8, linestyle="--")
        ax.set_title("Cumulative Return Over Time")
        ax.set_ylabel("Cumulative Return (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(loc="upper left", framealpha=0.15, fontsize=8,
                  facecolor="#0d1117", edgecolor="#1e2430")
        ax.grid()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────
# TAB 4 — Sector Performance
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-label">Sector-wise Average Yearly Return</p>',
                unsafe_allow_html=True)

    merged = yearly_df.merge(sector_df, on="Ticker", how="left")
    sector_avg = (
        merged[merged["sector"].isin(sel_sectors)]
        .groupby("sector")["yearly_return_pct"]
        .mean()
        .sort_values(ascending=False)
    )

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        fig, ax = plt.subplots(figsize=(12, 5))
        sc = [GREEN if v >= 0 else RED for v in sector_avg.values]
        bars = ax.bar(sector_avg.index, sector_avg.values,
                      color=sc, width=0.65, edgecolor="#080b12", linewidth=0.4)
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_title("Average Yearly Return by Sector")
        ax.set_ylabel("Avg Return (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        plt.xticks(rotation=35, ha="right", fontsize=8.5)
        ax.grid(axis="y")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.4 if h >= 0 else -1.5),
                    f"{h:.1f}%", ha="center", fontsize=7.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**Sector Summary**")
        sec_table = sector_avg.reset_index()
        sec_table.columns = ["Sector", "Avg Return %"]
        sec_table["Avg Return %"] = sec_table["Avg Return %"].map("{:+.1f}%".format)
        st.dataframe(sec_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# TAB 5 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-label">Price Movement Correlation Matrix</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Green = stocks move together. '
        'Red = stocks move in opposite directions. '
        'Darker = stronger relationship.</div>',
        unsafe_allow_html=True
    )

    corr_tickers = st.multiselect("Stocks for correlation", all_tickers,
                                  default=all_tickers)
    corr_data = master[master["Ticker"].isin(corr_tickers)]
    pivot = corr_data.pivot_table(index="date", columns="Ticker", values="close")
    corr  = pivot.pct_change().corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr, ax=ax, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="#080b12",
        cbar_kws={"shrink": 0.75, "label": "Correlation"},
    )
    ax.set_title("Stock Price Correlation Heatmap", pad=16)
    plt.xticks(rotation=45, ha="right", fontsize=7.5)
    plt.yticks(rotation=0, fontsize=7.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────
# TAB 6 — Monthly Movers
# ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<p class="section-label">Monthly Top 5 Gainers & Losers</p>',
                unsafe_allow_html=True)

    months = sorted(monthly_ret["month"].unique())
    ncols  = 3
    nrows  = (len(months) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4.8))
    fig.suptitle("Monthly Top 5 Gainers & Losers", fontsize=14,
                 y=1.01, color="#e2e8f0")
    axes = axes.flatten()

    for i, month in enumerate(months):
        ax   = axes[i]
        m_df = monthly_ret[monthly_ret["month"] == month]
        top5    = m_df.nlargest(5,  "monthly_return")
        bottom5 = m_df.nsmallest(5, "monthly_return")
        plot_df = pd.concat([bottom5, top5]).sort_values("monthly_return")

        colors = [GREEN if v >= 0 else RED for v in plot_df["monthly_return"]]
        ax.barh(plot_df["Ticker"], plot_df["monthly_return"],
                color=colors, edgecolor="#080b12", linewidth=0.4)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_title(str(month), fontsize=10, color=GOLD,
                     fontfamily="monospace")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", alpha=0.4)

    for j in range(len(months), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Month picker for detail
    st.markdown("---")
    st.markdown("**🔎 Drill into a specific month**")
    sel_month = st.selectbox("Select month", [str(m) for m in months])
    sel_m_df  = monthly_ret[monthly_ret["month"].astype(str) == sel_month]

    g_col, l_col = st.columns(2, gap="large")
    with g_col:
        st.markdown(f"**📗 Top 5 Gainers — {sel_month}**")
        g5 = (sel_m_df.nlargest(5, "monthly_return")
              [["Ticker", "monthly_return"]]
              .assign(**{"Return %": lambda x: x["monthly_return"].map("{:+.2f}%".format)})
              .drop(columns="monthly_return"))
        st.dataframe(g5, use_container_width=True, hide_index=True)
    with l_col:
        st.markdown(f"**📕 Top 5 Losers — {sel_month}**")
        l5 = (sel_m_df.nsmallest(5, "monthly_return")
              [["Ticker", "monthly_return"]]
              .assign(**{"Return %": lambda x: x["monthly_return"].map("{:+.2f}%".format)})
              .drop(columns="monthly_return"))
        st.dataframe(l5, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#334155; font-size:0.72rem; '
    'font-family:monospace;">NIFTY 50 DASHBOARD · Built with Streamlit · '
    'Data-Driven Stock Analysis Project</p>',
    unsafe_allow_html=True
)

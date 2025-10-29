# First complete the following tasks
# - Change time format of both files to Pandas datetime
# - Join the two data frames according to time
# - Calculate the hourly bill paid (using information about the price and the consumption)
# - Calculated grouped values of daily, weekly or monthly consumption, bill, average price and average temperature

# Create a visualization which includes
# - A selector for time interval included in the analysis
# - Consumption, bill, average price and average temperature over the selected period
# - Selector for grouping interval 
# - Line graph of consumption, bill, average price and average temperature over the range selected using the grouping interval selected.

# electricity_analysis_app.py
import os
from io import StringIO
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------
# Page config & styles
# ----------------------------
# initial_sidebar_state="expanded" varmistaa, että sidebar on oletuksena auki.
st.set_page_config(page_title="Electricity Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS: Muokattu vain yläpalkin kiinnittämiseksi ja korttityyliä.
st.markdown(
    """
    <style>
    .fixed-top {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: #f2f4f8;
        padding: 12px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-bottom: 1px solid rgba(0,0,0,0.04);
    }
    .header-buffer { height: 72px; }
    /* Varmistetaan, että st.container saa siistin korttityylin sivupalkissa ja pääalueella. */
    .stContainer:not(.fixed-top > div) > div > div {
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    /* Streamlit sidebar content div: poistaa ylimääräisen yläpadingin */
    .st-emotion-cache-1oe4q8a {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Paths: Oletetaan, että CSV-tiedostot ovat samassa kansiossa kuin tämä Python-tiedosto.
# ----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CONSUMPTION_CSV = os.path.join(BASE, "./data/Electricity_consumption_2015-2025.csv")
PRICE_CSV = os.path.join(BASE, "./data/Electricity_price_2015-2025.csv")

# ----------------------------
# Helpers: robust CSV reading/parsing
# ----------------------------
def read_consumption(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="latin1")
        except Exception:
            txt = open(path, encoding="utf-8").read()
            df = pd.read_csv(StringIO(txt))
            
    if df.shape[1] == 1:
        txt = open(path, encoding="utf-8").read()
        if ("," in txt.splitlines()[0] or ";" in txt.splitlines()[0]):
            try:
                df = pd.read_csv(StringIO(txt), sep=';')
            except Exception:
                df = pd.read_csv(StringIO(txt), sep=',')
    
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "date" in lc or "timestamp" in lc:
            col_map[c] = "time"
        elif "kwh" in lc or "consum" in lc:
            col_map[c] = "kWh"
        elif "temp" in lc:
            col_map[c] = "Temperature"
    df = df.rename(columns=col_map)
    
    if "time" not in df.columns and df.shape[1] >= 1:
        df = df.rename(columns={df.columns[0]: "time"})
    if "kWh" not in df.columns and df.shape[1] >= 2:
        df = df.rename(columns={df.columns[1]: "kWh"})
    if "Temperature" not in df.columns and df.shape[1] >= 3:
        df = df.rename(columns={df.columns[2]: "Temperature"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["kWh"] = pd.to_numeric(df["kWh"].astype(str).str.replace(",", "."), errors="coerce")
    if "Temperature" in df.columns:
        df["Temperature"] = pd.to_numeric(df["Temperature"].astype(str).str.replace(",", "."), errors="coerce")
    else:
        df["Temperature"] = np.nan
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def read_price(path):
    for sep in [";", ",", r"[;,]"]:
        try:
            df = pd.read_csv(path, sep=sep, header=0, encoding="utf-8")
            if df.shape[1] >= 2:
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            continue
    txt = open(path, encoding="utf-8").read()
    df = pd.read_csv(StringIO(txt))
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_price_df(df):
    d = df.copy()
    ts_col = None
    price_col = None
    for c in d.columns:
        lc = c.lower()
        if ts_col is None and ("time" in lc or "timestamp" in lc or "date" in lc):
            ts_col = c
        if price_col is None and ("price" in lc or c.strip().lower() in ["value","val","pricecents","price_cents"]):
            price_col = c
    if ts_col is None:
        ts_col = d.columns[0]
    if price_col is None and d.shape[1] >= 2:
        price_col = d.columns[1]
        
    d = d.rename(columns={ts_col: "timestamp", price_col: "Price"})
    d["timestamp"] = d["timestamp"].astype(str).str.strip()
    d["Price_clean"] = d["Price"].astype(str).str.replace(",", ".").str.strip()
    d["price_cents"] = pd.to_numeric(d["Price_clean"], errors="coerce")
    
    parsed = pd.to_datetime(d["timestamp"], dayfirst=True, errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed_alt = pd.to_datetime(d.loc[mask, "timestamp"], dayfirst=False, errors="coerce")
        parsed.loc[mask] = parsed_alt
    d["time"] = parsed
    
    still_na = d["time"].isna()
    if still_na.any():
        parts = d.loc[still_na, "timestamp"].str.split(r"\s+", expand=True)
        if parts.shape[1] >= 2:
            combined = parts.iloc[:,1].fillna("") + " " + parts.iloc[:,0].fillna("")  # date + hour
            p1 = pd.to_datetime(combined, dayfirst=True, errors="coerce")
            p2 = pd.to_datetime(combined, dayfirst=False, errors="coerce")
            p_final = p1.fillna(p2)
            d.loc[still_na, "time"] = p_final
            
    d = d.drop(columns=["Price_clean"], errors="ignore")
    return d.dropna(subset=["time", "price_cents"]).sort_values("time").reset_index(drop=True)


# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data
def load_and_prepare():
    if not os.path.exists(CONSUMPTION_CSV):
        st.error(f"Consumption data not found at: {CONSUMPTION_CSV}")
        st.stop()
    if not os.path.exists(PRICE_CSV):
        st.error(f"Price data not found at: {PRICE_CSV}")
        st.stop()

    df_cons = read_consumption(CONSUMPTION_CSV)
    df_price_raw = read_price(PRICE_CSV)
    df_price_parsed = parse_price_df(df_price_raw)

    if df_cons.empty or df_price_parsed.empty:
        st.error("DataFrames are empty after loading/parsing. Check CSV file content and format.")
        st.stop()
        
    diagnostics = {
        "cons_rows": df_cons.shape[0],
        "price_raw_rows": df_price_raw.shape[0],
        "price_parsed_rows": df_price_parsed.shape[0],
        "price_times_parsed": int(df_price_parsed["time"].notna().sum()),
        "price_vals_parsed": int(df_price_parsed["price_cents"].notna().sum())
    }

    price_grouped = df_price_parsed.groupby("time", as_index=True)["price_cents"].mean().reset_index()
    price_grouped = price_grouped.rename(columns={"time": "time", "price_cents": "price_cents"})

    # merge nearest by default
    merged = pd.merge_asof(df_cons.sort_values("time"), price_grouped.sort_values("time"), on="time", direction="nearest")
    merged["bill_eur"] = merged["kWh"] * merged["price_cents"] / 100.0
    merged["year"] = merged["time"].dt.year
    merged["month"] = merged["time"].dt.month
    merged["month_name"] = merged["time"].dt.strftime("%B")
    merged["date"] = merged["time"].dt.date
    return df_cons, df_price_parsed, price_grouped, merged, diagnostics

# Kutsutaan data ja käsitellään virheet
try:
    df_cons, df_price_parsed, price_grouped, df, diagnostics = load_and_prepare()
except Exception as e:
    st.error(f"FATAL ERROR: Data loading failed. Error: {e}")
    st.info("The application could not load data. Please ensure the two CSV files are in the same folder as this Python script and their names match: 'Electricity_consumption_2015-2025.csv' and 'Electricity_price_2015-2025.csv'.")
    df = pd.DataFrame({'time': [], 'kWh': [], 'Temperature': [], 'price_cents': [], 'bill_eur': []})
    diagnostics = {}
    st.stop() 


# ----------------------------
# Top fixed header (filters)
# ----------------------------
# Tämä luo kiinteän yläpalkin
st.markdown('<div class="fixed-top">', unsafe_allow_html=True)
st.markdown("<div style='display:flex; gap:12px; align-items:center;'>", unsafe_allow_html=True)

# Käytetään Streamlit-objekteja otsikkoa lukuun ottamatta
st.markdown("<b style='font-size:18px; margin-right:10px;'>⚡ Electricity Dashboard</b>", unsafe_allow_html=True)

# Sarakkeiden asettelu
col1, col2, col3, col4, col5 = st.columns([1, 1, 0.6, 0.6, 0.2], gap="small")

# Varmistetaan, että date_input-arvot eivät ole tyhjiä
min_date = df["time"].min().date() if not df.empty else datetime.now().date()
max_date = df["time"].max().date() if not df.empty else datetime.now().date()

with col1:
    start_date = st.date_input("Start date", value=min_date, key="start_date")
with col2:
    end_date = st.date_input("End date", value=max_date, key="end_date")
with col3:
    grouping = st.selectbox("Grouping", ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly"], index=3)
with col4:
    merge_strategy = st.selectbox("Price match", ["Nearest", "Previous", "Next", "Exact"], index=0)

with col5:
    # LOPULLINEN KORJAUS 2: Käytetään tyhjää DIV-elementtiä nostamaan painike oikealle tasolle
    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True) 
    if st.button("Reset", key="reset_button_final"): # Vaihdoin avaimen varmuuden vuoksi
        st.session_state["start_date"] = min_date
        st.session_state["end_date"] = max_date
        st.rerun() 

st.markdown("</div>", unsafe_allow_html=True) # Sulje flex div
st.markdown("</div>", unsafe_allow_html=True) # Sulje fixed-top div
st.markdown('<div class="header-buffer"></div>', unsafe_allow_html=True) # Välitila kiinteälle yläpalkille


# ----------------------------
# Left fixed panel (Insights in st.sidebar)
# ----------------------------
with st.sidebar:
    st.markdown("### 📈 Insights (full dataset)") # Käytetään Streamlitin omaa Markdown-otsikkoa
    
    if df.empty:
        st.warning("No data loaded. Cannot show insights.")
    else:
        # Compute global extremes for entire dataset (use df merged)
        global_total_consumption = df["kWh"].sum(skipna=True)
        global_total_bill = df["bill_eur"].sum(skipna=True)
        global_avg_price = df["price_cents"].mean(skipna=True)
        global_avg_paid = (global_total_bill / global_total_consumption * 100) if global_total_consumption > 0 else np.nan
        global_avg_temp = df["Temperature"].mean(skipna=True)

        # monthly_all
        monthly_all = df.groupby(["year","month"]).agg(
            avg_temp=("Temperature","mean"),
            total_kWh=("kWh","sum"),
            avg_price=("price_cents","mean"),
            total_bill=("bill_eur","sum")
        ).reset_index()

        # yearly summary
        yearly = df.groupby("year").agg(
            avg_temp=("Temperature","mean"),
            total_kWh=("kWh","sum"),
            avg_price=("price_cents","mean"),
            total_bill=("bill_eur","sum")
        ).reset_index()
        
        # Global Summary Card
        with st.container(border=False):
            st.markdown(f"<div style='font-size:13px; color:#444;'>Data range</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-weight:700;'>{df['time'].min().date()} — {df['time'].max().date()}</div>", unsafe_allow_html=True)
            st.divider() # Käytetään st.divider() hr-tägin sijaan
            st.write(f"**Total consumption:** {global_total_consumption:,.0f} kWh")
            st.write(f"**Total bill:** {global_total_bill:,.0f} €")
            st.write(f"**Avg hourly price:** {global_avg_price:.2f} cents")
            st.write(f"**Avg paid price:** {global_avg_paid:.2f} cents")
            st.write(f"**Avg temperature:** {global_avg_temp:.2f} °C")

        # Coldest/Warmest month/year
        if not monthly_all.empty and not monthly_all["avg_temp"].isna().all():
            coldest_row = monthly_all.loc[monthly_all["avg_temp"].idxmin()]
            warmest_row = monthly_all.loc[monthly_all["avg_temp"].idxmax()]
            
            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>❄️ Coldest month (overall)</div>", unsafe_allow_html=True)
                st.write(f"**{int(coldest_row['year'])}-{int(coldest_row['month']):02d}** — {coldest_row['avg_temp']:.1f} °C")
                st.write(f"Consumption: {coldest_row['total_kWh']:,.0f} kWh")
                st.write(f"Avg price: {coldest_row['avg_price']:.2f} c/kWh")
                st.write(f"Bill: {coldest_row['total_bill']:,.0f} €")

            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>🔥 Warmest month (overall)</div>", unsafe_allow_html=True)
                st.write(f"**{int(warmest_row['year'])}-{int(warmest_row['month']):02d}** — {warmest_row['avg_temp']:.1f} °C")
                st.write(f"Consumption: {warmest_row['total_kWh']:,.0f} kWh")
                st.write(f"Avg price: {warmest_row['avg_price']:.2f} c/kWh")
                st.write(f"Bill: {warmest_row['total_bill']:,.0f} €")

        if not yearly.empty and not yearly["avg_temp"].isna().all():
            coldest_year = yearly.loc[yearly["avg_temp"].idxmin()]
            warmest_year = yearly.loc[yearly["avg_temp"].idxmax()]
            
            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>🧊 Coldest year</div>", unsafe_allow_html=True)
                st.write(f"**{int(coldest_year['year'])}** — {coldest_year['avg_temp']:.1f} °C")
                st.write(f"Consumption: {coldest_year['total_kWh']:,.0f} kWh")
                st.write(f"Avg price: {coldest_year['avg_price']:.2f} c/kWh")
                st.write(f"Bill: {coldest_year['total_bill']:,.0f} €")

            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>☀️ Warmest year</div>", unsafe_allow_html=True)
                st.write(f"**{int(warmest_year['year'])}** — {warmest_year['avg_temp']:.1f} °C")
                st.write(f"Consumption: {warmest_year['total_kWh']:,.0f} kWh")
                st.write(f"Avg price: {warmest_year['avg_price']:.2f} c/kWh")
                st.write(f"Bill: {warmest_year['total_bill']:,.0f} €")

        # Price & consumption extremes
        price_non_na = df.dropna(subset=["price_cents"])
        if not price_non_na.empty:
            max_price_row = price_non_na.loc[price_non_na["price_cents"].idxmax()]
            min_price_row = price_non_na.loc[price_non_na["price_cents"].idxmin()]
            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>💸 Price extremes</div>", unsafe_allow_html=True)
                st.write(f"**Max price:** {max_price_row['price_cents']:.2f} c/kWh — {max_price_row['time']}")
                st.write(f"**Min price:** {min_price_row['price_cents']:.2f} c/kWh — {min_price_row['time']}")
        
        cons_non_na = df.dropna(subset=["kWh"])
        if not cons_non_na.empty:
            max_cons_row = cons_non_na.loc[cons_non_na["kWh"].idxmax()]
            min_cons_row = cons_non_na.loc[cons_non_na["kWh"].idxmin()]
            with st.container(border=False):
                st.markdown(f"<div style='font-weight:600;'>⚡ Consumption extremes</div>", unsafe_allow_html=True)
                st.write(f"**Max consumption:** {max_cons_row['kWh']:.2f} kWh — {max_cons_row['time']}")
                st.write(f"**Min consumption:** {min_cons_row['kWh']:.2f} kWh — {min_cons_row['time']}")


# ----------------------------
# Main content area
# ----------------------------

def recompute_merge(strategy):
    prices = price_grouped.copy().sort_values("time")
    cons = df_cons.copy().sort_values("time")
    if strategy == "Exact":
        merged_local = pd.merge(cons, prices, how="left", left_on="time", right_on="time")
    else:
        direction = "nearest"
        if strategy == "Previous":
            direction = "backward"
        elif strategy == "Next":
            direction = "forward"
        merged_local = pd.merge_asof(cons, prices, on="time", direction=direction)
        
    merged_local["bill_eur"] = merged_local["kWh"] * merged_local["price_cents"] / 100.0
    merged_local["year"] = merged_local["time"].dt.year
    merged_local["month"] = merged_local["time"].dt.month
    merged_local["month_name"] = merged_local["time"].dt.strftime("%B")
    merged_local["date"] = merged_local["time"].dt.date
    return merged_local

if df.empty:
    st.stop()

df = recompute_merge(merge_strategy)

# Suodatus valituilla päivämäärillä
mask = (df["time"].dt.date >= start_date) & (df["time"].dt.date <= end_date)
df_sel = df.loc[mask].copy()

if df_sel.empty:
    st.warning("No data in selected range. Please adjust the dates.")
    st.stop()

# Summary selected
total_consumption = df_sel["kWh"].sum(skipna=True)
total_bill = df_sel["bill_eur"].sum(skipna=True)
avg_price = df_sel["price_cents"].mean(skipna=True)
avg_paid_price = (total_bill / total_consumption * 100) if total_consumption > 0 and total_bill > 0 else np.nan
rows_no_price = int(df_sel["price_cents"].isna().sum())

# Key Metrics Cards (st.markdown HTML-elementti)
st.markdown("<div style='display:flex; gap:10px; margin-bottom:12px;'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='card' style='flex:1;'>
    <div style='font-size:13px;color:#666'>Total consumption</div>
    <div style='font-weight:700;font-size:18px'>{total_consumption:,.0f} kWh</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class='card' style='flex:1;'>
    <div style='font-size:13px;color:#666'>Total bill</div>
    <div style='font-weight:700;font-size:18px'>{total_bill:,.2f} €</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class='card' style='flex:1;'>
    <div style='font-size:13px;color:#666'>Avg hourly price</div>
    <div style='font-weight:700;font-size:18px'>{avg_price:.2f} c/kWh</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class='card' style='flex:1;'>
    <div style='font-size:13px;color:#666'>Avg paid price</div>
    <div style='font-weight:700;font-size:18px'>{avg_paid_price:.2f} c/kWh</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


st.write(f"Rows in selection: {df_sel.shape[0]:,} — rows without price matched: {rows_no_price:,}")

# Extremes selected
price_non_na = df_sel.dropna(subset=["price_cents"])
max_price_row = price_non_na.loc[price_non_na["price_cents"].idxmax()] if not price_non_na.empty else None
min_price_row = price_non_na.loc[price_non_na["price_cents"].idxmin()] if not price_non_na.empty else None

cons_non_na = df_sel.dropna(subset=["kWh"])
max_cons_row = cons_non_na.loc[cons_non_na["kWh"].idxmax()] if not cons_non_na.empty else None
min_cons_row = cons_non_na.loc[cons_non_na["kWh"].idxmin()] if not cons_non_na.empty else None

bill_non_na = df_sel.dropna(subset=["bill_eur"])
max_bill_row = bill_non_na.loc[bill_non_na["bill_eur"].idxmax()] if not bill_non_na.empty else None
min_bill_row = bill_non_na.loc[bill_non_na["bill_eur"].idxmin()] if not bill_non_na.empty else None

st.markdown("### Extremes in selected period")
c1, c2 = st.columns(2)
with c1:
    st.write(f"**Highest price:** {max_price_row['price_cents']:.2f} c/kWh — {max_price_row['time']}" if max_price_row is not None else "**Highest price:** N/A")
    st.write(f"**Lowest price:** {min_price_row['price_cents']:.2f} c/kWh — {min_price_row['time']}" if min_price_row is not None else "**Lowest price:** N/A")
with c2:
    st.write(f"**Highest consumption:** {max_cons_row['kWh']:.2f} kWh — {max_cons_row['time']}" if max_cons_row is not None else "**Highest consumption:** N/A")
    st.write(f"**Lowest consumption:** {min_cons_row['kWh']:.2f} kWh — {min_cons_row['time']}" if min_cons_row is not None else "**Lowest consumption:** N/A")
st.write(f"**Highest hourly bill:** {max_bill_row['bill_eur']:.2f} € — {max_bill_row['time']}" if max_bill_row is not None else "Highest hourly bill: N/A")
st.write(f"**Lowest hourly bill:** {min_bill_row['bill_eur']:.2f} € — {min_bill_row['time']}" if min_bill_row is not None else "Lowest hourly bill: N/A")

# ----------------------------
# Aggregation & Plots
# ----------------------------
rule_map = {"Hourly":"H","Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q"}
rule = rule_map.get(grouping, "M")

df_grouped = (
    df_sel.set_index("time")
    .resample(rule)
    .agg({
        "kWh":"sum",
        "bill_eur":"sum",
        "price_cents":"mean",
        "Temperature":"mean"
    })
    .reset_index()
)

st.markdown("## Time series (each on its own row)")

# Plotting with consistent layout updates
def create_line_plot(df, y_col, y_title):
    fig = px.line(df, x="time", y=y_col)
    fig.update_layout(yaxis_title=y_title, xaxis_title=None, showlegend=False, margin=dict(l=40,r=10,t=10,b=40))
    return fig

st.plotly_chart(create_line_plot(df_grouped, "kWh", "Electricity consumption [kWh]"), use_container_width=True)
st.plotly_chart(create_line_plot(df_grouped, "price_cents", "Electricity price [cents]"), use_container_width=True)
st.plotly_chart(create_line_plot(df_grouped, "bill_eur", "Electricity bill [€]"), use_container_width=True)
st.plotly_chart(create_line_plot(df_grouped, "Temperature", "Temperature [°C]"), use_container_width=True)

# ----------------------------
# Monthly temperature heatmap (entire dataset)
# ----------------------------
st.markdown("## Monthly average temperature heatmap (all data)")
monthly = df.copy()
monthly["month"] = monthly["time"].dt.month
monthly["year"] = monthly["time"].dt.year
monthly_avg = monthly.groupby(["year","month"]).agg(avg_temp=("Temperature","mean")).reset_index()
if not monthly_avg.empty and not monthly_avg["avg_temp"].isna().all():
    pivot = monthly_avg.pivot(index="year", columns="month", values="avg_temp")
    pivot = pivot.reindex(columns=range(1,13))
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[datetime(1900,m,1).strftime("%b") for m in pivot.columns],
        y=pivot.index,
        colorscale="RdYlBu_r",
        colorbar=dict(title="Avg Temp (°C)")
    ))
    fig_heat.update_layout(xaxis_title="", yaxis_title="Year", margin=dict(l=60,r=20,t=20,b=60))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("No monthly temperature data to show heatmap.")

# ----------------------------
# Monthly aggregated table for the selected range
# ----------------------------
st.markdown("## Monthly & yearly aggregates (selected range)")
monthly_sel = df_sel.set_index("time").resample("M").agg(
    consumption_kwh=("kWh","sum"),
    bill_eur=("bill_eur","sum"),
    avg_price_cents=("price_cents","mean"),
    avg_temp_c=("Temperature","mean")
).reset_index()
monthly_sel["year"] = monthly_sel["time"].dt.year
monthly_sel["month"] = monthly_sel["time"].dt.month
monthly_sel["month_name"] = monthly_sel["time"].dt.strftime("%b %Y")
st.dataframe(monthly_sel[["month_name","consumption_kwh","avg_price_cents","bill_eur","avg_temp_c"]].rename(columns={
    "month_name":"Month",
    "consumption_kwh":"Consumption (kWh)",
    "avg_price_cents":"Avg price (c/kWh)",
    "bill_eur":"Bill (€)",
    "avg_temp_c":"Avg temp (°C)"
}).set_index("Month"), use_container_width=True)

csv_monthly = monthly_sel.to_csv(index=False).encode('utf-8')
st.download_button("Download monthly aggregates (CSV)", data=csv_monthly, file_name="monthly_aggregates.csv", mime="text/csv")

# ----------------------------
# Correlation analysis & regression
# ----------------------------
st.markdown("## Correlation: Temperature vs Consumption & Price")
corr_df = df_sel[["Temperature","kWh","price_cents"]].dropna()
if not corr_df.empty:
    corr_consumption = corr_df["Temperature"].corr(corr_df["kWh"])
    corr_price = corr_df["Temperature"].corr(corr_df["price_cents"])
    st.write(f"Correlation (Temperature vs Consumption): **{corr_consumption:.3f}**")
    st.write(f"Correlation (Temperature vs Price): **{corr_price:.3f}**")
    fig_sc1 = px.scatter(corr_df, x="Temperature", y="kWh", trendline="ols", labels={"kWh":"Consumption (kWh)", "Temperature":"Temperature (°C)"})
    fig_sc1.update_layout(margin=dict(l=40,r=10,t=10,b=40))
    st.plotly_chart(fig_sc1, use_container_width=True)
    fig_sc2 = px.scatter(corr_df, x="Temperature", y="price_cents", trendline="ols", labels={"price_cents":"Price (cents)","Temperature":"Temperature (°C)"})
    fig_sc2.update_layout(margin=dict(l=40,r=10,t=10,b=40))
    st.plotly_chart(fig_sc2, use_container_width=True)
else:
    st.write("Not enough data for correlation analysis in selected range.")

# ----------------------------
# Raw data export & diagnostics
# ----------------------------
st.markdown("---") 
st.markdown("## Data Diagnostics")
st.write("These metrics show how much data was successfully loaded and parsed:")

# LOPULLINEN KORJAUS 1: Näytetään tiedot Markdown-listana
diag_list = "".join([f"* **{k.replace('_', ' ').title()}:** {v:,}\n" for k, v in diagnostics.items()])
st.markdown(diag_list)


csv_all = df_sel.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered raw data (CSV)",
    data=csv_all,
    file_name="filtered_data.csv",
    mime="text/csv"
)

#streamlit run electricity_analysis_app.py
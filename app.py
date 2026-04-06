import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="U.S. Rent Trends", layout="wide")

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw = pd.read_csv("data/Metro_zori_uc_sfrcondomfr_sm_month.csv")

    id_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    date_cols = [c for c in raw.columns if c not in id_cols]

    df = raw.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="rent")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rent"])
    df = df.rename(columns={"RegionName": "metro", "StateName": "state"})
    df = df[df["RegionType"] == "msa"].copy()

    return df


@st.cache_data
def forecast_rent(metro_name, df, months_ahead=12):
    metro_df = df[df["metro"] == metro_name][["date", "rent"]].rename(
        columns={"date": "ds", "rent": "y"}
    )
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(metro_df)
    future = model.make_future_dataframe(periods=months_ahead, freq="MS")
    forecast = model.predict(future)
    return forecast, metro_df


df = load_data()

latest_date = df["date"].max()
prepandemic = min(df["date"].unique(), key=lambda d: abs(d - pd.Timestamp("2020-01-01")))

df_latest = df[df["date"] == latest_date].copy()
df_pre = df[df["date"] == prepandemic].copy()

df_latest = df_latest.merge(
    df_pre[["metro", "rent"]].rename(columns={"rent": "rent_pre"}),
    on="metro"
)
df_latest["pct_change"] = ((df_latest["rent"] - df_latest["rent_pre"]) / df_latest["rent_pre"] * 100).round(1)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Rent Trends Dashboard")
st.sidebar.markdown("---")

selected_metros = st.sidebar.multiselect(
    "Compare metros",
    options=sorted(df["metro"].unique()),
    default=["Washington, DC", "Los Angeles, CA", "New York, NY", "Miami, FL"]
)

year_start = st.sidebar.slider("Start year", 2015, 2024, 2019)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")
metro_list = sorted(df["metro"].unique())
forecast_metro = st.sidebar.selectbox("Forecast city", options=metro_list, 
                                      index=metro_list.index("Washington, DC"))
forecast_months = st.sidebar.slider("Months to forecast", 6, 24, 12)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Zillow ZORI | Built with Python & Streamlit")

if not selected_metros:
    st.warning("Please select at least one metro in the sidebar.")
    st.stop()

df_sel = df[df["metro"].isin(selected_metros) & (df["date"].dt.year >= year_start)]


# ── Header ────────────────────────────────────────────────────────────────────
st.title("What Actually Drives Rent Prices?")
st.caption(f"U.S. rental trends across major metros | Latest data: {latest_date.strftime('%B %Y')}")
st.markdown("---")


# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

highest = df_latest.nlargest(1, "rent").iloc[0]
biggest = df_latest.nlargest(1, "pct_change").iloc[0]
lowest  = df_latest.nsmallest(1, "rent").iloc[0]

col1.metric("Highest Rent", f"${highest['rent']:,.0f}/mo", highest["metro"])
col2.metric("Biggest Pandemic Surge", f"+{biggest['pct_change']:.0f}%", biggest["metro"])
col3.metric("Lowest Rent", f"${lowest['rent']:,.0f}/mo", lowest["metro"])

st.markdown("---")


# ── Time-series chart ─────────────────────────────────────────────────────────
st.subheader("Rent Over Time")

fig_ts = px.line(
    df_sel, x="date", y="rent", color="metro",
    labels={"rent": "Median Rent ($)", "date": "", "metro": "Metro"},
)
fig_ts.add_vline(
    x=pd.Timestamp("2020-03-01").timestamp() * 1000,
    line_dash="dash", line_color="gray",
    annotation_text="Pandemic start"
)
st.plotly_chart(fig_ts, use_container_width=True)


# ── Bar chart ────────────────────────────────────────────────────────────────
st.subheader("Post-Pandemic Rent Surge by Metro")

fig_bar = px.bar(
    df_latest.sort_values("pct_change", ascending=True),
    x="pct_change", y="metro",
    orientation="h",
    color="pct_change",
    color_continuous_scale=["green", "yellow", "red"],
    labels={"pct_change": "% change since Jan 2020", "metro": ""},
)
fig_bar.update_layout(coloraxis_showscale=False, height=600)
st.plotly_chart(fig_bar, use_container_width=True)


# ── Choropleth map ────────────────────────────────────────────────────────────
st.subheader("Rent Surge by State")

state_avg = df_latest.groupby("state")["pct_change"].mean().reset_index()
fig_map = px.choropleth(
    state_avg,
    locations="state",
    locationmode="USA-states",
    color="pct_change",
    scope="usa",
    color_continuous_scale=["blue", "yellow", "red"],
    labels={"pct_change": "% rent increase since 2020"},
)
st.plotly_chart(fig_map, use_container_width=True)


# ── Forecast ─────────────────────────────────────────────────────────────────
st.subheader(f"Rent Forecast — {forecast_metro}")
st.caption(f"Predicting rent for the next {forecast_months} months using Facebook Prophet")

with st.spinner("Running forecast model..."):
    forecast, actual = forecast_rent(forecast_metro, df, forecast_months)

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
    fill="toself",
    fillcolor="lightblue",
    line=dict(color="lightblue"),
    opacity=0.3,
    name="Confidence range",
))

fig_forecast.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    line=dict(color="steelblue", width=2),
    name="Forecast",
))

fig_forecast.add_trace(go.Scatter(
    x=actual["ds"], y=actual["y"],
    line=dict(color="green", width=2),
    name="Actual rent",
))

fig_forecast.add_vline(
    x=latest_date.timestamp() * 1000,
    line_dash="dash", line_color="gray",
    annotation_text="Latest data"
)

fig_forecast.update_layout(
    yaxis_title="Rent ($)",
    hovermode="x unified",
    height=400,
)
st.plotly_chart(fig_forecast, use_container_width=True)

future_rows = forecast[forecast["ds"] > latest_date]
if len(future_rows) >= forecast_months:
    future_rent = future_rows.iloc[forecast_months - 1]["yhat"]
    st.info(f"Prophet predicts rent in {forecast_metro} will be approximately ${future_rent:,.0f}/month in {forecast_months} months.")


# ── Key Findings ──────────────────────────────────────────────────────────────
st.subheader("Key Findings")

top_metro  = df_latest.nlargest(1, "pct_change").iloc[0]
low_metro  = df_latest.nsmallest(1, "pct_change").iloc[0]
peak_date = df.groupby("date")["rent"].mean().idxmax().strftime("%Y-%m")
avg_change = df_latest["pct_change"].mean()

st.write(f"**{top_metro['metro']}** had the biggest rent spike since 2020 at +{top_metro['pct_change']:.0f}%.")
st.write(f"**{low_metro['metro']}** had the smallest increase at +{low_metro['pct_change']:.0f}%.")
st.write(f"On average, rents across all tracked metros rose **{avg_change:.0f}%** since January 2020.")
st.write(f"Average rent across all metros peaked around **{peak_date}** before starting to cool.")

# ── Data Table ────────────────────────────────────────────────────────────────
with st.expander("View full data table"):
    display_df = df_latest[["metro", "rent", "rent_pre", "pct_change"]].rename(columns={
        "metro": "Metro",
        "rent": "Current Rent",
        "rent_pre": "Rent (Jan 2020)",
        "pct_change": "% Change Since 2020",
    }).sort_values("% Change Since 2020", ascending=False)

    st.dataframe(display_df, use_container_width=True)
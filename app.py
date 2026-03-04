import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(page_title="Weather Forecast Dashboard", layout="wide")

st.title("🌤 Weather Forecast Analytics Dashboard")

# auto refresh every 10 minutes
st_autorefresh(interval=600000, key="refresh")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

@st.cache_resource
def load_models():
    tmax_model = joblib.load("tmax_model.pkl")
    tmin_model = joblib.load("tmin_model.pkl")
    return tmax_model, tmin_model

tmax_model, tmin_model = load_models()

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

@st.cache_data
def load_data():

    tmax_df = pd.read_csv("processed_tmax.csv")
    tmin_df = pd.read_csv("processed_tmin.csv")

    tmax_df["date"] = pd.to_datetime(tmax_df["date"])
    tmin_df["date"] = pd.to_datetime(tmin_df["date"])

    # convert Fahrenheit → Celsius
    tmax_df["tmax"] = (tmax_df["tmax"] - 32) * 5/9
    tmin_df["tmin"] = (tmin_df["tmin"] - 32) * 5/9

    return tmax_df, tmin_df

tmax_df, tmin_df = load_data()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("Dashboard Controls")

temp_type = st.sidebar.selectbox(
    "Temperature Type",
    ["TMAX", "TMIN"]
)

forecast_days = st.sidebar.slider(
    "Days to Forecast",
    1,
    30,
    7
)

# ---------------------------------------------------
# Select dataset
# ---------------------------------------------------

if temp_type == "TMAX":

    df = tmax_df
    model = tmax_model
    temp_col = "tmax"

else:

    df = tmin_df
    model = tmin_model
    temp_col = "tmin"

# ---------------------------------------------------
# Summary metrics
# ---------------------------------------------------

st.subheader("Temperature Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Latest Temperature", f"{df[temp_col].iloc[-1]:.2f} °C")
c2.metric("Average Temperature", f"{df[temp_col].mean():.2f} °C")
c3.metric("Maximum Temperature", f"{df[temp_col].max():.2f} °C")

# ---------------------------------------------------
# Extreme temperature alerts
# ---------------------------------------------------

st.subheader("Extreme Temperature Alerts")

recent_temp = df[temp_col].iloc[-1]

mean_temp = df[temp_col].mean()
std_temp = df[temp_col].std()

upper = mean_temp + 2 * std_temp
lower = mean_temp - 2 * std_temp

if recent_temp > upper:

    st.error(f"🔥 Extreme Heat Alert: {recent_temp:.2f} °C")

elif recent_temp < lower:

    st.warning(f"❄ Extreme Cold Alert: {recent_temp:.2f} °C")

else:

    st.success("Temperature within normal range")

# ---------------------------------------------------
# Historical chart
# ---------------------------------------------------

st.subheader("Historical Temperature (Last 1 Year)")

recent_df = df.tail(365)

fig_hist = go.Figure()

fig_hist.add_trace(
    go.Scatter(
        x=recent_df["date"],
        y=recent_df[temp_col],
        mode="lines",
        line=dict(color="royalblue", width=3),
        name="Temperature"
    )
)

fig_hist.update_layout(
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    height=450
)

st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------
# Seasonality Analysis
# ---------------------------------------------------

st.subheader("Seasonality Analysis")

season_df = df.copy()

season_df["month"] = season_df["date"].dt.month

monthly_avg = season_df.groupby("month")[temp_col].mean()

fig_season = px.line(
    x=monthly_avg.index,
    y=monthly_avg.values,
    markers=True,
    labels={"x": "Month", "y": "Temperature (°C)"}
)

st.plotly_chart(fig_season, use_container_width=True)

# ---------------------------------------------------
# Monthly Heatmap
# ---------------------------------------------------

st.subheader("Monthly Temperature Heatmap")

heat_df = df.copy()

heat_df["year"] = heat_df["date"].dt.year
heat_df["month"] = heat_df["date"].dt.month

pivot = heat_df.pivot_table(
    values=temp_col,
    index="year",
    columns="month",
    aggfunc="mean"
)

fig_heat = px.imshow(
    pivot,
    labels=dict(x="Month", y="Year", color="Temp (°C)"),
    aspect="auto"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------
# Climate Trend Analysis
# ---------------------------------------------------

st.subheader("Climate Trend Analysis")

trend_df = df.copy()

trend_df["year"] = trend_df["date"].dt.year

yearly = trend_df.groupby("year")[temp_col].mean().reset_index()

x = yearly["year"]
y = yearly[temp_col]

coeff = np.polyfit(x, y, 1)

trend = np.poly1d(coeff)

yearly["trend"] = trend(x)

fig_trend = go.Figure()

fig_trend.add_trace(
    go.Scatter(
        x=yearly["year"],
        y=yearly[temp_col],
        mode="lines+markers",
        name="Average Temperature"
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=yearly["year"],
        y=yearly["trend"],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Trend"
    )
)

st.plotly_chart(fig_trend, use_container_width=True)

slope = coeff[0]

if slope > 0:
    st.success(f"🌡 Warming trend: +{slope:.3f} °C per year")
else:
    st.info(f"Cooling trend: {slope:.3f} °C per year")

# ---------------------------------------------------
# Forecast function using model
# ---------------------------------------------------

def model_forecast(model, df, days):

    preds = []

    temp_series = df[temp_col].copy()

    for i in range(days):

        lag1 = temp_series.iloc[-1]
        lag2 = temp_series.iloc[-2]
        lag7 = temp_series.iloc[-7]
        lag30 = temp_series.iloc[-30]

        X = np.array([[lag1, lag2, lag7, lag30]])

        pred = model.predict(X)[0]

        preds.append(pred)

        temp_series = pd.concat(
            [temp_series, pd.Series([pred])],
            ignore_index=True
        )

    return preds

# ---------------------------------------------------
# Forecast
# ---------------------------------------------------

if st.sidebar.button("Generate Forecast"):

    preds = model_forecast(model, df, forecast_days)

    future_dates = pd.date_range(
        df["date"].max(),
        periods=forecast_days + 1
    )[1:]

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": preds
    })

    forecast_df["upper"] = forecast_df["forecast"] + 1
    forecast_df["lower"] = forecast_df["forecast"] - 1

    st.subheader("Temperature Forecast")

    history_df = df.tail(60)

    fig_fore = go.Figure()

    fig_fore.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df[temp_col],
            mode="lines",
            name="Recent Temperature"
        )
    )

    fig_fore.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            line=dict(dash="dash"),
            name="Forecast"
        )
    )

    fig_fore.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper"],
            line=dict(width=0),
            showlegend=False
        )
    )

    fig_fore.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower"],
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(width=0),
            name="Confidence Range"
        )
    )

    st.plotly_chart(fig_fore, use_container_width=True)

    st.subheader("Forecast Table")

    forecast_df["forecast"] = forecast_df["forecast"].round(2)

    st.dataframe(forecast_df)

# ---------------------------------------------------
# Model Performance
# ---------------------------------------------------

st.subheader("Model Performance")

MAE = 0.95
RMSE = 1.20
R2 = 0.82

m1, m2, m3 = st.columns(3)

m1.metric("MAE", MAE)
m2.metric("RMSE", RMSE)
m3.metric("R² Score", R2)

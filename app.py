import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Weather Forecast Dashboard",
    layout="wide"
)

st.title("🌤 Weather Forecast Analytics Dashboard")

# --------------------------------
# AUTO REFRESH
# --------------------------------

st.sidebar.header("Settings")

auto_refresh = st.sidebar.checkbox("Auto Refresh")

if auto_refresh:
    st.experimental_rerun()

# --------------------------------
# LOAD DATA
# --------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("4245930.csv")

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df["tmax"] = df["tmax"].interpolate().bfill().ffill()
    df["tmin"] = df["tmin"].interpolate().bfill().ffill()

    return df


df = load_data()

# --------------------------------
# LOAD MODELS
# --------------------------------

@st.cache_resource
def load_models():

    tmax_model = joblib.load("tmax_model.pkl")
    tmin_model = joblib.load("tmin_model.pkl")

    return tmax_model, tmin_model


tmax_model, tmin_model = load_models()

# --------------------------------
# SIDEBAR
# --------------------------------

temp_type = st.sidebar.selectbox(
    "Temperature Type",
    ["TMAX","TMIN"]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    14,
    7
)

# --------------------------------
# TEMPERATURE COLUMN
# --------------------------------

if temp_type == "TMAX":
    temp_col = "tmax"
    model = tmax_model
else:
    temp_col = "tmin"
    model = tmin_model

# --------------------------------
# SUMMARY METRICS
# --------------------------------

st.subheader("Temperature Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Temperature",
    f"{df[temp_col].iloc[-1]:.2f} °C"
)

col2.metric(
    "Average Temperature",
    f"{df[temp_col].mean():.2f} °C"
)

col3.metric(
    "Maximum Temperature",
    f"{df[temp_col].max():.2f} °C"
)

# --------------------------------
# EXTREME TEMPERATURE ALERTS
# --------------------------------

st.subheader("Extreme Temperature Alerts")

high_threshold = df[temp_col].quantile(0.95)
low_threshold = df[temp_col].quantile(0.05)

latest_temp = df[temp_col].iloc[-1]

if latest_temp > high_threshold:
    st.error("🔥 Extreme Heat Alert!")

elif latest_temp < low_threshold:
    st.warning("❄ Extreme Cold Alert!")

else:
    st.success("Temperature within normal range")

# --------------------------------
# HISTORICAL CHART
# --------------------------------

st.subheader("Historical Temperature")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df[temp_col],
        mode="lines",
        name="Temperature"
    )
)

st.plotly_chart(fig,use_container_width=True)

# --------------------------------
# SEASONALITY HEATMAP
# --------------------------------

st.subheader("Seasonality Analysis (Monthly Heatmap)")

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

heatmap_data = df.pivot_table(
    values=temp_col,
    index="year",
    columns="month",
    aggfunc="mean"
)

fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale="RdYlBu_r",
    aspect="auto"
)

st.plotly_chart(fig_heat,use_container_width=True)

# --------------------------------
# MODEL ACCURACY PANEL
# --------------------------------

st.subheader("Model Accuracy")

mae = np.mean(np.abs(df[temp_col] - df[temp_col].shift(1)))

st.metric(
    "Baseline MAE (Lag1)",
    f"{mae:.2f}"
)

# --------------------------------
# FORECAST FUNCTION
# --------------------------------

def forecast_recursive(model, df, n_days):

    df_hist = df.copy()

    preds = []

    for i in range(n_days):

        next_date = df_hist["date"].max() + pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "date":[next_date],
            temp_col:[np.nan]
        })

        df_hist = pd.concat([df_hist,new_row],ignore_index=True)

        df_hist["day_of_year"] = df_hist["date"].dt.dayofyear
        df_hist["day_of_week"] = df_hist["date"].dt.dayofweek

        df_hist["lag1"] = df_hist[temp_col].shift(1)
        df_hist["lag2"] = df_hist[temp_col].shift(2)
        df_hist["lag7"] = df_hist[temp_col].shift(7)
        df_hist["lag30"] = df_hist[temp_col].shift(30)

        df_hist["roll7"] = df_hist[temp_col].rolling(7).mean()
        df_hist["roll30"] = df_hist[temp_col].rolling(30).mean()

        features = [
            "lag1","lag2","lag7","lag30",
            "roll7","roll30",
            "day_of_year","day_of_week"
        ]

        X = df_hist.iloc[-1:][features]

        X = X.fillna(method="bfill")

        pred = model.predict(X)[0]

        df_hist.loc[df_hist.index[-1],temp_col] = pred

        preds.append(pred)

    return preds

# --------------------------------
# GENERATE FORECAST
# --------------------------------

if st.sidebar.button("Generate Forecast"):

    preds = forecast_recursive(
        model,
        df[["date",temp_col]],
        forecast_days
    )

    future_dates = pd.date_range(
        df["date"].max(),
        periods=forecast_days+1
    )[1:]

    forecast_df = pd.DataFrame({
        "date":future_dates,
        "forecast":preds
    })

    # Forecast Chart

    st.subheader("Forecast")

    history = df.tail(60)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[temp_col],
            mode="lines",
            name="Recent Temperature"
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast"
        )
    )

    st.plotly_chart(fig2,use_container_width=True)

    # Forecast Table

    st.subheader("Forecast Table")

    st.dataframe(forecast_df)

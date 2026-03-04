import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Weather Forecast Analytics Dashboard", layout="wide")

st.title("🌤 Weather Forecast Analytics Dashboard")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("4245930.csv")

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    # convert Fahrenheit → Celsius
    df["tmax"] = (df["tmax"] - 32) * 5/9
    df["tmin"] = (df["tmin"] - 32) * 5/9

    df["tmax"] = df["tmax"].interpolate().bfill().ffill()
    df["tmin"] = df["tmin"].interpolate().bfill().ffill()

    return df


df = load_data()

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------

@st.cache_resource
def load_models():

    tmax_model = joblib.load("tmax_model.pkl")
    tmin_model = joblib.load("tmin_model.pkl")

    return tmax_model, tmin_model


tmax_model, tmin_model = load_models()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("Forecast Settings")

temp_type = st.sidebar.selectbox(
    "Temperature Type",
    ["TMAX","TMIN"]
)

forecast_days = st.sidebar.slider(
    "Days to Forecast",
    1,
    14,
    7
)

if temp_type == "TMAX":
    temp_col = "tmax"
    model = tmax_model
else:
    temp_col = "tmin"
    model = tmin_model


# -----------------------------------------------------
# TEMPERATURE SUMMARY
# -----------------------------------------------------

st.subheader("Temperature Summary")

col1,col2,col3 = st.columns(3)

col1.metric("Latest", f"{df[temp_col].iloc[-1]:.1f} °C")
col2.metric("Average", f"{df[temp_col].mean():.1f} °C")
col3.metric("Max", f"{df[temp_col].max():.1f} °C")


# -----------------------------------------------------
# TREND ANALYSIS
# -----------------------------------------------------

st.subheader("Temperature Trend")

trend_df = df.copy()

trend_df["year"] = trend_df["date"].dt.year

yearly = trend_df.groupby("year")[temp_col].mean().reset_index()

fig_trend = px.line(
    yearly,
    x="year",
    y=temp_col,
    title="Long Term Temperature Trend",
)

st.plotly_chart(fig_trend, use_container_width=True)

trend_change = yearly[temp_col].iloc[-1] - yearly[temp_col].iloc[0]

if trend_change > 0:
    st.success("📈 Long term warming trend detected")
else:
    st.info("📉 Cooling trend detected")


# -----------------------------------------------------
# SEASONAL HEATMAP
# -----------------------------------------------------

st.subheader("Seasonal Temperature Pattern")

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

heat = df.pivot_table(
    values=temp_col,
    index="year",
    columns="month",
    aggfunc="mean"
)

fig_heat = px.imshow(
    heat,
    color_continuous_scale="RdYlBu_r",
    aspect="auto"
)

st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------------------------------
# EXTREME ALERT
# -----------------------------------------------------

st.subheader("Extreme Temperature Alert")

high = df[temp_col].quantile(0.95)
low = df[temp_col].quantile(0.05)

latest = df[temp_col].iloc[-1]

if latest > high:
    st.error("🔥 Extreme Heat Alert")

elif latest < low:
    st.warning("❄ Extreme Cold Alert")

else:
    st.success("Temperature Normal")


# -----------------------------------------------------
# HISTORICAL CHART
# -----------------------------------------------------

st.subheader("Historical Temperature")

fig_hist = go.Figure()

fig_hist.add_trace(
    go.Scatter(
        x=df["date"],
        y=df[temp_col],
        mode="lines",
        name="Temperature"
    )
)

st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------

def create_features(data, column):

    data = data.copy()

    data["lag1"] = data[column].shift(1)
    data["lag2"] = data[column].shift(2)
    data["lag7"] = data[column].shift(7)

    data["rolling7"] = data[column].rolling(7).mean()

    data["dayofyear"] = data["date"].dt.dayofyear
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek

    return data


# -----------------------------------------------------
# FORECAST FUNCTION
# -----------------------------------------------------

def forecast_temperature(model, df, column, days):

    df_copy = df.copy()

    preds = []

    for i in range(days):

        df_copy = create_features(df_copy,column)

        features = [
            "lag1",
            "lag2",
            "lag7",
            "rolling7",
            "dayofyear",
            "month",
            "dayofweek"
        ]

        last_row = df_copy.iloc[-1:]

        X = last_row[features]

        X = X.fillna(method="bfill")

        pred = model.predict(X)[0]

        next_date = df_copy["date"].max() + pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "date":[next_date],
            column:[pred]
        })

        df_copy = pd.concat([df_copy,new_row],ignore_index=True)

        preds.append(pred)

    return preds


# -----------------------------------------------------
# GENERATE FORECAST
# -----------------------------------------------------

if st.sidebar.button("Generate Forecast"):

    preds = forecast_temperature(
        model,
        df[["date",temp_col]],
        temp_col,
        forecast_days
    )

    future_dates = pd.date_range(
        df["date"].max(),
        periods=forecast_days+1
    )[1:]

    forecast_df = pd.DataFrame({
        "Date":future_dates,
        "Forecast (°C)":np.round(preds,2)
    })

    st.subheader("Temperature Forecast")

    history = df.tail(60)

    fig_fore = go.Figure()

    fig_fore.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[temp_col],
            mode="lines",
            name="Recent Temperature"
        )
    )

    fig_fore.add_trace(
        go.Scatter(
            x=future_dates,
            y=preds,
            mode="lines+markers",
            name="Forecast"
        )
    )

    st.plotly_chart(fig_fore, use_container_width=True)

    st.subheader("Forecast Table")

    st.dataframe(forecast_df)

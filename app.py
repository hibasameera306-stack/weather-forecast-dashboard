import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Weather Forecast Dashboard", layout="wide")

st.title("🌤 Weather Forecast Analytics Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    file_path = "4245930.csv"

    if not os.path.exists(file_path):
        st.error("Dataset file '4245930.csv' not found in repository.")
        st.stop()

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    # Convert Fahrenheit to Celsius
    df["tmax"] = (df["tmax"] - 32) * 5/9
    df["tmin"] = (df["tmin"] - 32) * 5/9

    df["tmax"] = df["tmax"].interpolate().bfill().ffill()
    df["tmin"] = df["tmin"].interpolate().bfill().ffill()

    return df


df = load_data()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_models():

    if not os.path.exists("tmax_model.pkl"):
        st.error("tmax_model.pkl not found")
        st.stop()

    if not os.path.exists("tmin_model.pkl"):
        st.error("tmin_model.pkl not found")
        st.stop()

    tmax_model = joblib.load("tmax_model.pkl")
    tmin_model = joblib.load("tmin_model.pkl")

    return tmax_model, tmin_model


tmax_model, tmin_model = load_models()

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------

st.sidebar.header("Forecast Settings")

temp_type = st.sidebar.selectbox(
    "Temperature Type",
    ["TMAX", "TMIN"]
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

# -------------------------------------------------
# SUMMARY METRICS
# -------------------------------------------------

st.subheader("Temperature Summary")

col1, col2, col3 = st.columns(3)

latest_temp = df[temp_col].iloc[-1]
avg_temp = df[temp_col].mean()
max_temp = df[temp_col].max()

col1.metric("Latest", f"{latest_temp:.1f} °C")
col2.metric("Average", f"{avg_temp:.1f} °C")
col3.metric("Max", f"{max_temp:.1f} °C")

# -------------------------------------------------
# EXTREME ALERT
# -------------------------------------------------

st.subheader("Extreme Temperature Alert")

high = df[temp_col].quantile(0.95)
low = df[temp_col].quantile(0.05)

if latest_temp > high:
    st.error("🔥 Extreme Heat Alert")

elif latest_temp < low:
    st.warning("❄ Extreme Cold Alert")

else:
    st.success("Temperature Normal")

# -------------------------------------------------
# HISTORICAL CHART
# -------------------------------------------------

st.subheader("Historical Temperature")

fig_hist = go.Figure()

fig_hist.add_trace(
    go.Scatter(
        x=df["date"],
        y=df[temp_col],
        mode="lines",
        name="Temperature (°C)"
    )
)

st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------
# FEATURE CREATION
# -------------------------------------------------

def create_features(data, column):

    data = data.copy()

    data["lag1"] = data[column].shift(1)
    data["lag2"] = data[column].shift(2)
    data["lag7"] = data[column].shift(7)
    data["lag30"] = data[column].shift(30)

    data["rolling7"] = data[column].rolling(7).mean()
    data["rolling30"] = data[column].rolling(30).mean()

    data["dayofyear"] = data["date"].dt.dayofyear
    data["dayofweek"] = data["date"].dt.dayofweek

    return data

# -------------------------------------------------
# FORECAST FUNCTION
# -------------------------------------------------

def forecast_temperature(model, df, column, days):

    df_copy = df.copy()

    preds = []

    for i in range(days):

        df_copy = create_features(df_copy, column)

        last_row = df_copy.iloc[-1:]

        features = [
            "lag1","lag2","lag7","lag30",
            "rolling7","rolling30",
            "dayofyear","dayofweek"
        ]

        X = last_row[features]

        X = X.fillna(method="bfill")

        pred_f = model.predict(X)[0]

        # convert prediction to Celsius
        pred_c = (pred_f - 32) * 5/9

        next_date = df_copy["date"].max() + pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "date":[next_date],
            column:[pred_c]
        })

        df_copy = pd.concat([df_copy,new_row],ignore_index=True)

        preds.append(pred_c)

    return preds

# -------------------------------------------------
# FORECAST BUTTON
# -------------------------------------------------

if st.sidebar.button("Generate Forecast"):

    preds = forecast_temperature(model, df[["date",temp_col]], temp_col, forecast_days)

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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[temp_col],
            mode="lines",
            name="Recent Temperature (°C)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=preds,
            mode="lines+markers",
            name="Forecast (°C)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")

    st.dataframe(forecast_df)

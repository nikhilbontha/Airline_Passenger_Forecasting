import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background:#0F172A;
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* Sidebar width and style */
[data-testid="stSidebar"] {
    width: 25% !important;   /* Sidebar = 1/4 screen */
    min-width: 250px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.2);
    padding: 20px;
}

/* Hero Section */
.hero {
    background-image: url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05");
    background-size: cover;
    background-position: center;
    height: 360px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero-overlay {
    text-align: center;
    background: rgba(0,0,0,0.55);
    padding: 30px 40px;
    border-radius: 12px;
}
.hero h1 {
    font-size: 46px;
    font-weight: 700;
    margin: 0;
    color: #FFFFFF;
}
.hero p {
    font-size: 20px;
    margin-top: 12px;
    color: #38BDF8;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 22px;
}
[data-testid="stMetricLabel"] {
    color: #60A5FA !important;
    font-weight: 600;
    font-size: 16px;
}

/* DataFrame styling */
.stDataFrame {
    background: #1E293B;
    color: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.4);
}
.stDataFrame th {
    background: #2563EB;
    color: #FFFFFF;
    font-weight: 600;
}
.stDataFrame td {
    color: #FFFFFF;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#2563EB,#0EA5E9);
    color: white;
    border: none;
    height: 50px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("assets/logo.png", width=120)
    st.title("⚙ Settings")
    future_months = st.slider("Forecast Months", 1, 24, 12)
    st.markdown("---")
    st.success("RNN Forecast Model")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
loader = DataLoader("data/airline_passengers.csv")
df = loader.load_data()

# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-overlay">
    <h1>✈ Airline Passenger Forecasting</h1>
    <p>Predict Future Passenger Demand using Deep Learning (RNN)</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
mae, mse, rmse = Evaluator().evaluate()
m1, m2, m3 = st.columns(3)
m1.metric("MAE", round(mae,2))
m2.metric("MSE", round(mse,2))
m3.metric("RMSE", round(rmse,2))

st.write("")

# ---------------------------------------------------------
# DATA + TREND
# ---------------------------------------------------------
left, right = st.columns([1,2])
with left:
    st.subheader("Passenger Dataset")
    st.dataframe(df, height=400)
with right:
    st.subheader("Historical Passenger Trend")
    fig = px.line(df, x=df.index, y="Passengers", markers=True)
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# FORECAST
# ---------------------------------------------------------
st.header("Future Passenger Forecast")
if st.button("Generate Forecast"):
    with st.spinner("Running RNN Model..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)
        future_dates = pd.date_range(
            start=df.index[-1] + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )
        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Forecast Values")
        st.dataframe(forecast_df)
        csv = forecast_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Passengers"], name="Historical"))
        fig.add_trace(go.Scatter(
            x=forecast_df["Month"],
            y=forecast_df["Predicted Passengers"],
            name="Forecast",
            line=dict(color="red", dash="dash")
        ))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

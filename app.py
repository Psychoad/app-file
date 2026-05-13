import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Superstore Sales Dashboard", layout="wide", page_icon="🛒")

st.title("🛒 Superstore Sales Forecasting Dashboard")
st.markdown("### Time Series Analysis using ARIMA, SARIMA, Prophet & LSTM")

# Load Data
@st.cache_data
def load_data():
    daily = pd.read_csv("daily_sales.csv", parse_dates=['Date'])
    forecast = pd.read_csv("forecast_comparison.csv", parse_dates=['Date'])
    category = pd.read_csv("category_sales.csv")
    region = pd.read_csv("region_sales.csv")
    return daily, forecast, category, region

daily_sales, forecast, cat_sales, reg_sales = load_data()

# Sidebar
st.sidebar.header("Filters")
if st.sidebar.checkbox("Show Historical Data"):
    st.sidebar.write(daily_sales.head())

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis", "🔮 Forecast Comparison", "📈 Insights"])

with tab1:
    st.subheader("Daily Sales Trend (2015 - 2018)")
    fig1 = px.line(daily_sales, x='Date', y='Sales', title="Historical Daily Sales")
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales by Category")
        fig_cat = px.pie(cat_sales, names='Category', values='Sales', title="Category-wise Sales")
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Region")
        fig_reg = px.bar(reg_sales, x='Region', y='Sales', title="Region-wise Sales")
        st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    st.subheader("7-Day Future Sales Forecast")
    st.dataframe(forecast.style.format({
        "SARIMA": "${:,.0f}",
        "Prophet": "${:,.0f}",
        "ARIMA": "${:,.0f}",
        "LSTM": "${:,.0f}"
    }), use_container_width=True)

    st.subheader("Model Comparison Chart")
    fig_forecast = go.Figure()
    models = ['SARIMA', 'Prophet', 'ARIMA', 'LSTM']
    colors = ['blue', 'green', 'red', 'orange']
    
    for model, color in zip(models, colors):
        fig_forecast.add_trace(go.Scatter(
            x=forecast['Date'], 
            y=forecast[model],
            name=model,
            mode='lines+markers',
            line=dict(color=color)
        ))
    
    fig_forecast.update_layout(title="Forecast Comparison", xaxis_title="Date", yaxis_title="Predicted Sales")
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${daily_sales['Sales'].sum():,.0f}")
    with col2:
        st.metric("Average Daily Sales", f"${daily_sales['Sales'].mean():,.0f}")
    with col3:
        st.metric("Forecast Period", "7 Days (Jan 2019)")

    st.success("**Recommendation**: SARIMA model performed best for this dataset.")
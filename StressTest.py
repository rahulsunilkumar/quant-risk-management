import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Portfolio Stress Testing and Monte Carlo Simulations", layout="wide")

st.title('Portfolio Stress Testing and Monte Carlo Simulations')

st.write("""
This application showcases a comprehensive risk analysis for a portfolio consisting of SPY, BND, GLD, QQQ, and VTI.
We use **Stress Testing** to analyze portfolio performance under extreme market conditions and **Monte Carlo Simulations**
to predict potential future behavior under varying scenarios.

These analyses help assess portfolio resilience and potential returns over different time horizons.
""")

# Sidebar Inputs
st.sidebar.header('Portfolio Parameters')
all_symbols = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']
symbols = st.sidebar.multiselect('Select Tickers for Analysis', options=all_symbols, default=all_symbols)
investment_horizon = st.sidebar.slider('Investment Horizon (Years)', min_value=1, max_value=5, value=2)
num_simulations = st.sidebar.slider('Number of Monte Carlo Simulations', min_value=100, max_value=10000, value=1000)
stress_scenario = st.sidebar.selectbox('Select Stress Scenario', ['2008 Financial Crisis', 'COVID-19 Market Crash', 'Custom Scenario'])

# Load Stock Data
data = yf.download(symbols, period=f"{investment_horizon}y")['Adj Close'].dropna()
log_returns = np.log(data / data.shift(1)).dropna()

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["Stress Testing", "Monte Carlo Simulations", "Details"])

# Tab 1: Stress Testing
with tab1:
    st.subheader(f'Stress Testing - {stress_scenario}')
    # Simulate a historical drawdown scenario (2008 or COVID-19)
    if stress_scenario == '2008 Financial Crisis':
        stress_factor = -0.5  # Assume a 50% drop
    elif stress_scenario == 'COVID-19 Market Crash':
        stress_factor = -0.3  # Assume a 30% drop
    else:
        stress_factor = st.sidebar.slider('Custom Stress Factor (%)', min_value=-100, max_value=0, value=-20) / 100

    stressed_returns = log_returns + stress_factor
    stressed_portfolio_value = (stressed_returns + 1).cumprod()

    st.line_chart(stressed_portfolio_value, use_container_width=True)

    st.subheader('Correlation Heatmap During Stress')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(stressed_returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Tab 2: Monte Carlo Simulations
with tab2:
    st.subheader('Monte Carlo Simulation Results')
    np.random.seed(42)
    sim_results = []
    for _ in range(num_simulations):
        simulated_returns = log_returns.sample(n=investment_horizon * 252, replace=True).mean() * 252
        sim_results.append(simulated_returns)
    sim_results = pd.DataFrame(sim_results)

    st.write("### Projected Portfolio Paths")
    st.line_chart(sim_results.cumsum(), use_container_width=True)

    st.write("### Probability Distribution of Returns")
    fig, ax = plt.subplots()
    sns.histplot(sim_results.sum(), kde=True, ax=ax)
    ax.set_title('Distribution of Simulated Portfolio Returns')
    st.pyplot(fig)

# Tab 3: Details
with tab3:
    st.subheader('Detailed Metrics Summary')
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.write("### Historical Adjusted Close Prices")
        data.index = data.index.date
        st.dataframe(data.tail(), height=300)

        st.write("### Daily Log Returns")
        log_returns.index = log_returns.index.date
        st.dataframe(log_returns.tail(), height=300)

    with col2:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

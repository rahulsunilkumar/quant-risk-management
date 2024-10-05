import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Portfolio Stress Testing, Backtesting, and Monte Carlo Simulations", layout="wide")

st.title('Portfolio Stress Testing, Backtesting, and Monte Carlo Simulations')

st.write("""
This application provides a comprehensive risk analysis for a portfolio consisting of SPY, BND, GLD, QQQ, and VTI.

We employ three powerful tools to assess the resilience and expected performance of the portfolio:

1. **Stress Testing** - Analyzes portfolio performance under extreme market conditions.
2. **Monte Carlo Simulations** - Predicts potential future behaviors under different scenarios.
3. **Backtesting** - Evaluates how the portfolio would have performed historically based on past data.

These tools work together to provide a robust understanding of both the resilience and probabilistic outcomes of the portfolio under different market conditions. The combination of these three approaches offers a synergistic view of risk management and potential returns.
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

# Stress Testing
st.header(f'Stress Testing - {stress_scenario}')
# Simulate a historical drawdown scenario (2008 or COVID-19)
if stress_scenario == '2008 Financial Crisis':
    stress_factor = -0.5  # Assume a 50% drop
elif stress_scenario == 'COVID-19 Market Crash':
    stress_factor = -0.3  # Assume a 30% drop
else:
    stress_factor = st.sidebar.slider('Custom Stress Factor (%)', min_value=-100, max_value=0, value=-20) / 100

stressed_returns = log_returns + stress_factor
stressed_portfolio_value = (stressed_returns + 1).cumprod()

st.write("### Stressed Portfolio Performance")
st.line_chart(stressed_portfolio_value, use_container_width=True)

# Monte Carlo Simulations
st.header('Monte Carlo Simulation Results')
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

# Backtesting
st.header('Backtesting Portfolio Performance')
cumulative_returns = (log_returns + 1).cumprod()

st.write("### Historical Portfolio Performance")
st.line_chart(cumulative_returns, use_container_width=True)

# Summary of Synergy
st.header('Synergy Between the Analyses')
st.write("""
- **Stress Testing** allows us to understand how the portfolio would react in extreme conditions, giving insights into its resilience.
- **Monte Carlo Simulations** provide a probabilistic view of future outcomes, helping to quantify the range of potential returns and risks.
- **Backtesting** shows how the portfolio would have performed in the past, validating the strategy against historical data.

Together, these approaches offer a complete view of the portfolio's risk and return profile, supporting informed decision-making.
""")

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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

# Portfolio Metrics
def calculate_portfolio_metrics(log_returns, weights):
    portfolio_return = np.dot(weights, log_returns.mean()) * 252  # Annualized Return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))  # Annualized Volatility
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Optimize Portfolio
def portfolio_optimization(log_returns):
    n_assets = len(log_returns.columns)
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    def objective(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_volatility

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

optimal_weights = portfolio_optimization(log_returns)
weights_df = pd.DataFrame({'Ticker': symbols, 'Optimal Weight': optimal_weights})
portfolio_return, portfolio_std_dev, sharpe_ratio = calculate_portfolio_metrics(log_returns, optimal_weights)

# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Stress Testing", "Monte Carlo Simulations", "Insights & Summary"])

# Tab 1: Portfolio Overview
with tab1:
    st.subheader('Portfolio Overview')
    fig = go.Figure(data=[go.Pie(labels=weights_df['Ticker'], values=weights_df['Optimal Weight'], hole=.3)])
    fig.update_layout(title_text="Optimal Portfolio Allocation", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Key Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Expected Annual Return", value=f"{portfolio_return:.2%}")
    col2.metric(label="Portfolio Volatility (Risk)", value=f"{portfolio_std_dev:.2%}")
    col3.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")

# Tab 2: Stress Testing
with tab2:
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

# Tab 3: Monte Carlo Simulations
with tab3:
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

# Tab 4: Insights & Summary
with tab4:
    st.subheader('Dynamic Insights & Summary')
    avg_stress_impact = stressed_portfolio_value.iloc[-1].mean() - 1
    avg_monte_carlo_return = sim_results.sum().mean()

    st.write("""
    ### Key Insights
    - **Stress Test Impact**: Under the selected stress scenario, the average portfolio value change is **{avg_stress_impact:.2%}**.
    - **Monte Carlo Average Return**: Based on the simulations, the average return is projected to be **{avg_monte_carlo_return:.2%}** over the selected investment horizon.

    These insights suggest that under adverse conditions, certain assets are more resilient, which highlights the importance of diversification. The Monte Carlo simulations give a broad sense of potential returns, allowing investors to understand both upside and downside risks.
    """)

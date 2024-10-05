import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization with Advanced Metrics", layout="wide")

st.title('Portfolio Optimization Using Machine Learning and Advanced Metrics')

st.write("""
This application demonstrates how machine learning and advanced financial metrics can be used to optimize a portfolio of financial assets. By estimating future returns and applying optimization techniques, we aim to construct an optimal portfolio that balances return and risk.

Key metrics like expected return, standard deviation, and Sharpe ratio are used to assess the optimal allocation.
""")

# Sidebar Inputs
st.sidebar.header('Portfolio Optimization Parameters')
all_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'BABA', 'TSM', 'JPM', 'V', 'SPY', 'BND', 'GLD', 'QQQ', 'VTI']
symbols = st.sidebar.multiselect('Select Tickers for Optimization', options=all_symbols, default=['SPY', 'BND', 'GLD', 'QQQ', 'VTI'])
investment_horizon = st.sidebar.slider('Investment Horizon (Days)', min_value=30, max_value=365, value=180)
risk_tolerance = st.sidebar.slider('Risk Tolerance (0 - Low, 1 - High)', min_value=0.0, max_value=1.0, value=0.5)

# Load Stock Data
data = yf.download(symbols, period="2y")['Adj Close'].dropna()
log_returns = np.log(data / data.shift(1)).dropna()
risk_free_rate = 0.01

# Portfolio Optimization Metrics
def portfolio_optimization(log_returns, risk_tolerance, risk_free_rate):
    n_assets = len(log_returns.columns)
    mean_returns = log_returns.mean() * 252  # Annualize returns
    cov_matrix = log_returns.cov() * 252     # Annualize covariance

    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -((1 - risk_tolerance) * portfolio_return - risk_tolerance * portfolio_volatility)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return optimal_weights, portfolio_return, portfolio_volatility, sharpe_ratio

optimal_weights, expected_return, portfolio_std_dev, sharpe_ratio = portfolio_optimization(log_returns, risk_tolerance, risk_free_rate)
weights_df = pd.DataFrame({'Ticker': symbols, 'Optimal Weight': optimal_weights})

# Tabs for Navigation
tab1, tab2 = st.tabs(["Optimal Portfolio", "Metrics Summary"])

with tab1:
    st.subheader('Optimal Portfolio Allocation')
    fig = go.Figure(data=[go.Pie(labels=weights_df['Ticker'], values=weights_df['Optimal Weight'], hole=.3)])
    if len(weights_df[weights_df['Optimal Weight'] > 0]) == 1:
        fig.update_traces(showlegend=False)
    fig.update_layout(title_text="Optimal Portfolio Allocation", annotations=[dict(text='100%', x=0.5, y=0.5, font_size=20, showarrow=False)] if len(weights_df[weights_df['Optimal Weight'] > 0]) == 1 else None)
    st.plotly_chart(fig)
    st.write(weights_df)

    st.subheader('Portfolio Metrics')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(label="Expected Annual Return", value=f"{expected_return:.2%}")
        st.metric(label="Portfolio Standard Deviation (Risk)", value=f"{portfolio_std_dev:.2%}")
    with col2:
        st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")

with tab2:
    st.subheader('Detailed Metrics Summary')
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.write("### Historical Adjusted Close Prices")
        st.dataframe(data.tail(), height=300)

        st.write("### Daily Log Returns")
        st.dataframe(log_returns.tail(), height=300)

    with col2:
        st.write("### Portfolio Allocation Summary")
        st.write(weights_df)

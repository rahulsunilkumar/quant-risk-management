import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization with Advanced Metrics", layout="wide")

st.title('Portfolio Optimization Using Machine Learning and Advanced Metrics')

st.write("""
This application demonstrates how machine learning and advanced financial metrics can be used to optimize a portfolio of financial assets. By estimating future returns and applying optimization techniques, we aim to construct an optimal portfolio that balances return and risk.

Key metrics like expected return, standard deviation, and Sharpe ratio are used to assess the optimal allocation, with risk-free rate fetched from the Federal Reserve Economic Data (FRED) API.
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

# Get Risk-Free Rate from FRED API
st.sidebar.header('Risk-Free Rate')
fred_api_key = "YOUR_FRED_API_KEY"
risk_free_rate = 0.01  # Default value
try:
    response = requests.get(f"https://api.stlouisfed.org/fred/series/observations", params={
        "series_id": "DGS10",
        "api_key": fred_api_key,
        "file_type": "json",
        "frequency": "d"
    })
    if response.status_code == 200:
        data_json = response.json()
        latest_rate = float(data_json['observations'][-1]['value'])
        risk_free_rate = latest_rate / 100  # Convert to decimal
        st.sidebar.text(f"Latest Risk-Free Rate: {risk_free_rate:.2%}")
    else:
        st.sidebar.text("Unable to fetch risk-free rate, using default 1%.")
except Exception as e:
    st.sidebar.text("Error fetching risk-free rate, using default 1%.")

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

# Displaying Results
st.subheader('Optimal Portfolio Results')
fig = go.Figure(data=[go.Pie(labels=weights_df['Ticker'], values=weights_df['Optimal Weight'], hole=.3)])
fig.update_layout(title_text="Optimal Portfolio Allocation")
st.plotly_chart(fig)
st.write(weights_df)

# Portfolio Metrics
st.subheader('Portfolio Metrics')
st.write(f"**Expected Annual Return**: {expected_return:.2%}")
st.write(f"**Portfolio Standard Deviation (Risk)**: {portfolio_std_dev:.2%}")
st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")

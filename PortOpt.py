import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import quantstats as qs

st.set_page_config(page_title="Portfolio Optimization with Advanced Metrics", layout="wide")

st.title('Portfolio Optimization Using Machine Learning and Advanced Metrics')

st.write("""
This application demonstrates how machine learning and advanced financial metrics can be used to optimize a portfolio of financial assets. By estimating future returns and applying optimization techniques, we aim to construct an optimal portfolio that balances return and risk.

Key metrics like expected return, standard deviation, Sharpe ratio, and other advanced metrics are used to assess the optimal allocation.
""")

# Sidebar Inputs
st.sidebar.header('Portfolio Optimization Parameters')
all_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'BABA', 'TSM', 'JPM', 'V', 'SPY', 'BND', 'GLD', 'QQQ', 'VTI']
symbols = st.sidebar.multiselect('Select Tickers for Optimization', options=all_symbols, default=['SPY', 'BND', 'GLD', 'QQQ', 'VTI'])
investment_horizon = st.sidebar.slider('Investment Horizon (Years)', min_value=1, max_value=5, value=2)
risk_tolerance = st.sidebar.slider('Risk Tolerance (0 - Low, 1 - High)', min_value=0.0, max_value=1.0, value=0.5)

# Load Stock Data
data = yf.download(symbols, period=f"{investment_horizon}y")['Adj Close'].dropna()
log_returns = np.log(data / data.shift(1)).dropna()
risk_free_rate = 0.01

# Portfolio Optimization Metrics
def portfolio_optimization(log_returns, risk_tolerance, risk_free_rate):
    n_assets = len(log_returns.columns)
    mean_returns = log_returns.mean() * 12  # Annualize returns based on monthly data
    cov_matrix = log_returns.cov() * 12  # Annualize covariance based on monthly data

    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns) * investment_horizon
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(investment_horizon)
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

# Portfolio Performance Metrics using QuantStats
portfolio_returns = log_returns.dot(optimal_weights)
qs.extend_pandas()
metrics = {
    'CAGR': portfolio_returns.cagr(),
    'Max Drawdown': portfolio_returns.max_drawdown(),
    'Volatility': portfolio_returns.volatility(),
    'Sharpe Ratio': portfolio_returns.sharpe(rf=risk_free_rate),
    'Sortino Ratio': portfolio_returns.sortino(rf=risk_free_rate),
    'Calmar Ratio': portfolio_returns.calmar(),
        'Skewness': portfolio_returns.skew(),
    'Kurtosis': portfolio_returns.kurtosis(),
    'Value at Risk (VaR)': portfolio_returns.value_at_risk()
}

# Load Benchmark Data
benchmark_symbol = 'SPY'
benchmark_data = yf.download(benchmark_symbol, period=f"{investment_horizon}y")['Adj Close'].dropna()
benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()

# Tabs for Navigation
tab1, tab2 = st.tabs(["Optimal Portfolio", "Details"])

with tab1:
    st.subheader('Benchmark Comparison')
    cumulative_portfolio_returns = (portfolio_returns + 1).cumprod()
    cumulative_portfolio_returns.index = cumulative_portfolio_returns.index.tz_localize(None)
    cumulative_benchmark_returns = (benchmark_returns + 1).cumprod()
    cumulative_benchmark_returns.index = cumulative_benchmark_returns.index.tz_localize(None)
    comparison_df = pd.DataFrame({"Portfolio": cumulative_portfolio_returns, "Benchmark (SPY)": cumulative_benchmark_returns})
    st.line_chart(comparison_df, use_container_width=True)

    st.subheader('Optimal Portfolio Allocation')
    fig = go.Figure(data=[go.Pie(labels=weights_df['Ticker'], values=weights_df['Optimal Weight'], hole=.3)])
    if len(weights_df[weights_df['Optimal Weight'] > 0]) == 1:
        fig.update_traces(showlegend=False)
    fig.update_layout(title_text="Optimal Portfolio Allocation", annotations=[dict(text='100%', x=0.5, y=0.5, font_size=20, showarrow=False)], template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Risk-Return Trade-off')
    fig_tradeoff = go.Figure()
    for ticker in symbols:
        asset_return = log_returns[ticker].mean() * 12
        asset_volatility = log_returns[ticker].std() * np.sqrt(12)
        fig_tradeoff.add_trace(go.Scatter(x=[asset_volatility], y=[asset_return], mode='markers', name=ticker))
    portfolio_volatility = portfolio_std_dev
    portfolio_return = expected_return
    fig_tradeoff.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return], mode='markers', name='Optimized Portfolio', marker=dict(color='red', size=10)))
    fig_tradeoff.update_layout(title='Risk-Return Trade-off', xaxis_title='Volatility (Risk)', yaxis_title='Expected Return', template='plotly_dark')
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    st.subheader('Portfolio Metrics')
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric(label="Expected Annual Return", value=f"{expected_return:.2%}")
        st.metric(label="Portfolio Standard Deviation (Risk)", value=f"{portfolio_std_dev:.2%}")
        st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
    with col2:
        st.metric(label="CAGR", value=f"{metrics['CAGR']:.2%}")
        st.metric(label="Max Drawdown", value=f"{metrics['Max Drawdown']:.2%}")
        st.metric(label="Volatility", value=f"{metrics['Volatility']:.2%}")
        st.metric(label="Sortino Ratio", value=f"{metrics['Sortino Ratio']:.2f}")
    with col3:
        st.metric(label="Calmar Ratio", value=f"{metrics['Calmar Ratio']:.2f}")
        st.metric(label="Skewness", value=f"{metrics['Skewness']:.2f}")
        st.metric(label="Kurtosis", value=f"{metrics['Kurtosis']:.2f}")
        st.metric(label="Value at Risk (VaR)", value=f"{metrics['Value at Risk (VaR)']:.2f}")

    
with tab2:
    st.subheader('Detailed Metrics Summary')
    st.subheader('Correlation Heatmap')
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.write("### Historical Adjusted Close Prices")
        data.index = data.index.date
st.dataframe(data.tail(), height=300)

        st.write("### Daily Log Returns")
        log_returns.index = log_returns.index.date
st.dataframe(log_returns.tail(), height=300)

    with col2:
        st.write("### Portfolio Allocation Summary")
        st.write(weights_df)

        st.write("### Additional Portfolio Metrics")
        for metric, value in metrics.items():
            st.write(f"**{metric}**: {value:.2f}")

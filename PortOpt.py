import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization with ML", layout="wide")

st.title('Portfolio Optimization Using Machine Learning')

st.write("""
This application demonstrates how machine learning can be used to optimize a portfolio of financial assets. By using predictive models to estimate future returns and advanced optimization techniques, we can construct an optimal portfolio that balances return and risk.

The goal of this project is to show how machine learning can be leveraged in finance to make data-driven decisions for portfolio management.
""")

# Sidebar Inputs
st.sidebar.header('Portfolio Optimization Parameters')
all_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'BABA', 'TSM', 'JPM', 'V']
symbols = st.sidebar.multiselect('Select Tickers for Optimization', options=all_symbols, default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'])
investment_horizon = st.sidebar.slider('Investment Horizon (Days)', min_value=30, max_value=365, value=180)
risk_tolerance = st.sidebar.slider('Risk Tolerance (0 - Low, 1 - High)', min_value=0.0, max_value=1.0, value=0.5)

# Load Stock Data
data = yf.download(symbols, period="2y")['Adj Close'].dropna()
returns = data.pct_change().dropna()
train_data, test_data = train_test_split(returns, test_size=0.2, shuffle=False)

# Machine Learning Model for Return Prediction
predictions = pd.DataFrame(index=test_data.index, columns=test_data.columns)
for symbol in symbols:
    model = LinearRegression()
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data[symbol].values
    model.fit(X_train, y_train)
    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    predictions[symbol] = model.predict(X_test)

# Portfolio Optimization
def portfolio_optimization(predicted_returns, risk_tolerance):
    n_assets = len(predicted_returns.columns)
    mean_returns = predicted_returns.mean()
    cov_matrix = predicted_returns.cov()

    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -((1 - risk_tolerance) * portfolio_return - risk_tolerance * portfolio_volatility)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

optimal_weights = portfolio_optimization(predictions, risk_tolerance)
weights_df = pd.DataFrame({'Ticker': symbols, 'Optimal Weight': optimal_weights})

# Tabs for Navigation
tab1, tab2 = st.tabs(["Portfolio Allocation", "Predicted Returns"])

with tab1:
    st.subheader('Optimal Portfolio Allocation')
    fig = go.Figure(data=[go.Pie(labels=weights_df['Ticker'], values=weights_df['Optimal Weight'], hole=.3)])
    fig.update_layout(title_text="Optimal Portfolio Allocation")
    st.plotly_chart(fig)
    st.write(weights_df)

with tab2:
    st.subheader('Predicted Returns')
    st.write("### Historical Adjusted Close Prices", data.tail())
    st.write("### Daily Returns", returns.tail())
    st.write("### Predicted Returns", predictions.tail())
    st.write("### Dynamic Summary of Predicted Returns")
    mean_pred_returns = predictions.mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mean_pred_returns.index, y=mean_pred_returns.values, name='Mean Predicted Return'))
    fig.update_layout(title_text='Mean Predicted Returns for Selected Assets', xaxis_title='Ticker', yaxis_title='Mean Predicted Return')
    st.plotly_chart(fig)


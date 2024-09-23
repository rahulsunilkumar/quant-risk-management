import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Title of the app
st.title("Stress Testing and Scenario Analysis")

# Introduction and explanation
st.markdown("""
### Overview:
This project evaluates the resilience of the portfolio by simulating various market stress scenarios. These stress tests help assess how the portfolio would perform during extreme market events, such as a financial crisis or a sudden rise in interest rates.
""")

# Sidebar: User Input for Stress Test
st.sidebar.header('Stress Test Scenarios')
tickers = st.sidebar.multiselect('Select assets for stress testing', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX'], default=['AAPL', 'GOOGL', 'MSFT'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
stress_scenarios = st.sidebar.selectbox("Select Stress Scenario", ('2008 Financial Crisis', 'COVID-19 Crash', 'Custom Scenario'))

# Placeholder: Optimized portfolio weights from Project 2 (replace with actual values)
np.random.seed(42)
num_assets = len(tickers)
optimal_weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()  # Simulated weights
capital = st.sidebar.number_input('Initial Investment ($)', value=10000)

# Function to get stock data
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Load stock data
if len(tickers) > 0:
    stock_prices, returns = get_stock_data(tickers, start_date, end_date)
    st.write(f"Displaying stock price data for selected assets: {tickers}")
    st.line_chart(stock_prices)

    # Calculate portfolio returns under normal conditions
    portfolio_returns_normal = np.dot(returns, optimal_weights)

    # Define stress test scenarios
    st.markdown("### Stress Testing Results")
    
    if stress_scenarios == '2008 Financial Crisis':
        # Assume all assets drop by 40%
        stressed_returns = returns - 0.40
    elif stress_scenarios == 'COVID-19 Crash':
        # Assume all assets drop by 30%
        stressed_returns = returns - 0.30
    elif stress_scenarios == 'Custom Scenario':
        # Custom stress: user-defined percentage drops for each asset
        drop_percentage = st.sidebar.slider("Custom Drop (%):", min_value=0, max_value=50, value=10) / 100
        stressed_returns = returns - drop_percentage

    # Calculate portfolio returns under stress conditions
    portfolio_returns_stressed = np.dot(stressed_returns, optimal_weights)

    # Display portfolio performance under normal vs. stress conditions
    st.write(f"### Portfolio Value (Initial Investment: ${capital:,.2f})")
    portfolio_value_normal = np.cumprod(1 + portfolio_returns_normal) * capital
    portfolio_value_stressed = np.cumprod(1 + portfolio_returns_stressed) * capital
    
    st.write(f"Final Portfolio Value (Normal): ${portfolio_value_normal[-1]:,.2f}")
    st.write(f"Final Portfolio Value (Stress): ${portfolio_value_stressed[-1]:,.2f}")

    # Plot portfolio value over time (normal vs. stress)
    fig, ax = plt.subplots()
    ax.plot(stock_prices.index[1:], portfolio_value_normal, label="Normal Conditions", color="blue")
    ax.plot(stock_prices.index[1:], portfolio_value_stressed, label=f"{stress_scenarios} Scenario", color="red")
    ax.set_title("Portfolio Value Under Normal vs. Stress Conditions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    st.pyplot(fig)

    # Risk Metrics under Stress Conditions
    st.markdown("### Risk Metrics Under Stress Conditions")

    def calculate_var(returns, alpha=0.05):
        var = np.percentile(returns, 100 * alpha)
        return var

    def calculate_cvar(returns, alpha=0.05):
        var = calculate_var(returns, alpha)
        cvar = np.mean(returns[returns <= var])
        return cvar

    # Calculate VaR and CVaR under stress conditions
    VaR_95_stress = calculate_var(portfolio_returns_stressed, alpha=0.05)
    CVaR_95_stress = calculate_cvar(portfolio_returns_stressed, alpha=0.05)

    st.write(f"95% Value-at-Risk (VaR) under {stress_scenarios}: {VaR_95_stress*100:.2f}%")
    st.write(f"95% Conditional Value-at-Risk (CVaR) under {stress_scenarios}: {CVaR_95_stress*100:.2f}%")

    # Plot distribution of stressed portfolio returns
    fig, ax = plt.subplots()
    ax.hist(portfolio_returns_stressed, bins=50, alpha=0.75, color='blue')
    ax.axvline(VaR_95_stress, color='red', linestyle='--', label=f"VaR (95%) = {VaR_95_stress:.4f}")
    ax.axvline(CVaR_95_stress, color='green', linestyle='--', label=f"CVaR (95%) = {CVaR_95_stress:.4f}")
    ax.set_title(f"Distribution of Portfolio Returns Under {stress_scenarios}")
    ax.set_xlabel("Portfolio Returns")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

else:
    st.write("Please select at least one asset.")

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Title of the app
st.title("Portfolio Optimization with Machine Learning and Factor Models")

# Introduction and explanation
st.markdown("""
### Overview:
This project focuses on optimizing a portfolio based on predicted returns (from Project 1) and using classical mean-variance optimization. We also incorporate factor models and advanced risk metrics like VaR and CVaR.
""")

# Sidebar: User Input for Predicted Returns and Constraints
st.sidebar.header('User Input Parameters')
num_assets = st.sidebar.slider("Select the number of assets in your portfolio", min_value=2, max_value=10, value=4)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=0.01) / 100
max_allocation = st.sidebar.slider("Maximum allocation per asset (%)", min_value=0, max_value=100, value=50) / 100
min_allocation = st.sidebar.slider("Minimum allocation per asset (%)", min_value=0, max_value=100, value=0) / 100

# Placeholder: Simulated returns or use Project 1 returns (replace with actual predicted returns from ML model)
np.random.seed(42)
predicted_returns = np.random.randn(252, num_assets) / 100  # Simulated predicted returns for 252 trading days (1 year)
mean_returns = np.mean(predicted_returns, axis=0)
cov_matrix = np.cov(predicted_returns, rowvar=False)

# Mean-Variance Optimization using SciPy's minimize
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

# Objective function: minimize the negative Sharpe ratio
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (returns - risk_free_rate) / risk
    return -sharpe_ratio

# Constraints: weights must sum to 1 and be within user-defined bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))

# Initial guess for weights
init_guess = np.array([1./num_assets] * num_assets)

# Optimize portfolio weights
opt_result = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix, risk_free_rate), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

# Calculate the optimized portfolio performance
opt_returns, opt_risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
sharpe_ratio = (opt_returns - risk_free_rate) / opt_risk

# Display optimal weights and portfolio performance
st.markdown("### Optimized Portfolio Weights and Performance")
st.write(f"Optimal Portfolio Weights: {optimal_weights.round(2)}")
st.write(f"Expected Return: {opt_returns*100:.2f}%")
st.write(f"Risk (Standard Deviation): {opt_risk*100:.2f}%")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot the Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, risk_free_rate):
    frontier_returns = []
    frontier_risks = []
    for r in np.linspace(min(mean_returns), max(mean_returns), 100):
        constraints = [{'type': 'eq', 'fun': lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - r},
                       {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1], init_guess, bounds=bounds, constraints=constraints)
        frontier_returns.append(r)
        frontier_risks.append(result.fun)
    return frontier_returns, frontier_risks

# Generate efficient frontier
frontier_returns, frontier_risks = efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

# Plot efficient frontier and optimal portfolio
st.markdown("### Efficient Frontier")
fig, ax = plt.subplots()
ax.plot(frontier_risks, frontier_returns, 'b--', label="Efficient Frontier")
ax.scatter(opt_risk, opt_returns, color='r', label="Optimal Portfolio", marker='x')
ax.set_xlabel('Risk (Standard Deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
st.pyplot(fig)

# Advanced Risk Metrics (Value-at-Risk and Conditional VaR)
def calculate_var(returns, alpha=0.05):
    var = np.percentile(returns, 100 * alpha)
    return var

def calculate_cvar(returns, alpha=0.05):
    var = calculate_var(returns, alpha)
    cvar = np.mean(returns[returns <= var])
    return cvar

# Calculate Value-at-Risk and Conditional VaR
simulated_portfolio_returns = np.dot(predicted_returns, optimal_weights)
VaR_95 = calculate_var(simulated_portfolio_returns, alpha=0.05)
CVaR_95 = calculate_cvar(simulated_portfolio_returns, alpha=0.05)

# Display VaR and CVaR
st.markdown("### Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)")
st.write(f"95% Value-at-Risk (VaR): {VaR_95*100:.2f}%")
st.write(f"95% Conditional Value-at-Risk (CVaR): {CVaR_95*100:.2f}%")

# Plot histogram of simulated portfolio returns with VaR and CVaR
fig, ax = plt.subplots()
ax.hist(simulated_portfolio_returns, bins=50, alpha=0.75, color='blue')
ax.axvline(VaR_95, color='red', linestyle='--', label=f"VaR (95%) = {VaR_95:.4f}")
ax.axvline(CVaR_95, color='green', linestyle='--', label=f"CVaR (95%) = {CVaR_95:.4f}")
ax.set_title("Distribution of Simulated Portfolio Returns")
ax.set_xlabel("Portfolio Returns")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

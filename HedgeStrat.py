import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Title of the app
st.title("Hedging Strategies Using Derivatives")

# Introduction and explanation
st.markdown("""
### Overview:
This project implements various hedging strategies using derivatives like options and futures to mitigate portfolio risk. You can choose between delta hedging, gamma hedging, and analyze volatility surfaces for different strategies.
""")

# Sidebar: User Input for Option Parameters
st.sidebar.header('Option and Asset Parameters')
asset_price = st.sidebar.number_input('Current Asset Price', value=100.0)
strike_price = st.sidebar.number_input('Strike Price', value=105.0)
time_to_maturity = st.sidebar.number_input('Time to Maturity (Years)', value=1.0)
volatility = st.sidebar.slider('Volatility (%)', min_value=5, max_value=100, value=20) / 100
risk_free_rate = st.sidebar.number_input('Risk-Free Interest Rate (%)', value=1.0) / 100
option_type = st.sidebar.selectbox("Option Type", ('Call', 'Put'))

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return price, delta, gamma, vega

# Calculate option price, delta, gamma, and vega using Black-Scholes
option_price, delta, gamma, vega = black_scholes(asset_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())

# Display the option price and Greeks
st.markdown(f"### {option_type} Option Pricing and Greeks")
st.write(f"**Option Price:** ${option_price:.2f}")
st.write(f"**Delta:** {delta:.4f}")
st.write(f"**Gamma:** {gamma:.4f}")
st.write(f"**Vega:** {vega:.4f}")

# Explanation of Delta Hedging
st.markdown("""
### Delta Hedging:
Delta hedging is a strategy where the portfolio is hedged to become delta-neutral, meaning that small movements in the underlying asset will not affect the portfolio's value.
""")

# Delta Hedge Calculation
num_options = st.sidebar.number_input('Number of Options', value=100)
position_in_asset = -num_options * delta

st.write(f"To delta hedge this portfolio, you would need to hold **{position_in_asset:.2f} units** of the underlying asset to neutralize the delta risk.")

# Explanation of Gamma Hedging
st.markdown("""
### Gamma Hedging:
Gamma hedging goes one step further, aiming to reduce the risk of large movements in the underlying asset by neutralizing both delta and gamma risk. This requires a dynamic adjustment of both the asset and the option positions.
""")

# Gamma Hedging (can be further implemented with multiple options)
# For simplicity, we'll just show how gamma affects the portfolio sensitivity for now.

# Plot Delta vs. Asset Price
st.markdown("### Delta Sensitivity to Asset Price")
asset_prices = np.linspace(asset_price - 50, asset_price + 50, 100)
deltas = [black_scholes(S, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())[1] for S in asset_prices]

fig, ax = plt.subplots()
ax.plot(asset_prices, deltas, label="Delta")
ax.axhline(0, color="red", linestyle="--", label="Delta-Neutral")
ax.set_xlabel("Asset Price")
ax.set_ylabel("Delta")
ax.set_title(f"Delta vs. Asset Price for {option_type} Option")
ax.legend()
st.pyplot(fig)

# Volatility Surface Visualization
st.markdown("""
### Volatility Surface:
Volatility surfaces help us understand how the price of options changes as volatility, time to maturity, and strike prices vary. This is crucial for identifying the right options to hedge different risk exposures.
""")

strike_prices = np.linspace(80, 120, 100)
volatilities = np.linspace(0.05, 0.50, 100)
option_prices_surface = np.zeros((100, 100))

# Generate option prices across different strike prices and volatilities
for i, K in enumerate(strike_prices):
    for j, sigma in enumerate(volatilities):
        option_prices_surface[i, j], _, _, _ = black_scholes(asset_price, K, time_to_maturity, risk_free_rate, sigma, option_type.lower())

# Plot the volatility surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(strike_prices, volatilities)
ax.plot_surface(X, Y, option_prices_surface.T, cmap='viridis')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Volatility')
ax.set_zlabel('Option Price')
ax.set_title(f'Volatility Surface for {option_type} Option')
st.pyplot(fig)


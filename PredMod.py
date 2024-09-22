import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title of the app
st.title("Predictive Modeling of Asset Returns with Machine Learning")

# Introduction and explanation
st.markdown("""
### Overview:
This project uses advanced machine learning techniques to predict the returns of various assets (e.g., stocks, bonds, commodities). The predicted returns can later be used for portfolio optimization and risk management.

Key Steps:
1. Select your assets and date range.
2. Use machine learning models (Gradient Boosting, Random Forest) to predict future returns.
3. Evaluate model performance and visualize actual vs. predicted returns.
""")

# Sidebar: User Input for Asset Selection and Date Range
st.sidebar.header('User Input Parameters')
tickers = st.sidebar.multiselect('Select assets for prediction', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX'], default=['AAPL', 'GOOGL'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
model_choice = st.sidebar.selectbox("Select Machine Learning Model", ('Gradient Boosting', 'Random Forest'))
test_size = st.sidebar.slider("Select Test Data Size (as %)", min_value=10, max_value=50, value=20)

# Function to get stock data
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Load and display stock data
if len(tickers) > 0:
    stock_prices, returns = get_stock_data(tickers, start_date, end_date)
    st.write(f"Displaying stock price data for selected assets: {tickers}")
    st.line_chart(stock_prices)

    # Feature Engineering: Create rolling features (e.g., moving averages)
    for ticker in tickers:
        returns[f'{ticker}_MA10'] = returns[ticker].rolling(window=10).mean()
        returns[f'{ticker}_Volatility'] = returns[ticker].rolling(window=10).std()

    # Drop rows with NaN values after creating rolling features
    returns = returns.dropna()

    # Split the data into train and test sets
    X = returns.drop(columns=tickers)  # Features (rolling averages, volatility)
    y = returns[tickers]  # Target (returns)

    # If predicting for a single asset, flatten y to 1D array
    if len(tickers) == 1:
        y = y.values.ravel()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)

    # Select model based on user input
    if model_choice == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_choice == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Display evaluation metrics
    st.markdown("### Model Performance")
    if len(tickers) == 1:
        # Single asset evaluation
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"**{tickers[0]}**: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    else:
        # Multiple asset evaluation
        for i, ticker in enumerate(tickers):
            rmse = np.sqrt(mean_squared_error(y_test[ticker], y_pred[:, i]))
            r2 = r2_score(y_test[ticker], y_pred[:, i])
            st.write(f"**{ticker}**: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    # Plot actual vs. predicted returns for each asset
    st.markdown("### Actual vs. Predicted Returns")
    if len(tickers) == 1:
        # Single asset plot
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test, label="Actual Returns", color="blue")
        ax.plot(y_test.index, y_pred, label="Predicted Returns", color="red", linestyle="--")
        ax.set_title(f"Actual vs. Predicted Returns for {tickers[0]}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.legend()
        st.pyplot(fig)
    else:
        # Multiple asset plots
        for i, ticker in enumerate(tickers):
            fig, ax = plt.subplots()
            ax.plot(y_test.index, y_test[ticker], label="Actual Returns", color="blue")
            ax.plot(y_test.index, y_pred[:, i], label="Predicted Returns", color="red", linestyle="--")
            ax.set_title(f"Actual vs. Predicted Returns for {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns")
            ax.legend()
            st.pyplot(fig)

    # Show feature importance (only for models that support it)
    if model_choice == 'Random Forest':
        st.markdown("### Feature Importance")
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='green')
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

else:
    st.write("Please select at least one asset.")

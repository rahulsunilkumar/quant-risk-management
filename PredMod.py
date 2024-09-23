import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Title of the app
st.title("Predictive Modeling of Asset Returns with Machine Learning & Sentiment Analysis")

# Introduction and explanation
st.markdown("""
### Overview:
In this project, we use machine learning to predict stock returns, incorporating financial sentiment analysis and macroeconomic data. The model will be evaluated using various metrics and feature importance to showcase the effectiveness of incorporating external data.
""")

# Sidebar: User Input for Asset Selection and Date Range
st.sidebar.header('User Input Parameters')
tickers = st.sidebar.multiselect('Select assets for prediction', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX'], default=['AAPL', 'GOOGL'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
test_size = st.sidebar.slider("Test Size (as %)", min_value=10, max_value=50, value=20)
model_choice = st.sidebar.selectbox("Select Machine Learning Model", ('Random Forest', 'Gradient Boosting'))

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

    # Feature Engineering: Create rolling features (moving averages, volatility)
    for ticker in tickers:
        returns[f'{ticker}_MA10'] = returns[ticker].rolling(window=10).mean()
        returns[f'{ticker}_Volatility'] = returns[ticker].rolling(window=10).std()

    # Drop rows with NaN values after creating rolling features
    returns = returns.dropna()
    
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Check and download the VADER lexicon
    nltk.download('vader_lexicon')
    
    # Sentiment Analysis (VADER)
    def fetch_news_sentiment():
        sid = SentimentIntensityAnalyzer()
        sample_news = [
            "The stock market is booming today.",
            "Interest rates are rising quickly.",
            "The economy is facing headwinds due to high inflation."
        ]
        sentiment_scores = [sid.polarity_scores(news)["compound"] for news in sample_news]
        return pd.Series(sentiment_scores, index=returns.index[:len(sentiment_scores)])

    # Fetch sentiment data
    sentiment = fetch_news_sentiment()
    returns['Sentiment'] = sentiment.fillna(0)
    
    st.line_chart(sentiment)

    # Add Macroeconomic Data (Placeholder)
    st.markdown("### Macroeconomic Data Integration")
    macro_data = pd.DataFrame({
        'Interest_Rate': np.random.randn(len(returns)),
        'GDP_Growth': np.random.randn(len(returns)),
        'Inflation': np.random.randn(len(returns))
    }, index=returns.index)

    # Concatenate features for ML model
    X = pd.concat([returns.drop(columns=tickers), macro_data], axis=1)
    y = returns[tickers]

    # If predicting for a single asset, flatten y
    if len(tickers) == 1:
        y = y.values.ravel()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)

    # Machine Learning Model Selection
    if model_choice == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        st.write("XGBoost/Gradient Boosting implementation goes here.")  # Add XGBoost model setup if necessary

    # Train the model
    model.fit(X_train, y_train)

    # Predict returns
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    st.markdown("### Model Evaluation")
    if len(tickers) == 1:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"**{tickers[0]}**: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    else:
        for i, ticker in enumerate(tickers):
            rmse = np.sqrt(mean_squared_error(y_test[ticker], y_pred[:, i]))
            r2 = r2_score(y_test[ticker], y_pred[:, i])
            st.write(f"**{ticker}**: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    # Feature Importance for Random Forest
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

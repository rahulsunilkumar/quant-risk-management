# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page configuration
st.set_page_config(page_title='Stock Price Prediction', layout='wide')

# Title and description
st.title('📈 Stock Price Prediction using LSTM')
st.markdown("""
This app uses an LSTM neural network to predict stock prices based on historical data.
""")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
    end_date = st.sidebar.date_input('End Date', date.today())
    n_epochs = st.sidebar.slider('Number of Epochs', 1, 50, 5)
    return ticker, start_date, end_date, n_epochs

ticker, start_date, end_date, n_epochs = user_input_features()

# Fetch data from Yahoo Finance
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data_load_state.text('Loading data... Done!')

# Check if data is available
if data.empty:
    st.error('No data available for the selected stock and date range. Please adjust your input.')
    st.stop()

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

plot_raw_data()

# Prepare data for LSTM model
df = data[['Date', 'Close']]
df.set_index('Date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)

# Create training and testing datasets
training_data_len = int(len(scaled_data) * 0.8)

train_data = scaled_data[0:training_data_len]
test_data = scaled_data[training_data_len - 60:]

# Create datasets
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Create training data
X_train, y_train = create_dataset(train_data, 60)
# Create testing data
X_test, y_test = create_dataset(test_data, 60)

# Reshape data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
st.subheader('Model Summary')
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
st.text('\n'.join(model_summary))

# Train the model
st.subheader('Training the Model')
with st.spinner('Training in progress...'):
    history = model.fit(X_train, y_train, batch_size=1, epochs=n_epochs, verbose=0)
st.success('Training completed!')

# Plot training loss
st.subheader('Training Loss')
fig2, ax2 = plt.subplots()
ax2.plot(history.history['loss'], label='Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
st.pyplot(fig2)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Get the actual prices
actual_prices = df.values[training_data_len:]

# Calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual_prices[-len(predictions):], predictions))

# Prepare data for plotting
train = df[:training_data_len]
valid = df[training_data_len:]
valid = valid.copy()
valid['Predictions'] = predictions

# Plot the predictions
def plot_predictions():
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train.index, train['Close'], label='Training Data')
    ax.plot(valid.index, valid['Close'], label='Actual Price')
    ax.plot(valid.index, valid['Predictions'], label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

st.subheader('Actual vs. Predicted Prices')
plot_predictions()

# Show the predicted vs actual values
st.subheader('Comparison Table')
st.write(valid[['Close', 'Predictions']])

# Display RMSE
st.write(f'**Root Mean Squared Error:** {rmse:.2f}')

# Footer
st.markdown("""
---
Developed by [Your Name]
""")

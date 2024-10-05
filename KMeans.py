import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# Load ESG Dataset from Yahoo Finance
# ESG-focused stocks (example selection based on ESG funds and ratings)
symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'ADBE', 'ORCL', 'INTC', 'GOOGL', 'AMZN', 'CRM']

# Download stock data
data = yf.download(symbols, start='2023-01-01', end='2023-12-31')['Adj Close']
data = data.pct_change().dropna()

# Download ESG scores from Yahoo Finance
esg_scores = {}
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    esg_data = ticker.sustainability
    if esg_data is not None and 'totalEsg' in esg_data.columns:
        esg_scores[symbol] = esg_data['totalEsg'].values[0]
    else:
        esg_scores[symbol] = np.nan

# Drop stocks with missing ESG scores
esg_scores = {k: v for k, v in esg_scores.items() if not np.isnan(v)}
symbols = list(esg_scores.keys())
data = data[symbols]

# Add ESG scores to the dataframe
data['esg_score'] = np.mean([esg_scores[symbol] for symbol in symbols])

# Streamlit UI
st.title('ESG Asset Clustering Using Machine Learning')
st.sidebar.header('Clustering Parameters')

# Sidebar Inputs
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4)
features = st.sidebar.multiselect('Features to Include', options=data.columns, default=list(data.columns))

# Filter dataset to selected features
filtered_data = data[features]

# Apply PCA for Dimensionality Reduction (optional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(filtered_data)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_data)

# Add clusters to dataframe
data['Cluster'] = clusters

# Plotting the Clusters
fig, ax = plt.subplots()
for cluster in range(n_clusters):
    cluster_data = reduced_data[clusters == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend()
st.pyplot(fig)

# Cluster Summary
st.header('Cluster Summary')
for cluster in range(n_clusters):
    st.subheader(f'Cluster {cluster}')
    cluster_info = data[data['Cluster'] == cluster]
    st.write(cluster_info.describe())

# Add Insights Section
st.header('Insights')
st.write("Explore how different clusters of ESG assets behave under various market conditions. Adjust the clustering parameters to gain insights into how ESG scores and financial metrics correlate.")

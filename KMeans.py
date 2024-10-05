import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import shap

# Load Stock Dataset from Yahoo Finance
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

# Download stock data
data = yf.download(symbols, start='2023-01-01', end='2023-12-31')['Adj Close']
data = data.pct_change().dropna()

# Streamlit UI
st.title('Asset Clustering Using Machine Learning')
st.sidebar.header('Clustering Parameters')

# Sidebar Inputs
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4)
features = st.sidebar.multiselect('Features to Include', options=data.columns.tolist(), default=data.columns.tolist() if not data.empty else [])
apply_pca = st.sidebar.checkbox('Apply PCA for Dimensionality Reduction', value=True)

# Filter dataset to selected features
filtered_data = data[features]

# Apply PCA for Dimensionality Reduction (optional)
if apply_pca:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(filtered_data)
else:
    reduced_data = filtered_data.values

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

ax.set_xlabel('PCA Component 1' if apply_pca else 'Feature 1')
ax.set_ylabel('PCA Component 2' if apply_pca else 'Feature 2')
ax.legend()
st.pyplot(fig)

# Cluster Summary
st.header('Cluster Summary')
for cluster in range(n_clusters):
    st.subheader(f'Cluster {cluster}')
    cluster_info = data[data['Cluster'] == cluster]
    st.write(cluster_info.describe())

# Feature Importance Visualization using Mean Decrease in Variance
st.header('Feature Importance Visualization')
if not filtered_data.empty:
    feature_importances = np.var(filtered_data, axis=0)
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting Feature Importances
    fig, ax = plt.subplots()
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance (Variance)')
    ax.set_title('Feature Importance in Clustering')
    st.pyplot(fig)

# Add Insights Section
st.header('Insights')
st.write("Explore how different clusters of assets behave under various market conditions. Adjust the clustering parameters to gain insights into how different features correlate.")

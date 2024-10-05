import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# Load Stock Dataset from Yahoo Finance
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

# Download stock data
data = yf.download(symbols, start='2023-01-01', end='2023-12-31')['Adj Close']
data = data.pct_change().dropna()

# Streamlit UI
st.title('Asset Clustering Using Machine Learning')

st.write("""
This application demonstrates how machine learning can be used to cluster financial assets based on their price movements. By selecting different clustering parameters and features, you can explore how similar or different assets behave, providing insights into diversification opportunities and market behavior.

Clustering is a powerful tool for understanding patterns in financial data, and in this app, we leverage k-means clustering combined with optional dimensionality reduction (PCA) to group similar stocks.
""")

st.sidebar.header('Clustering Parameters')

# Sidebar Inputs
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4)
features = st.sidebar.multiselect('Features to Include', options=data.columns.tolist(), default=data.columns.tolist() if not data.empty else [])
apply_pca = st.sidebar.checkbox('Apply PCA for Dimensionality Reduction', value=True)
dimension = st.sidebar.slider('Select Number of Dimensions for Plotting', min_value=2, max_value=3, value=2)

# Filter dataset to selected features
filtered_data = data[features]

# Apply PCA for Dimensionality Reduction (optional)
if apply_pca:
    pca = PCA(n_components=dimension)
    reduced_data = pca.fit_transform(filtered_data)
else:
    reduced_data = filtered_data.values

# Apply KMeans Clustering
if len(filtered_data) < n_clusters:
    st.error("Number of clusters cannot be greater than the number of data points. Please adjust the number of clusters.")
else:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # Add clusters to dataframe
    data['Cluster'] = clusters

    # Plotting the Clusters
    fig = plt.figure()
    if dimension == 2:
        ax = fig.add_subplot(111)
        for cluster in range(n_clusters):
            cluster_data = reduced_data[clusters == cluster]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
        ax.set_xlabel('PCA Component 1' if apply_pca else 'Feature 1')
        ax.set_ylabel('PCA Component 2' if apply_pca else 'Feature 2')
    else:
        ax = fig.add_subplot(111, projection='3d')
        for cluster in range(n_clusters):
            cluster_data = reduced_data[clusters == cluster]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster}')
        ax.set_xlabel('PCA Component 1' if apply_pca else 'Feature 1')
        ax.set_ylabel('PCA Component 2' if apply_pca else 'Feature 2')
        ax.set_zlabel('PCA Component 3' if apply_pca else 'Feature 3')

    ax.legend()
    st.pyplot(fig)

    # Cluster Summary
    st.header('Cluster Summary')
    st.write("""
    Below, you can see a summary of each cluster, including statistics such as the mean, standard deviation, and range of values for each feature. This can help you understand the characteristics of each cluster, such as which assets are more volatile or have higher returns.
    """)

    cols = st.columns(n_clusters)
    for cluster, col in zip(range(n_clusters), cols):
        with col:
            st.subheader(f'Cluster {cluster}')
            cluster_info = data[data['Cluster'] == cluster]
            st.write(cluster_info.describe())

    # Feature Importance Visualization using Mean Decrease in Variance
    st.header('Feature Importance Visualization')
    st.write("""
    Feature importance is calculated here by looking at the variance of each feature. Features with higher variance tend to have a larger impact on clustering since they introduce more differentiation between data points.
    """)
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
    st.write("""
    ### Insights from Clustering
    - **Cluster Characteristics**: The clustering results can help identify groups of stocks that move similarly. This could be useful for building diversified portfolios or understanding sector-based movements.
    - **Impact of Dimensionality Reduction**: Applying PCA helps reduce the complexity of the dataset while retaining the essential information, making visualization easier and clustering more efficient. You can toggle PCA to see how it affects the clusters.
    - **3D Visualization**: The ability to visualize in three dimensions provides a richer understanding of how the clusters are formed, especially when there are more features involved.

    Adjust the parameters on the left to experiment with different clustering scenarios, and observe how the clusters change.
    """)

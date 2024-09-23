import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import numpy as np

# Title and Introduction
st.title("K-Means Clustering with Streamlit")
st.markdown("""
This app allows you to perform K-Means clustering on either your own dataset or one of the provided sample datasets.
1. Select a dataset or upload your own.
2. Select the features to cluster.
3. Choose the number of clusters.
4. Visualize the clustered data.
""")

# Sidebar: Dataset Selection
st.sidebar.header("Select a Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset or upload your own",
    ("Iris", "Wine", "Breast Cancer", "Upload Your Own")
)

# Load sample datasets
def load_sample_data(dataset_choice):
    if dataset_choice == "Iris":
        data = load_iris(as_frame=True)
        df = data['data']
        df['target'] = data['target']
        return df
    elif dataset_choice == "Wine":
        data = load_wine(as_frame=True)
        df = data['data']
        df['target'] = data['target']
        return df
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = data['data']
        df['target'] = data['target']
        return df

# Step 1: Upload Dataset or Select Sample Dataset
uploaded_file = None
if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview")
    st.write(data.head())
elif dataset_choice != "Upload Your Own":
    # Load the sample dataset
    data = load_sample_data(dataset_choice)
    st.write(f"Sample Dataset ({dataset_choice}) Preview")
    st.write(data.head())

# Proceed if a dataset has been loaded
if 'data' in locals():
    # Step 2: Feature Selection
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect('Select features for clustering', data.columns)

    if len(features) > 0:
        # Display selected features
        st.write(f"Selected features: {features}")
        X = data[features]

        # Step 3: Standardize the Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 4: Select the number of clusters (K)
        st.sidebar.header("Select Number of Clusters")
        num_clusters = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)

        # Step 5: Perform K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        # Add cluster labels to the dataset
        data['Cluster'] = labels

        # Step 6: Visualize the Clusters
        st.header("Cluster Visualization")
        if len(features) == 2:
            # 2D Scatter Plot
            fig, ax = plt.subplots()
            sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=data, palette='Set1', ax=ax)
            ax.set_title(f"K-Means Clustering with {num_clusters} Clusters")
            st.pyplot(fig)
        elif len(features) == 3:
            # 3D Scatter Plot (if 3 features are selected)
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[features[0]], data[features[1]], data[features[2]], c=labels, cmap='Set1', s=50)
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2])
            ax.set_title(f"K-Means Clustering with {num_clusters} Clusters")
            st.pyplot(fig)
        else:
            st.write("Please select 2 or 3 features for visualization.")

        # Step 7: Display Cluster Centers
        st.header("Cluster Centers")
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
        st.write(cluster_centers_df)

    else:
        st.write("Please select at least one feature for clustering.")
else:
    st.write("Please select a sample dataset or upload your own.")

`

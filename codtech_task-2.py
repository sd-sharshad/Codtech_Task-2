import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generate synthetic data
def generate_data(n_samples=300, n_features=2, n_clusters=3):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return StandardScaler().fit_transform(X)

# Perform K-means clustering
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

# Perform hierarchical clustering
def hierarchical_clustering(X, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    return labels

# Perform DBSCAN clustering
def dbscan_clustering(X, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

# Evaluate clustering results
def evaluate_clustering(X, labels):
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        return silhouette, davies_bouldin
    else:
        return None, None

# Visualize clustering results
def visualize_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()

# Main function
def main():
    # Generate data
    X = generate_data()

    # Perform clustering
    kmeans_labels = kmeans_clustering(X, n_clusters=3)
    hc_labels = hierarchical_clustering(X, n_clusters=3)
    dbscan_labels = dbscan_clustering(X)

    # Evaluate clustering results
    clustering_methods = [
        ("K-means", kmeans_labels),
        ("Hierarchical", hc_labels),
        ("DBSCAN", dbscan_labels)
    ]

    for method_name, labels in clustering_methods:
        silhouette, davies_bouldin = evaluate_clustering(X, labels)
        
        if silhouette is not None and davies_bouldin is not None:
            print(f"{method_name} Clustering Results:")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            print()
        else:
            print(f"{method_name} Clustering: Unable to compute scores (possibly only one cluster found)")
            print()

        # Visualize clusters
        visualize_clusters(X, labels, f"{method_name} Clustering")

if __name__ == "__main__":
    main()
    
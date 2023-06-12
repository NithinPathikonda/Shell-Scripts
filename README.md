# Shell-Scripts

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Generate random data points

np.random.seed(42)

n_samples = 100

data = pd.DataFrame({

    'Feature1': np.random.rand(n_samples),

    'Feature2': np.random.rand(n_samples),

    'Feature3': np.random.rand(n_samples),

    'Feature4': np.random.rand(n_samples)

})

# Preprocess the data

features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']

data_scaled = StandardScaler().fit_transform(data[features])

# Perform K-means clustering

k = 3

kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(data_scaled)

# Analyze and justify the results

cluster_centers = kmeans.cluster_centers_

cluster_labels = kmeans.labels_

data['Cluster'] = cluster_labels

# Compare cluster statistics

cluster_stats = data.groupby('Cluster')[features].mean()

print(cluster_stats)

# Visualize the clusters

for cluster in range(k):

    cluster_data = data[data['Cluster'] == cluster]

    plt.scatter(cluster_data['Feature1'], cluster_data['Feature2'], label=f'Cluster {cluster+1}')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', label='Cluster Centers')

plt.xlabel('Feature1')

plt.ylabel('Feature2')

plt.title('K-means Clustering')

plt.legend()

plt.show()

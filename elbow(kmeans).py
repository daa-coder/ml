# this code is of k-means using elbow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load your dataset
# Replace this with your actual file
df = pd.read_csv('your_file.csv')

# Step 2: Select numeric features
X = df.select_dtypes(include=[np.number])

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method to determine optimal k
inertia = []
K_RANGE = range(1, 11)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Step 5: Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_RANGE, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Step 6: Choose k (for example, from visual inspection or manually set)
optimal_k = 4  # <-- Change this to your elbow-determined k

# Step 7: Apply KMeans with optimal k
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Step 8: Visualize Clusters using PCA (if dimensions > 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centers')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f'KMeans Clustering Visualization (k={optimal_k})')
plt.legend()
plt.grid(True)
plt.show()
# plz note this : 
# This will work for any dataset with numeric features.

# PCA is used for visualization if your dataset has more than 2 features.

# You can adjust optimal_k manually or use automation based on the elbow bend.

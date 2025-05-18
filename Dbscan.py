import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Option 1: Generate synthetic data (comment this out if using your own dataset)
# X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Option 2: Load your own dataset (uncomment and update file path and feature columns)
data = pd.read_csv('your_dataset.csv')
X = data[['feature1', 'feature2']].values

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

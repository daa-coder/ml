# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

# Step 1: Load your dataset (replace 'your_dataset.csv' with your file path)
# Make sure your dataset is in CSV format and features are numeric
df = pd.read_csv('your_dataset.csv')

# Step 2: Select feature columns for clustering (adjust column selection as needed)
# Here, we use all columns except the last one assuming it’s a label or target column
X = df.iloc[:, :-1].values  

# Step 3: Initialize KMedoids with desired number of clusters (k)
k = 4  # Change k according to your problem
kmedoids = KMedoids(n_clusters=k, random_state=42)

# Step 4: Fit the model and predict cluster labels
labels = kmedoids.fit_predict(X)

# Step 5: Visualize clusters (works best if features are 2D)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', label='Data points')

# Highlight medoids (cluster centers)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], 
            s=300, c='red', marker='X', label='Medoids')

# Plot settings
plt.title(f'K-Medoids Clustering with k={k}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#Just update:

#'your_dataset.csv' → your actual dataset file path

#df.iloc[:, :-1] → columns you want to cluster on (adjust slicing if needed)

#k = 4 → number of clusters you want


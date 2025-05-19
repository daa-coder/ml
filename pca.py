# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========== OPTION 1: Manual dataset (custom data) ==========
# You can replace this with your actual data
data = {
    'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
    'Feature2': [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]
}
df = pd.DataFrame(data)

X = df.values  # Extract features
y = None       # No labels in manual input

# ========== OPTION 2: Iris Dataset (Uncomment to use) ==========
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
# target_names = iris.target_names

# Step 1: Standardize the data (very important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA (2 principal components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Plot the PCA result
plt.figure(figsize=(8, 6))

# For manual data (no labels)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=50)
plt.title("PCA: 2 Components (Manual Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# Step 4: Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ========== For Iris Dataset visualization ==========
# Uncomment below if using Iris dataset
# plt.figure(figsize=(8, 6))
# for i, target_name in enumerate(target_names):
#     plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
# plt.title("PCA: 2 Components (Iris Dataset)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.grid(True)
# plt.show()

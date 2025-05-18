# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your dataset (replace 'your_dataset.csv' with your file path)
df = pd.read_csv('your_dataset.csv')

# Step 2: Prepare data for K-Prototypes
# Identify categorical columns by their names or indices (adjust accordingly)
categorical_columns = ['cat_col1', 'cat_col2']  # replace with your categorical column names

# Convert categorical columns to integer codes for K-Prototypes
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 3: Extract data as numpy array for K-Prototypes
X = df.values

# Step 4: Set the indices of categorical columns (0-based indexing)
cat_columns_idx = [df.columns.get_loc(col) for col in categorical_columns]

# Step 5: Initialize K-Prototypes model with number of clusters (k)
k = 3  # change k as needed
kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)

# Step 6: Fit the model and predict cluster labels
clusters = kproto.fit_predict(X, categorical=cat_columns_idx)

# Step 7: Visualization (only if you have 2 numerical features)
# Assuming first two columns are numerical for visualization
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
plt.title(f'K-Prototypes Clustering with k={k}')
plt.xlabel('Feature 1 (numerical)')
plt.ylabel('Feature 2 (numerical)')
plt.show()

# How to customize:
# Replace 'your_dataset.csv' with your dataset path.

# Update categorical_columns with your categorical column names.

# Change k for number of clusters.

# Visualization works best if you have 2 numerical columns (adjust the plotting lines if needed).

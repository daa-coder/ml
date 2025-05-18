df = pd.read_csv('kmeans_data.csv')
df.dropna(inplace=True)

X = df[['feature1', 'feature2']]  # Replace with actual numeric features

from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    silhouette_scores.append(sil_score)

plt.plot(range(2, 11), silhouette_scores, marker='x')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

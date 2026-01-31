# ----------------------------------------------------
# Implementing and Evaluating K-Means Clustering from Scratch
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------------------------------
# Step 1: Generate Synthetic Dataset
# ----------------------------------------------------
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)


# ----------------------------------------------------
# Step 2: K-Means Implementation from Scratch
# ----------------------------------------------------
class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def fit(self, X):
        # Random initialization of centroids
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # E-step: Assign clusters
            self.labels = self._assign_clusters(X)

            # M-step: Update centroids
            new_centroids = self._update_centroids(X)

            # Convergence check
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tolerance):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        return np.array([
            X[self.labels == i].mean(axis=0) for i in range(self.k)
        ])

    def inertia(self, X):
        total_wcss = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            total_wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return total_wcss


# ----------------------------------------------------
# Step 3: Run Scratch K-Means
# ----------------------------------------------------
scratch_kmeans = KMeansFromScratch(k=3)
scratch_kmeans.fit(X)

scratch_inertia = scratch_kmeans.inertia(X)
scratch_silhouette = silhouette_score(X, scratch_kmeans.labels)

print("Scratch K-Means Results")
print("-----------------------")
print("Inertia (WCSS):", scratch_inertia)
print("Silhouette Score:", scratch_silhouette)


# ----------------------------------------------------
# Step 4: Scikit-learn KMeans for Comparison
# ----------------------------------------------------
sk_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
sk_labels = sk_kmeans.fit_predict(X)

sk_inertia = sk_kmeans.inertia_
sk_silhouette = silhouette_score(X, sk_labels)

print("\nScikit-learn KMeans Results")
print("----------------------------")
print("Inertia (WCSS):", sk_inertia)
print("Silhouette Score:", sk_silhouette)


# ----------------------------------------------------
# Step 5: Visualization of Scratch K-Means Result
# ----------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=scratch_kmeans.labels, cmap="viridis", s=30)
plt.scatter(
    scratch_kmeans.centroids[:, 0],
    scratch_kmeans.centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.title("K-Means Clustering from Scratch")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()




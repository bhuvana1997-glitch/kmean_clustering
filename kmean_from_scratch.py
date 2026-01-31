# K-Means Clustering from Scratch

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Generate Sample Data
# -----------------------------
np.random.seed(42)

X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])
X3 = np.random.randn(50, 2) + np.array([2, -2])

X = np.vstack((X1, X2, X3))


# -----------------------------
# Step 2: K-Means Class
# -----------------------------
class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # Randomly initialize centroids
        random_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # Assign clusters
            self.labels = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = self._update_centroids(X)

            # Check convergence
            if np.all(self.centroids == new_centroids):
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
        total = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            total += np.sum((cluster_points - self.centroids[i]) ** 2)
        return total


# -----------------------------
# Step 3: Train Model
# -----------------------------
kmeans = KMeansFromScratch(k=3)
kmeans.fit(X)

print("Final Centroids:\n", kmeans.centroids)
print("Inertia (WCSS):", kmeans.inertia(X))


# -----------------------------
# Step 4: Visualization
# -----------------------------
plt.figure(figsize=(6, 6))
for i in range(3):
    plt.scatter(
        X[kmeans.labels == i][:, 0],
        X[kmeans.labels == i][:, 1],
        label=f"Cluster {i+1}"
    )

plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c='black',
    s=200,
    marker='X',
    label='Centroids'
)

plt.title("K-Means Clustering from Scratch")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
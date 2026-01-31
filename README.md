1.Project Description
This project focuses on implementing the K-Means clustering algorithm entirely from scratch using NumPy, without relying on machine learning libraries for the core algorithm. The objective is to understand the internal working of K-Means, including centroid initialization, iterative optimization, convergence behavior, and performance evaluation.
Synthetic data is generated using sklearn.datasets.make_blobs, and the performance of the scratch implementation is evaluated and compared with Scikit-learn’s KMeans using standard clustering metrics.
2.Objectives
Implement K-Means clustering from first principles
Understand E-step (cluster assignment) and M-step (centroid update)
Analyze convergence using centroid movement tolerance
Evaluate clustering quality using Inertia (WCSS) and Silhouette Score
Compare scratch implementation with Scikit-learn KMeans
Visualize clustering results in 2D
3.Technologies Used
Python
NumPy (core K-Means implementation)
Matplotlib (visualization)
Scikit-learn (data generation, evaluation, and comparison)
4.Project Structure
Copy code

KMeans_From_Scratch/
│
├── kmeans_from_scratch.py
├── README.md
5. Methodology
1. Data Generation
Synthetic dataset created using make_blobs
300 samples with 3 cluster centers
Ensures clear cluster separation for evaluation
2. K-Means from Scratch
Centroids initialized randomly
E-step: Assign each data point to the nearest centroid
M-step: Update centroids as the mean of assigned points
Iterations continue until:
Centroid movement is below a tolerance value, or
Maximum iterations are reached
3. Evaluation Metrics
Inertia (WCSS): Measures compactness of clusters
Silhouette Score: Measures cluster separation and cohesion
4. Comparison
Results from scratch implementation are compared with:
Scikit-learn’s KMeans
Both inertia and silhouette scores are reported
6.Output
Final centroid coordinates
Inertia (WCSS) values
Silhouette scores
2D scatter plot showing:
Clustered data points
Final centroids

*Project Overview
This project demonstrates the implementation of the K-Means clustering algorithm on a synthetic dataset. The goal is to generate data, determine the optimal number of clusters using evaluation techniques, apply K-Means clustering, and visualize the results.
The project uses Scikit-learn for data generation and clustering, along with Matplotlib for visualization.
*Objectives
Generate a synthetic dataset with multiple clusters
Apply K-Means clustering using K-Means++ initialization
Identify the optimal number of clusters
Evaluate clustering performance
Visualize clustering results
*Technologies Used
Python
NumPy
Matplotlib
Scikit-learn
*Dataset Generation
A synthetic dataset is created using sklearn.datasets.make_blobs with the following characteristics:
Number of samples: 500
Number of clusters: 4
Features: 2
Random state: 42 (for reproducibility)
This ensures clear cluster separation and meaningful evaluation.
* K-Means Clustering
K-Means is an unsupervised learning algorithm that groups data points into K clusters by minimizing the within-cluster sum of squares (Inertia).
K-Means++ initialization is used to:
Select better initial centroids
Improve convergence speed
Reduce the chances of poor clustering
* Determining the Optimal Number of Clusters
1. Elbow Method
Calculates Inertia for different values of K (2 to 8)
Plots K vs Inertia
The “elbow point” indicates an optimal balance between cluster count and compactness
2. Silhouette Analysis
Measures how well data points fit within their assigned clusters
Silhouette score ranges from -1 to 1
A higher score indicates better-defined clusters
The value of K with the highest silhouette score is selected as optimal
* Final Model & Evaluation
K-Means is executed using the optimal K
The final Silhouette Score is calculated to evaluate clustering quality
Higher silhouette score confirms better clustering performance
*Visualizations
Two key plots are generated:
Elbow Plot – Shows inertia values for different cluster counts
Cluster Scatter Plot – Displays final clustered data points with centroids
These visualizations help interpret clustering behavior and validate results.
*Conclusion
This project successfully demonstrates:
Synthetic data generation
Optimal cluster selection using evaluation metrics
Effective clustering using K-Means++
Visual analysis of clustering results
The approach can be extended to real-world datasets for exploratory data analysis and segmentation tasks.

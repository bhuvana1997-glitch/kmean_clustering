📌 Implementing and Evaluating K-Means Clustering from Scratch
📖 Project Overview
This project focuses on implementing the K-Means clustering algorithm from scratch using Python and NumPy, without relying on high-level machine learning libraries for the core algorithm. The objective is to understand the internal working of K-Means, evaluate its performance, and compare it with the standard scikit-learn KMeans implementation.
The project also includes data generation, optimal cluster selection using the Elbow Method, visualization of clusters, and runtime comparison.
🎯 Objectives
Implement K-Means clustering algorithm from scratch
Understand centroid initialization and iterative optimization
Select the optimal number of clusters using the Elbow Method
Visualize clustering results
Compare custom implementation with scikit-learn KMeans
🛠 Technologies Used
Programming Language: Python
Libraries:
NumPy
Matplotlib
scikit-learn (for comparison only)
📊 Dataset
Synthetic dataset generated using make_blobs
Number of samples: 500
Number of clusters: 4
Dataset contains two numerical features for easy visualization
⚙️ Methodology
Generate synthetic data points
Randomly initialize centroids
Assign data points to nearest centroids using Euclidean distance
Update centroids as the mean of assigned points
Repeat until convergence or maximum iterations
Use Elbow Method to find optimal K
Compare results with scikit-learn KMeans
📈 Elbow Method
The Elbow Method is used to determine the optimal number of clusters by plotting Inertia vs Number of Clusters (K).
The point where the curve bends indicates the best value of K.
📉 Results and Visualization
Clusters formed by the custom K-Means algorithm closely match those produced by scikit-learn
Visualization shows clear cluster separation
Custom implementation takes more runtime compared to scikit-learn due to optimization differences
✅ Conclusion
This project successfully demonstrates the implementation of K-Means clustering from scratch and provides a clear understanding of how clustering algorithms work internally. While the custom implementation is educational, scikit-learn’s optimized version performs significantly faster in practice.

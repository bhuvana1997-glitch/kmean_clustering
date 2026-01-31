1.Project Overview
This project demonstrates the implementation of the K-Means clustering algorithm from scratch using Python and NumPy. The goal is to understand how K-Means works internally without using any built-in machine learning libraries such as scikit-learn.
The project includes data generation, cluster assignment, centroid updates, convergence checking, evaluation using inertia, and visualization of final clusters.
2.Objectives
Implement K-Means clustering from basic principles
Understand iterative optimization (E-step and M-step)
Evaluate clustering performance using inertia (WCSS)
Visualize clusters and centroids
3.Technologies Used
Python
NumPy
Matplotlib
4.Algorithm Steps
Initialize cluster centroids randomly
Assign each data point to the nearest centroid (E-step)
Update centroids as the mean of assigned points (M-step)
Repeat until centroids do not change or max iterations reached
Evaluate clustering using inertia
Visualize final clusters
5.Evaluation Metric
Inertia (WCSS): Measures the sum of squared distances between data points and their cluster centroids.
Lower inertia indicates better clustering.
6.Output
Printed final centroid values
Inertia score
Scatter plot showing clustered data points and centroids


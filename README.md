1. Project Title
Implementing and Analyzing the K-Means Clustering Algorithm from Scratch Using Python
2. Objective
The objective of this project is to implement the K-Means clustering algorithm from scratch using NumPy, understand its internal working, determine the optimal number of clusters using the Elbow Method, and evaluate clustering performance using the Silhouette Score. The results are also compared with Scikit-learn’s K-Means implementation.
3. Tools and Technologies Used
Python
NumPy
Matplotlib
Scikit-learn (only for comparison and evaluation)
4. Dataset Description
A synthetic dataset is generated using NumPy.
The dataset contains multiple data points with two numerical features and is designed to form clearly separable clusters for effective clustering analysis.
5. K-Means Algorithm Implementation (From Scratch)
The K-Means algorithm is implemented manually using NumPy without relying on machine learning libraries.
Steps followed:
Random initialization of centroids
Assignment of data points to the nearest centroid using Euclidean distance
Updating centroids by computing the mean of assigned points
Repeating steps until convergence
This implementation provides a clear understanding of how K-Means works internally.
6. Elbow Method for Optimal K Selection
The Elbow Method is used to identify the optimal number of clusters.
Inertia is calculated for different values of K
A graph of K vs Inertia is plotted
The point where the reduction in inertia slows down significantly is chosen as the optimal K
Result:
Based on the elbow graph, K = 3 is selected as the optimal number of clusters because it shows a clear bend, indicating diminishing returns beyond this point.
7. Cluster Characteristics Analysis
After clustering with K = 3, the characteristics of each cluster are analyzed.
Mean feature values are calculated for each cluster
Each cluster shows distinct mean values
This confirms that the algorithm successfully grouped similar data points together
The clusters are well separated, indicating effective clustering.
8. Silhouette Score Evaluation
The Silhouette Score is used to measure the quality of clustering.
Scores Obtained:
Scratch Implementation: Calculated using a manually implemented silhouette score
Scikit-learn Implementation: Calculated using sklearn.metrics.silhouette_score
Comparison and Analysis:
The silhouette scores from both implementations are very close.
Minor differences occur due to:
Random centroid initialization
Optimization techniques used in Scikit-learn
Overall, the scratch implementation performs comparably to Scikit-learn, validating the correctness of the algorithm.
9. Results and Visualization
Elbow Method graph for optimal K selection
Scatter plot showing clustered data points
Centroids marked clearly
Printed text outputs for:
Optimal K reasoning
Cluster characteristics
Silhouette score comparison
10. Conclusion
This project successfully demonstrates the implementation of the K-Means clustering algorithm from scratch.
The Elbow Method and Silhouette Score confirm that the chosen number of clusters is optimal.
The close similarity between scratch and Scikit-learn results shows that the implementation is accurate and effective.
11. Future Scope
Apply the algorithm to real-world datasets
Extend to higher-dimensional data
Improve initialization using K-Means++
Compare with other clustering algorithms

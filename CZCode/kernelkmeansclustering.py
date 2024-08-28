import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

def generate_circles(n_samples=300, noise=0.05, factor=0.3):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y

def generate_two_moons(n_samples=300, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


def kernel_k_means(X, n_clusters=2, max_iter=100, gamma=1.0):
    """
    Apply kernel k-means clustering to the data.
    
    Parameters:
    - X: Input data, shape (n_samples, n_features)
    - n_clusters: Number of clusters
    - max_iter: Maximum number of iterations
    - gamma: Parameter for the RBF kernel
    
    Returns:
    - labels: Cluster labels for each point
    """
    # Compute the RBF (Gaussian) kernel
    K = rbf_kernel(X, gamma=gamma)
    
    # Initialize KMeans to find initial cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(K)
    
    for _ in range(max_iter):
        # Compute the distance to the cluster centers in the feature space
        distances = np.zeros((X.shape[0], n_clusters))
        for j in range(n_clusters):
            mask = (labels == j)
            n_points = np.sum(mask)
            if n_points == 0:
                continue
            cluster_center = K[:, mask].sum(axis=1) / n_points
            distances[:, j] = K.diagonal() - 2 * cluster_center + K[mask][:, mask].sum() / (n_points**2)
        
        # Update labels
        new_labels = distances.argmin(axis=1)
        
        # Check for convergence
        if np.all(labels == new_labels):
            break
        labels = new_labels
    
    return labels

def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=30)
    plt.title(title)
##    plt.xlabel('X1')
##    plt.ylabel('X2')
    plt.show()

# Generate the circles dataset
X, y = generate_circles(n_samples=300, noise=0.05, factor=0.35)

# Apply kernel k-means clustering
labels = kernel_k_means(X, n_clusters=2, max_iter=100, gamma=1.0)

# Plot the clustered data
plot_clusters(X, labels,title='')

X2, y2 = generate_two_moons(n_samples=300, noise=0.1)

labels2 = kernel_k_means(X, n_clusters=2, max_iter=100, gamma=1.0)

plot_clusters(X2, labels2,title='')


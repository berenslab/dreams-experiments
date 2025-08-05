from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import scipy
import numpy as np

def embedding_quality(X, Z, classes, knn=10, knn_classes=6, subsetsize=1000, seed=0):
    """
    Evaluate the quality of a low-dimensional embedding.
    
    Parameters:
    X : ndarray
        Original high-dimensional data (N samples x D features).
    Z : ndarray
        Low-dimensional embedding (N samples x d features, where d < D).
    classes : array-like
        Class labels for each sample (length N).
    knn : int, optional
        Number of nearest neighbors for local neighborhood overlap. Default is 10.
    knn_classes : int, optional
        Number of nearest neighbors for class-based global neighborhood overlap. Default is 10.
    subsetsize : int, optional
        Number of samples to use for distance correlation calculation. Default is 1000.
    
    Returns:
    mnn : float
        Mean nearest-neighbor overlap (local).
    mnn_global : float
        Mean nearest-neighbor overlap of class centroids.
    rho : float
        Spearman correlation between pairwise distances in high-dimensional and low-dimensional spaces.
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    
    # Compute local nearest-neighbor indices for high-dimensional data
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)  # Nearest neighbor indices for X

    # Compute local nearest-neighbor indices for low-dimensional embedding
    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)  # Nearest neighbor indices for Z

    # Calculate mean nearest-neighbor overlap (local)
    intersections = 0.0
    for i in range(X.shape[0]):
        # Count intersection of nearest neighbors in original and embedded spaces
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn  # Normalize by total samples and neighbors

    # Compute class centroids for high-dimensional and embedded spaces
    cl, cl_inv = np.unique(classes, return_inverse=True)  # Unique class labels and their indices
    C = cl.size  # Number of unique classes
    mu1 = np.zeros((C, X.shape[1]))  # Centroids for X
    mu2 = np.zeros((C, Z.shape[1]))  # Centroids for Z
    for c in range(C):
        # Calculate centroids for each class in X and Z
        mu1[c, :] = np.mean(X[cl_inv == c, :], axis=0)
        mu2[c, :] = np.mean(Z[cl_inv == c, :], axis=0)

    # Compute global nearest-neighbor indices based on class centroids
    nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)  # Nearest neighbor indices for class centroids in X
    nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)  # Nearest neighbor indices for class centroids in Z

    # Calculate mean nearest-neighbor overlap (global, class-based)
    intersections = 0.0
    for i in range(C):
        # Count intersection of nearest neighbors in class centroid space
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes  # Normalize by number of classes and neighbors

    # Sample data for pairwise distance correlation calculation
    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)  # Random subset of samples

    # Compute pairwise distances in original and embedded spaces
    d1 = pdist(X[subset, :])  # Pairwise distances in high-dimensional space
    d2 = pdist(Z[subset, :])  # Pairwise distances in low-dimensional space

    # Calculate Spearman correlation between the pairwise distances
    rho = scipy.stats.spearmanr(d1[:, None], d2[:, None]).correlation

    # 10. Return the three metrics
    return (mnn, mnn_global, rho)


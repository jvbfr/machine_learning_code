import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def find_n_components(X, treshhold = 0.95):
        """
    Compute the optimal number of components to retain for PCA.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    treshold : int
        Explanied variance treshold to select the component

    Returns
    -------
    n_components : int
        The optimal number of components to retain based on the elbow method.
    """

    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the principal components
    pca = PCA()
    pca.fit(X_centered)
    
    # Plot the explained variance ratio for each component
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    # Use the elbow method to choose the number of components
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    elbow_index = np.argmax(cumulative_variance >= treshhold)  # Select the component where 95% of the variance is explained
    n_components = elbow_index + 1
    
    return n_components
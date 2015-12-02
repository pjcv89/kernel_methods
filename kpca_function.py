
#Author: Pablo Campos V.

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array
        gamma: Parameter for the RBF kernel.
        n_components: The number of components to be returned.

    Returns the k eigenvectors (alphas) that correspond to the k largest 
        eigenvalues (lambdas).

    """
    # Squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding 
    # eigenvectors from the centered Kernel matrix.
    # Here we use eigsh, which is a wrapper to ARPACK solver for large scale eigenvalue problems.
    eigvals, eigvecs = eigsh(K_norm, k = n_components, which = 'LM')
    
    # Obtaining the i eigenvectors  that corresponds to the i highest eigenvalues (lambdas).
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]

    return alphas, lambdas
    
    
#Example: Iris dataset 
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :4]  
y = iris.target
 
 
gamma = .2
alphas, lambdas = stepwise_kpca(X, gamma, n_components=4)
    

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)
    
def itera(base_new, X, gamma, alphas, lambdas):
     projection = np.array([project_x(row, X, gamma, alphas=alphas, lambdas=lambdas) for row in base_new])
     return projection
   
def grafica(base_projected, i, j):
    plt.plot(base_projected[:, i-1], base_projected[:, j-1], "bo")
    plt.title("Projections")
    plt.xlabel("Component $i$")
    plt.ylabel("Component $j$")
    plt.show()
 
#Call to -itera- function  
base_new_projected = itera(X,X,gamma,alphas,lambdas)   

#Visualization 
grafica(base_new_projected,1,2) 
grafica(base_new_projected,1,3) 
grafica(base_new_projected,1,4) 
grafica(base_new_projected,2,3) 
grafica(base_new_projected,2,4)
grafica(base_new_projected,3,4)
import numpy as np


def constraint_unit_diagonal(flattened_array: np.array):
    """
    Reconstructs the matrix based on the provided array, which is assumed to be 
    flattened. In context of copula estimation, initial values need to be provided
    to the optimizer for ML estimation. This is done by decomposing the 
    initial correlation matrix using cholesky in order to obtain the unique
    elements of the correlation matrix and flatten them using np.flatten. This
    is different from vecl or vec.
    
    This function reconstructs the matrix and check if the diagonals are unity
    for a valid correlation matrix. Note, we use np.flatten

    Args:
        flattened_array (np.array): _description_

    Returns:
        _type_: _description_
    """
    # Get dimension of the matrix, lenght of the array should be a square number
    dim = int(np.sqrt(len(flattened_array)))
    
    assert isinstance(dim, int), 'Incorrect array, dim of vector must be of n^2 x 1'
    
    L = np.reshape(flattened_array, (dim, dim))
    
    reconst_mat = np.diag(L @ L.T)

    return reconst_mat - 1



def constraint_is_positive_definite(flattened_array: np.array) -> float:
    n = int(np.sqrt(len(flattened_array)))  # Assuming square matrix
    matrix = flattened_array.reshape(n, n)
    
    try:
        np.linalg.cholesky(matrix)
        return 0  # If Cholesky decomposition succeeds, return 0 (satisfying equality constraint)
    except np.linalg.LinAlgError:
        return 1  # Otherwise return 1 (violating equality constraint)

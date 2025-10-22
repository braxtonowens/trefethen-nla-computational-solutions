import numpy as np

def compute_svd(A):
    """
    Compute the singular value decomposition (SVD) of a real matrix A.

    This function uses numpy.linalg.svd to compute the matrices U, S, and Vh such that
    A = U @ np.diag(S) @ Vh. The singular values in S are the square roots of the eigenvalues
    of A.T @ A, and U and Vh are orthonormal ([numpy.linalg.svd — NumPy v2.2 Manual](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html#:~:text=The%20decomposition%20is%20performed%20using,gesdd)).

    Parameters
    ----------
    A : ndarray
        A real or complex matrix of shape (m, n).

    Returns
    -------
    U : ndarray
        Unitary matrix of shape (m, m) whose columns are the left singular vectors.
    S : ndarray
        1D array of length min(m, n) containing the singular values.
    Vh : ndarray
        Unitary matrix of shape (n, n) whose rows are the right singular vectors.

    Example
    -------
    >>> A = np.array([[1, 2], [3, 4], [5, 6]])
    >>> U, S, Vh = compute_svd(A)
    >>> # Verify reconstruction
    >>> np.allclose(A, U @ np.diag(S) @ Vh)
    True
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    return U, s, Vh

if __name__ == "__main__":
    # Example usage
    A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    U, S, Vh = compute_svd(A)
    print("U =\n", U)
    print("Singular values:", S)
    print("Vh =\n", Vh)
    # reconstruct
    print("Reconstruction:\n", U @ np.diag(S) @ Vh)

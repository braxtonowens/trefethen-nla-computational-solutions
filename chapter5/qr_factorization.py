import numpy as np

def qr_factorization(A):
    """
    Compute the QR factorization of a real or complex matrix A.
    This function uses numpy.linalg.qr to compute matrices Q and R such that A = Q @ R,
    where Q has orthonormal columns and R is upper triangular.  The QR factorization
    is fundamental in solving least squares problems and computing projections.  In
    particular, if Q is obtained from A, the orthogonal projection of a vector v onto
    the range of A can be written as Q @ Q.T @ v, which is equivalent to A @ (A.T @ A)^{-1} @ A.T @ v for full-rank A ([17. Projections](https://www.akshayagrawal.com/lecture-notes/html/projections.html#:~:text=17.%20Projections%20,onto%20%20is)).

    Parameters
    ----------
    A : ndarray
        An m x n matrix (with m >= n) to factorize.

    Returns
    -------
    Q : ndarray
        An m x n matrix with orthonormal columns.
    R : ndarray
        An n x n upper triangular matrix.

    Example
    -------
    >>> A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]])
    >>> Q, R = qr_factorization(A)
    >>> # Verify that Q @ R reconstructs A
    >>> np.allclose(A, Q @ R)
    True
    >>> # Project a vector onto the column space of A
    >>> v = np.array([1., 0., 0.])
    >>> projection = Q @ Q.T @ v
    """
    Q, R = np.linalg.qr(A)
    return Q, R

if __name__ == "__main__":
    # Example usage
    A = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 10.]])
    Q, R = qr_factorization(A)
    print("Q =\n", Q)
    print("R =\n", R)
    print("Reconstruction:\n", Q @ R)
    # Example projection
    v = np.array([1., 0., 0.])
    proj = Q @ Q.T @ v
    print("Projection of v onto range(A):", proj)

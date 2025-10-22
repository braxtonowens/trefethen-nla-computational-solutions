import numpy as np

def gram_schmidt(V):
    """
    Perform the classical Gram–Schmidt process on the columns of matrix V.

    The Gram–Schmidt process takes a set of linearly independent vectors and constructs an
    orthonormal set spanning the same subspace by iteratively subtracting projections onto previously
    computed orthonormal vectors and normalizing ([Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#:~:text=function%20U%20%3D%20gramschmidt%28V%29%20,i%29%29%3B%20end%20end)).

    Parameters
    ----------
    V : array-like of shape (m, n)
        Matrix whose columns are the vectors to orthonormalize.

    Returns
    -------
    Q : ndarray of shape (m, n)
        Orthonormal matrix whose columns span the same subspace as the columns of V.
    """
    V = np.asarray(V, dtype=float)
    m, n = V.shape
    Q = np.zeros((m, n))
    for i in range(n):
        # Take the i-th column vector from V
        qi = V[:, i].copy()
        for j in range(i):
            # Subtract projection onto previous orthonormal vector
            proj = np.dot(Q[:, j], qi) * Q[:, j]
            qi = qi - proj
        # Normalize the vector
        norm = np.linalg.norm(qi)
        if norm == 0:
            raise ValueError("Vectors are linearly dependent or zero")
        Q[:, i] = qi / norm
    return Q

if __name__ == "__main__":
    # Example usage
    V = np.array([[3., 2.],
                  [1., 2.]])
    Q = gram_schmidt(V)
    print("Orthonormal basis:")
    print(Q)
    # Verify orthonormality by computing Q^T Q ~ I
    print("Q^T Q =")
    print(np.round(Q.T @ Q, decimals=6))

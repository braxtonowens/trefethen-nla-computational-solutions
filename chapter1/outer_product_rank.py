import numpy as np

def outer_product_rank(u, v):
    """
    Compute the outer product of two vectors and return the resulting matrix and its rank.
    According to linear algebra, the outer product of two nonzero vectors u and v yields a rank 1 matrix because each column
    of the matrix is a scalar multiple of u ([Outer product](https://en.wikipedia.org/wiki/Outer_product#:~:text=Rank%20of%20an%20outer%20product)).

    Parameters
    ----------
    u : array-like (m,)
        First vector.
    v : array-like (n,)
        Second vector.

    Returns
    -------
    A : ndarray of shape (m, n)
        The outer product matrix u v^T.
    r : int
        Rank of A (should be 1 if both u and v are nonzero).
    """
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(1, -1)
    A = u @ v  # compute outer product
    r = np.linalg.matrix_rank(A)
    return A, r

if __name__ == "__main__":
    # Example usage:
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    A, r = outer_product_rank(u, v)
    print("Outer product matrix:\n", A)
    print("Rank of the outer product:", r)

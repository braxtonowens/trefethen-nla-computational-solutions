import numpy as np

def coefficients_in_column_space(A, v):
    """
    Given a matrix A (m x n) and a vector v (m,), compute coefficients c such that A @ c approximates v in the least squares sense.
    In linear algebra, any linear combination of the column vectors of a matrix A can be written as the product A @ c ([Row and column spaces](https://en.wikipedia.org/wiki/Row_and_column_spaces#:~:text=,of%20the%20corresponding%20matrix%20transformation)).
    If v lies exactly in the column space of A, the computed coefficients satisfy A @ c = v.

    Parameters
    ----------
    A : array-like of shape (m, n)
        Matrix whose columns form the basis.
    v : array-like of shape (m,)
        Vector to express as linear combination of columns of A.

    Returns
    -------
    c : ndarray of shape (n,)
        Coefficient vector that minimizes ||A c - v||_2.
    resid : float
        Residual norm ||A c - v||_2.
    """
    A = np.asarray(A)
    v = np.asarray(v)
    c, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)
    resid = np.linalg.norm(A @ c - v)
    return c, resid

if __name__ == "__main__":
    # Example usage
    A = np.array([[1, 0], [0, 1], [1, 1]])
    v = np.array([2, 3, 5])
    c, resid = coefficients_in_column_space(A, v)
    print("Coefficients:", c)
    print("Residual norm:", resid)

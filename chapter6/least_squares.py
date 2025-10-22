import numpy as np


def least_squares(A, b):
    """
    Solve the least squares problem min ||Ax - b||_2 for x.

    The least-squares solution x_hat yields a linear combination of the columns of A that
    has minimal distance to the vector b. In other words, the vector A @ x_hat is the orthogonal
    projection of b onto the column space of A ([leeyngdo/Numerical-Linear-Algebra](https://github.com/leeyngdo/Numerical-Linear-Algebra#:~:text=7.4.%20Least,b)). These coefficients are unique if and only if
    the columns of A are linearly independent ([leeyngdo/Numerical-Linear-Algebra](https://github.com/leeyngdo/Numerical-Linear-Algebra#:~:text=7.4.%20Least,b)).

    Parameters
    ----------
    A : ndarray of shape (m, n)
        The matrix whose column space we project onto.
    b : ndarray of shape (m,)
        The right-hand side vector.

    Returns
    -------
    x_hat : ndarray of shape (n,)
        The least squares solution.
    projection : ndarray of shape (m,)
        The orthogonal projection of b onto the column space of A.
    residual_norm : float
        The Euclidean norm of the residual b - A @ x_hat.
    """
    # Use numpy's least squares solver
    x_hat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    projection = A @ x_hat
    residual_norm = np.linalg.norm(b - projection)
    return x_hat, projection, residual_norm


if __name__ == "__main__":
    # Example usage: solve least squares for an overdetermined system
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2, 2], dtype=float)
    x_hat, proj, res_norm = least_squares(A, b)
    print("Least squares solution x_hat:", x_hat)
    print("Orthogonal projection of b:", proj)
    print("Residual norm:", res_norm)

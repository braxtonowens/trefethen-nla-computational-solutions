"""
norms.py - Compute common vector and matrix norms.

This script defines functions to compute the 1-norm, 2-norm, and infinity-norm of a vector or matrix.
A norm is a function that measures the size of a vector and must satisfy three properties:
1. Positive homogeneity: ||αx|| = |\u03b1| ||x||.
2. Triangle inequality: ||x + y|| ≤ ||x|| + ||y||.
3. Definiteness: ||x|| > 0 for any non-zero vector x, and ||0|| = 0 ([Exam Practice Questions for 'Numerical Linear Algebra' by ...](https://www.kilians.net/post/exam-questions-numerical-linear-algebra/#:~:text=Chapter%203%20)).
"""

import numpy as np

def vector_norms(x):
    """
    Returns a dictionary with the 1-norm, 2-norm, and infinity-norm of a vector x.
    """
    x = np.asarray(x)
    norms = {
        '1-norm': np.linalg.norm(x, ord=1),
        '2-norm': np.linalg.norm(x, ord=2),
        'inf-norm': np.linalg.norm(x, ord=np.inf)
    }
    return norms

def matrix_norms(A):
    """
    Returns a dictionary with the induced 1-norm, 2-norm, and infinity-norm of a matrix A.
    The induced p-norm of a matrix is the maximum of ||A x||_p over all x with ||x||_p = 1.
    """
    A = np.asarray(A)
    norms = {
        '1-norm': np.linalg.norm(A, ord=1),        # max column sum
        '2-norm': np.linalg.norm(A, ord=2),        # spectral norm (largest singular value)
        'inf-norm': np.linalg.norm(A, ord=np.inf)  # max row sum
    }
    return norms

if __name__ == "__main__":
    # Example usage:
    v = np.array([1, -2, 3])
    A = np.array([[1, 2], [-3, 4], [5, -6]])
    print("Vector norms:", vector_norms(v))
    print("Matrix norms:", matrix_norms(A))

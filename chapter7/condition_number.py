import numpy as np


def matrix_condition_numbers(A):
    """
    Compute the condition number of a matrix A using various norms.

    The condition number of a matrix measures the sensitivity of the solution of Ax = b to perturbations in b or A.
    For the 2‑norm, the condition number equals the ratio of the largest singular value to the smallest singular value ([Condition number](https://en.wikipedia.org/wiki/Condition_number#:~:text=If%20%5C,2%29%2C%20then)).
    The function returns a dictionary with condition numbers for the 1-norm, 2-norm, Frobenius norm, infinity norm, and also the built‑in default (2-norm) from numpy.linalg.cond.

    Parameters
    ----------
    A : array_like
        An m x n matrix.

    Returns
    -------
    dict
        A dictionary containing condition numbers for different norms: '2', '1', 'fro', 'inf'.

    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> conds = matrix_condition_numbers(A)
    >>> print(conds['2'])
    # prints the 2-norm condition number of A
    """
    A = np.asarray(A)
    conds = {}
    # Compute condition numbers for various norms; None corresponds to the 2-norm in numpy.linalg.cond
    for p in [None, 1, 2, 'fro', np.inf]:
        try:
            key = str(p if p is not None else 2)
            conds[key] = np.linalg.cond(A, p)
        except np.linalg.LinAlgError:
            # If matrix is singular, condition number is infinite
            key = str(p if p is not None else 2)
            conds[key] = np.inf
    return conds


if __name__ == "__main__":
    # Example usage
    A = np.array([[1.0, 2.0], [3.0, 4.5]])
    conds = matrix_condition_numbers(A)
    print("Condition numbers for A:")
    for norm_key, val in conds.items():
        print(f"Norm {norm_key}: {val:.4f}")

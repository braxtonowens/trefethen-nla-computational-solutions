# Chapter 4 – Singular Value Decomposition  

This chapter focuses on computing the singular value decomposition (SVD) of a matrix. The `svd.py` script implements a function `compute_svd(A)` that uses `numpy.linalg.svd` to compute the left singular vectors, singular values, and right singular vectors of a real matrix A. The docstring explains that the SVD factorizes a matrix A into U, \u03a3, and V^H where U and V are unitary and the singular values are the square roots of the eigenvalues of A^T A, citing the NumPy documentation ([numpy.linalg.svd — NumPy v2.2 Manual](https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html#:~:text=The%20decomposition%20is%20performed%20using,gesdd)). The example usage demonstrates computing the SVD of a sample matrix and verifying reconstruction.  

- `svd.py`: Contains the `compute_svd` function and a demonstration of its usage.

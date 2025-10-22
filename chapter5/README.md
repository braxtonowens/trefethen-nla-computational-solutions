# Chapter 5 – QR Factorization  

This directory contains a solution to the computational problems in Chapter 5 of Trefethen's *Numerical Linear Algebra*. The focus is on the QR factorization of a matrix.  

- **qr_factorization.py** – Implements the QR factorization using NumPy. The script defines a function `qr_factorization(A)` that computes matrices `Q` and `R` such that `A = Q @ R`. The QR factorization expresses a matrix with orthonormal columns in `Q` and an upper-triangular matrix `R`. It is often used in least-squares problems and to compute the orthogonal projection of a vector `v` onto the range of `A`. The orthogonal projection onto the range of `A` can be expressed as `A (A^T A)^{-1} A^T v` ([17. Projections](https://www.akshayagrawal.com/lecture-notes/html/projections.html#:~:text=17.%20Projections%20,onto%20%20is)), which can also be computed via `Q @ Q.T @ v` when `Q` is from the QR factorization.  

To run the example usage, execute the script directly. It demonstrates decomposing a matrix into `Q` and `R` and uses the resulting `Q` to project a vector onto the column space of the matrix. 

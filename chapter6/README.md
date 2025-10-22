# Chapter 6 – Least Squares  

This chapter includes a solution file for the computational problem in Chapter 6, which concerns least–squares approximation for overdetermined systems.  

- **least_squares.py** – demonstrates how to use `numpy.linalg.lstsq` to compute the least–squares solution `x` to an overdetermined system `A x ≈ b`.  The vector `A @ x` is the orthogonal projection of `b` onto the column space of `A`, yielding the linear combination of the columns of `A` that is closest to `b`.  When the columns of `A` are linearly independent, this least–squares solution is unique ([leeyngdo/Numerical-Linear-Algebra](https://github.com/leeyngdo/Numerical-Linear-Algebra#:~:text=7.4.%20Least,b)).  The script also prints the residual norm to quantify the approximation error. 

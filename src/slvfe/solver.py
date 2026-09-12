# -*- coding: utf-8 -*-
"""
GPU (cuSolverDn) linear-algebra kernels replaced by CPU solvers.

Fortran original (slvfe.F90, module sfecalc):

    subroutine posv_wrap(n, mat, vec, info)
        ! cusolverDnDpotrf + cusolverDnDpotrs
        ! Solve  A x = b  where A is (assumed) symmetric positive definite,
        ! using Cholesky factorization (upper triangle used, as in the
        ! Fortran code: CUBLAS_FILL_MODE_UPPER).

    subroutine syevr_wrap(n, mat, eigval, info)
        ! cusolverDnDsyevd
        ! Full eigen-decomposition of a symmetric matrix (eigenvalues +
        ! eigenvectors), used as a fallback when the correlation matrix is
        ! (numerically) singular / not positive definite.

Both are trivially replaced by CPU LAPACK calls via SciPy. There is no
GPU-specific behaviour to preserve; the algorithms (Cholesky / symmetric
eigendecomposition) are exactly the LAPACK routines cuSolverDn wraps on
the GPU.

All arrays are float64 (double precision), matching the original Fortran
code which is compiled with a compiler flag that promotes `real` to
8 bytes (e.g. -r8 / -fdefault-real-8). NumPy's default float dtype is
already float64, so no special handling is required for precision.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, LinAlgError

# Matches the Fortran parameters in posv_wrap
_RESIDUAL_ERROR = 1.0e-6
_ABS_ERROR = 1.0e6


def posv_wrap(mat: np.ndarray, vec: np.ndarray) -> tuple[np.ndarray, int]:
    """Solve ``mat @ x = vec`` for a symmetric (positive definite) ``mat``.

    Equivalent of the Fortran ``posv_wrap`` (cusolverDnDpotrf /
    cusolverDnDpotrs), performed on the CPU via LAPACK's DPOTRF/DPOTRS
    (through SciPy's Cholesky routines).

    Only the upper triangle of ``mat`` is referenced (matches
    CUBLAS_FILL_MODE_UPPER in the original code).

    Parameters
    ----------
    mat : (n, n) ndarray, float64
        Symmetric matrix A. Not modified.
    vec : (n,) ndarray, float64
        Right-hand side b. Not modified.

    Returns
    -------
    x : (n,) ndarray, float64
        Solution vector (meaningless if ``info != 0``).
    info : int
        0 on success. Non-zero means the Cholesky factorization failed
        (matrix not positive definite) or the residual/solution sanity
        checks below failed -- in the original Fortran code this signals
        the caller to fall back to the eigenvalue-decomposition solver.
    """
    mat = np.asarray(mat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    n = mat.shape[0]

    input_mat = mat.copy()
    input_vec = vec.copy()

    try:
        # lower=False -> use the upper triangle, like CUBLAS_FILL_MODE_UPPER
        c, lower = cho_factor(mat, lower=False, check_finite=False)
        x = cho_solve((c, lower), vec, check_finite=False)
        info = 0
    except LinAlgError:
        # Matrix is not (numerically) positive definite: this is exactly
        # the case handled by "if (info == 0)" being false in the
        # Fortran code (cusolverDnDpotrf reporting a non-zero info).
        return np.zeros(n, dtype=np.float64), 1

    # --- sanity checks copied verbatim from the Fortran implementation ---
    residual = np.abs(input_mat @ x - input_vec)
    if residual.max() > _RESIDUAL_ERROR:
        info = 1
    if np.abs(x).max() > _ABS_ERROR:
        info = 1

    return x, info


def syevr_wrap(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Full symmetric eigen-decomposition of ``mat``.

    Equivalent of the Fortran ``syevr_wrap`` (cusolverDnDsyevd), performed
    on the CPU via LAPACK's DSYEVD (through ``scipy.linalg.eigh``).

    Only the upper triangle of ``mat`` is referenced (matches
    CUSOLVER_EIG_MODE_VECTOR + CUBLAS_FILL_MODE_UPPER).

    Parameters
    ----------
    mat : (n, n) ndarray, float64
        Symmetric matrix. Not modified.

    Returns
    -------
    eigval : (n,) ndarray, float64
        Eigenvalues in ascending order (same convention as DSYEVD).
    eigvec : (n, n) ndarray, float64
        Column ``eigvec[:, i]`` is the eigenvector for ``eigval[i]``.
        In the Fortran code the eigenvectors overwrite ``mat`` in place
        (``mat`` becomes the eigenvector matrix); here they are returned
        separately, and the caller (sfecalc.py) uses them the same way
        Fortran used the overwritten ``mat``.
    info : int
        0 on success, non-zero if the eigen-decomposition failed to
        converge (mirrors LAPACK's INFO output).
    """
    mat = np.asarray(mat, dtype=np.float64)
    try:
        eigval, eigvec = eigh(mat, lower=False, check_finite=False)
        info = 0
    except LinAlgError:
        n = mat.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64), 1

    return eigval, eigvec, info

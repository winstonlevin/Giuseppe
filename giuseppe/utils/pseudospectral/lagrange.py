"""
This script is borrowed from Dymos, which is licensed under:
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
[1] R. Falck, J. S. Gray, K. Ponnapalli, and T. Wright, "Dymos: A Python package for Optimal Control of
Multidisciplinary Systems," Journal of Open Source Software, 6(59), 2809 (2021).
DOI: https://doi.org/10.21105/joss.02809.
"""
from typing import Optional

import numpy as np


def lagrange_matrices(x_disc, x_interp, compute_interp_matrix=True, compute_diff_matrix=True):
    """
    Compute the lagrange matrices.

    The lagrange matrics are given 'discretization' nodes at which
    values are specified and 'interpolation' nodes at which values will be desired,
    returns interpolation and differentiation matrices which provide polynomial
    values and derivatives.

    Parameters
    ----------
    x_disc : np.array
        The cardinal nodes at which values of the variable are specified.
    x_interp : np.array
        The interior nodes at which interpolated values of the variable or its derivative
        are desired.
    compute_interp_matrix : bool
        If True, construct and return the interpolation matrix, otherwise return None.
    compute_diff_matrix : bool
        If True, construct and return the differentiation matrix, otherwise return None. The
        differentiation matrix can be prohibitively expensive to build as the number of
        discretization points grows large.

    Returns
    -------
    np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at x_disc, returns the intepolated values at x_interp.

    np.array or None
        A num_i x num_c matrix which, when post-multiplied by values specified
        at x_disc, returns the intepolated derivatives at x_interp. This is returned
        as None if compute_diff_matrix is None.
    """
    nd = len(x_disc)
    ni = len(x_interp)

    if compute_interp_matrix or compute_diff_matrix:
        temp = np.zeros((ni, nd))
        # Barycentric Weights
        diff = np.reshape(x_disc, (nd, 1)) - np.reshape(x_disc, (1, nd))
        np.fill_diagonal(diff, 1.0)
        wb = np.prod(1.0 / diff, axis=1)

    # Compute Li
    diff = np.reshape(x_interp, (ni, 1)) - np.reshape(x_disc, (1, nd))
    if compute_interp_matrix:
        Li = np.zeros((ni, nd))
        for j in range(nd):
            temp[:] = diff[:]
            temp[:, j] = 1.0
            Li[:, j] = wb[j] * np.prod(temp, axis=1)
    else:
        Li = None

    # Compute Di
    if compute_diff_matrix:
        Di = np.zeros((ni, nd))
        for j in range(nd):
            for k in range(nd):
                if k != j:
                    temp[:] = diff[:]
                    temp[:, j] = 1.0
                    temp[:, k] = 1.0
                    Di[:, j] += wb[j] * np.prod(temp, axis=1)
    else:
        Di = None

    return Li, Di


# -------------------------------------------------------------------------------------------------------------------- #
# Lagrange Integration Matrix                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #
def integration_matrix(x_disc: np.ndarray, x_interp: Optional[np.ndarray] = None):
    """
    A simple brute-force implementation of Lagrange integration matrices defined by x_disc, interpolated at x_interp
    """
    if x_interp is None:
        x_interp = x_disc

    n_in = x_disc.shape[0]
    n_out = x_interp.shape[0]

    # Lagrange polynomials, in power basis form
    denominators = x_disc[:, None] - x_disc[None, :]
    lagrange_polynomials = []
    idces = np.arange(0, n_in, 1)
    for idx in idces:
        idces_remaining = np.delete(idces, idx)
        lagrange_polynomials.append(np.polynomial.Polynomial.fromroots(x_disc[idces_remaining]))
        lagrange_polynomials[-1] /= np.prod(denominators[idx, idces_remaining])

    # The integration matrix is defined as:
    # Aij = int(Lj, -1, taui)
    # [See Eq. 66 in https://doi.org/10.1016/j.automatica.2010.06.048]
    int_mat = np.empty(shape=(n_out, n_in))
    for jdx, poly in enumerate(lagrange_polynomials):
        poly_int = poly.integ()
        int_mat[:, jdx] = poly_int(x_interp) - poly_int(-1.)
    int_mat[np.abs(int_mat) < 1E-12] = 0.  # Set near-zero values to exactly zero

    return int_mat

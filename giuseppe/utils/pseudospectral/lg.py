import numpy as np

# Caches the LG nodes and weights using (int) n as the key
_lg_cache = {}


def _lg(n):
    """
    Returns the Legendre-Gauss (LG) nodes and weights.

    The nodes are on the range [-1, 1].

    The LG nodes {xi} are the roots of the P(x|n), where P(x|n) is the nth Legendre polynomial
    The LG weights {wi} follow the formula:

                wi = 1/(n+1)**2 * (1 - xi) / L(xi|n)**2

    [1] C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Zang, "Spectral Methods in Fluid Dynamics," ch. 2.3.1, p. 61,
    Springer-Verlag 1988. DOI: https://doi.org/10.1007/978-3-642-84108-8.

    Parameters
    ----------
    n : int
        The number of LG nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the LG nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LG weights at the nodes in x.
    """
    legendre_coeffs = np.zeros(shape=(n,), dtype=float)
    legendre_coeffs[-1] = 1.
    legendre_poly = np.polynomial.legendre.Legendre(legendre_coeffs)
    x = np.empty(shape=(n,), dtype=float)
    x[0] = -1.
    x[1:] = legendre_poly.roots()
    if n % 2 == 0:
        # For an odd number of quadrature points, the middle point is exactly 0
        x[n // 2] = 0.

    den_l_sqrt = legendre_poly.deriv()(x[1:])
    w = 2 / ((1 - x[1:])*(1 + x[1:])*den_l_sqrt*den_l_sqrt)

    return x, w


def lg(n):
    """
    Retrieve the LG nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with LG nodes and weights.
    """
    if n not in _lg_cache:
        _lg_cache[n] = _lg(n)
    return _lg_cache[n]

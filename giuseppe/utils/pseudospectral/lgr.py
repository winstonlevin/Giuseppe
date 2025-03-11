import numpy as np

# Caches the LGR nodes and weights using (int) n as the key
_lgr_cache = {}


def _lgr(n):
    """
    Returns the Legendre-Gauss-Radau (LGR) nodes and weights.

    The nodes are on the range [-1, 1].

    The LGR nodes {xi} are the roots of the P(x|n) + P(x|n-1), where P(x|n) is the nth Legendre polynomial
    The LGR weights {wi} follow the formula:

                wi = 1/(n+1)**2 * (1 - xi) / L(xi|n)**2

    [1] C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Zang, "Spectral Methods in Fluid Dynamics," ch. 2.3.1, p. 61,
    Springer-Verlag 1988. DOI: https://doi.org/10.1007/978-3-642-84108-8.

    Parameters
    ----------
    n : int
        The number of LGR nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the LGR nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LGR weights at the nodes in x.
    """
    legendre_coeffs = np.zeros(shape=(n+1,), dtype=float)
    legendre_coeffs[-2:] = 1.
    legendre_poly = np.polynomial.legendre.Legendre(legendre_coeffs)
    x = legendre_poly.roots()

    legendre_coeffs[-2] = 0.
    legendre_poly = np.polynomial.legendre.Legendre(legendre_coeffs)
    den_sqrt = n * legendre_poly(x)
    w = (1 - x) / (den_sqrt*den_sqrt)

    return x, w


def lgr(n):
    """
    Retrieve the LGR nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with LGR nodes and weights.
    """
    if n not in _lgr_cache:
        _lgr_cache[n] = _lgr(n)
    return _lgr_cache[n]

import numpy as np

# Caches the CG nodes and weights using (int) n as the key
_cg_cache = {}


def _cg(n):
    """
    Returns the Chebyshev-Gauss (CG) nodes and weights.

    The nodes are on the range [-1, 1].

    The CG nodes {xi} and weight {wi} follow the formula:

                xi = cos(pi * i/n)
                wi = pi/n * sin(pi * i/n)**2 = pi/n * (1 - xi**2)

    [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature

    Parameters
    ----------
    n : int
        The number of CG nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the CG nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CG weights at the nodes in x.
    """
    x = np.cos(np.pi * np.linspace(1, 0, n))
    if n % 2 == 1:
        # The middle collocation value is exactly zero b/c xi = cos(pi), eliminate numerical error:
        x[n // 2] = 0.
    w = np.pi/n * (1 - x*x)
    return x, w


def cg(n):
    """
    Retrieve the CG nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with CG nodes and weights.
    """
    if n not in _cg_cache:
        _cg_cache[n] = _cg(n)
    return _cg_cache[n]

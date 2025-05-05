import numpy as np


def cc(n):
    """
    Returns the Clenshaw-Curtis (CC) nodes and weights.

    The nodes are on the range [-1, 1].

    The CC nodes {xi} are the extrema of the T(x|n), where T(x|n) is the nth Chebyshev polynomial.
    The CC weights {wi} follow the formula:

                wi = { 1/N   If |xi| == 1
                     { 2/N   Otherwise

    https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature

    Parameters
    ----------
    n : int
        The number of CC nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the CC nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CC weights at the nodes in x.
    """
    n1 = n - 1

    if n > 1:
        # Nominal case
        x = np.pi*np.arange(1, -1/n1, -1/n1)  # Equally spaced nodes, raises ZeroDivisionError if n = 1
        x[-1] = 0.  # Remove error from arange, raises IndexError if n < 1
        x[:] = np.cos(x)
        if (n % 2) != 0:
            # Odd number of collocation points -> middle is exactly zero
            x[n // 2] = 0.

        w = np.empty_like(x)
        w[:] = 1/n1
        w[1:-1] *= 2.
    elif n == 1:
        # Trivial case
        x = np.array((0.,))
        w = np.array((2.,))
    else:
        # n <= 0 (Not allowed)
        raise ValueError(f"Clenshaw-Curtis quadrature requires n > 0, but n = {n}!") from None

    return x, w

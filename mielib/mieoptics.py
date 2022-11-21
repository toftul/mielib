import scipy.special as sp
import numpy as np
from mielib import extraspecial


def optics_mie_a(n, k0a, eps_p, mu_p=1, eps_h=1, mu_h=1):
    """
        Electric Mie coefficent. For detatails see Bohren p. 100

        Arguments:
            n - 2^n multipole order
            k0a - vacuum size parameter
            eps_p, mu_p - particle parameters
            esp_h, mu_h - host parameters
    """
    n_p, n_h = np.sqrt(eps_p * mu_p, dtype=complex), np.sqrt(eps_h * mu_h, dtype=complex)
    x = n_h * k0a
    m = n_p / n_h
    mu = mu_p / mu_h

    mx = m * x
    jnmx = sp.spherical_jn(n, mx)
    jnx = sp.spherical_jn(n, x)
    h1nx = extraspecial.spherical_h1(n, x)
    xjnx_p = jnx + x * sp.spherical_jn(n, x, 1)
    mxjnmx_p = jnmx + mx * sp.spherical_jn(n, mx, 1)
    xh1nx_p = h1nx + x * extraspecial.spherical_h1(n, x, p=1)

    return (m**2 * jnmx * xjnx_p - mu * jnx * mxjnmx_p) / (m**2 * jnmx * xh1nx_p - mu * h1nx * mxjnmx_p)


def optics_mie_b(n, k0a, eps_p, mu_p=1, eps_h=1, mu_h=1):
    """
        Electric Mie coefficent. For detatails see Bohren p. 100

        Arguments:
            n - 2^n multipole order
            k0a - vacuum size parameter
            eps_p, mu_p - particle parameters
            esp_h, mu_h - host parameters
    """
    n_p, n_h = np.sqrt(eps_p * mu_p, dtype=complex), np.sqrt(eps_h * mu_h, dtype=complex)
    x = n_h * k0a
    m = n_p / n_h
    mu = mu_p / mu_h

    mx = m * x
    jnmx = sp.spherical_jn(n, mx)
    jnx = sp.spherical_jn(n, x)
    h1nx = extraspecial.spherical_h1(n, x)
    xjnx_p = jnx + x * sp.spherical_jn(n, x, 1)
    mxjnmx_p = jnmx + mx * sp.spherical_jn(n, mx, 1)
    xh1nx_p = h1nx + x * extraspecial.spherical_h1p(n, x, p=0)
    
    return (mu * jnmx * xjnx_p - jnx * mxjnmx_p) / (mu * jnmx * xh1nx_p - h1nx * mxjnmx_p)
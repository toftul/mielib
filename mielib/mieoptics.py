import scipy.special as sp
import numpy as np
from mielib import extraspecial


def optics_mie_a(n, k0a, eps_p, mu_p=1, eps_h=1, mu_h=1, get_denominator=False):
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

    if get_denominator:
        # for dispersion of TM modes
        return m**2 * jnmx * xh1nx_p - mu * h1nx * mxjnmx_p
    else:
        return (m**2 * jnmx * xjnx_p - mu * jnx * mxjnmx_p) / (m**2 * jnmx * xh1nx_p - mu * h1nx * mxjnmx_p)


def optics_mie_b(n, k0a, eps_p, mu_p=1, eps_h=1, mu_h=1, get_denominator=False):
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
    
    if get_denominator:
        # for dispertion of TE modes
        return mu * jnmx * xh1nx_p - h1nx * mxjnmx_p
    else:
        return (mu * jnmx * xjnx_p - jnx * mxjnmx_p) / (mu * jnmx * xh1nx_p - h1nx * mxjnmx_p)


def _host_refractive_index(eps_h, mu_h):
    """
        Refractive index of the host for the cross sections.
        The cross sections are not defined in a lossy host, because the scattered field decays.

        Arguments:
            eps_h, mu_h - host parameters
    """
    eps_mu_h = np.asarray(eps_h * mu_h)
    if np.any(np.imag(eps_mu_h) != 0) or np.any(np.real(eps_mu_h) <= 0):
        raise ValueError("The cross sections need a lossless host. Use a real and positive eps_h * mu_h.")
    return np.sqrt(np.real(eps_mu_h))


def optics_scattering_cross_section(k0, a, eps_p, mu_p=1, eps_h=1, mu_h=1, nmin=1, nmax=None, norm='none'):
    k0a = np.asarray(k0 * a)

    n_host = _host_refractive_index(eps_h, mu_h)
    nstop = extraspecial.wiscombe_nstop(n_host * k0a)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1

    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_sc   = np.zeros(k0a.size, dtype=np.float64)
    sigma_sc_n_electric = np.zeros([nmax, k0a.size], dtype=np.float64)
    sigma_sc_n_magnetic = np.zeros([nmax, k0a.size], dtype=np.float64)

    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, optics_mie_a(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
            bn = np.where(n <= nstop, optics_mie_b(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
        sigma_sc_n_electric[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * np.abs(an**2)
        sigma_sc_n_magnetic[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * np.abs(bn**2)
        
    sigma_sc = np.sum(sigma_sc_n_electric + sigma_sc_n_magnetic, axis=0)

    return sigma_sc/sigma_norm, sigma_sc_n_electric/sigma_norm, sigma_sc_n_magnetic/sigma_norm



def optics_extinction_cross_section(k0, a, eps_p, mu_p=1, eps_h=1, mu_h=1, nmin=1, nmax=None, norm='none'):
    k0a = np.asarray(k0 * a)

    n_host = _host_refractive_index(eps_h, mu_h)
    nstop = extraspecial.wiscombe_nstop(n_host * k0a)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1

    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_ext   = np.zeros(k0a.size, dtype=np.float64)
    sigma_ext_n_electric = np.zeros([nmax, k0a.size], dtype=np.float64)
    sigma_ext_n_magnetic = np.zeros([nmax, k0a.size], dtype=np.float64)

    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, optics_mie_a(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
            bn = np.where(n <= nstop, optics_mie_b(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
        sigma_ext_n_electric[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * np.real(an)
        sigma_ext_n_magnetic[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * np.real(bn)
        
    sigma_ext = np.sum(sigma_ext_n_electric + sigma_ext_n_magnetic, axis=0)

    return sigma_ext/sigma_norm, sigma_ext_n_electric/sigma_norm, sigma_ext_n_magnetic/sigma_norm


def optics_absorption_cross_section(k0, a, eps_p, mu_p=1, eps_h=1, mu_h=1, nmin=1, nmax=None, norm='none'):
    k0a = np.asarray(k0 * a)

    n_host = _host_refractive_index(eps_h, mu_h)
    nstop = extraspecial.wiscombe_nstop(n_host * k0a)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1

    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_abs   = np.zeros(k0a.size, dtype=np.float64)
    sigma_abs_n_electric = np.zeros([nmax, k0a.size], dtype=np.float64)
    sigma_abs_n_magnetic = np.zeros([nmax, k0a.size], dtype=np.float64)

    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, optics_mie_a(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
            bn = np.where(n <= nstop, optics_mie_b(n, k0a, eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h), 0)
        sigma_abs_n_electric[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * (np.real(an) - np.abs(an)**2)
        sigma_abs_n_magnetic[n, :] = 2*np.pi / (n_host * k0)**2 * (2*n+1) * (np.real(bn) - np.abs(bn)**2)
        
    sigma_abs = np.sum(sigma_abs_n_electric + sigma_abs_n_magnetic, axis=0)

    return sigma_abs/sigma_norm, sigma_abs_n_electric/sigma_norm, sigma_abs_n_magnetic/sigma_norm
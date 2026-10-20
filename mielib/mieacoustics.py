import numpy as np
import scipy.special as sp
from mielib import special

__all__ = [
    "acoustics_mie_a",
    "acoustics_mie_c",
    "acoustics_tmatrix",
    "acoustics_scattering_cross_section",
    "acoustics_pressure_cross_section",
    "acoustics_extinction_cross_section",
    "acoustics_absorption_cross_section",
]

def acoustics_mie_a(n, ka, rho1, beta1, get_denominator=False):
    """
        Acoustic Mie scattering coefficient, Eq. (4) in the Supplemental Material of
        I. D. Toftul et al., Phys. Rev. Lett. 123, 183901 (2019), arXiv:1905.12216
        Time dependence exp(-i omega t), incident pressure p = sum_n A_n j_n(k r) P_n(cos theta), A_n = A i^n (2n+1).
        The scattered pressure is p = sum_n A_n a_n h_n(k r) P_n(cos theta).

        n - multipole order
        ka - size parameter in host media
        rho1 - relative density
        beta1 - relative compressibility
    """
    gamma = np.sqrt(beta1/rho1)
    k1a = ka * np.sqrt(beta1*rho1)
    jn1 = sp.spherical_jn(n, k1a)
    jn = sp.spherical_jn(n, ka)
    jn1p = sp.spherical_jn(n, k1a, 1)
    jnp = sp.spherical_jn(n, ka, 1)
    hn = special.spherical_h1(n, ka)
    hnp = special.spherical_h1(n, ka, derivative=1)
    up = (gamma * jn1p * jn - jn1 * jnp)
    down = (jn1 * hnp - gamma * jn1p * hn)
    
    if get_denominator:
        return down
    else:
        return up/down

def acoustics_mie_c(n, ka, rho1, beta1, get_denominator=False):
    """
        Internal acoustic Mie coefficient, Eq. (4) in the Supplemental Material of
        I. D. Toftul et al., Phys. Rev. Lett. 123, 183901 (2019), arXiv:1905.12216
        Time dependence exp(-i omega t), incident pressure p = sum_n A_n j_n(k r) P_n(cos theta), A_n = A i^n (2n+1).
        The pressure inside the particle is p = sum_n A_n c_n j_n(k1 r) P_n(cos theta), k1 = k sqrt(rho1 beta1).

        n - multipole order
        ka - size parameter in host media
        rho1 - relative density
        beta1 - relative compressibility
    """
    gamma = np.sqrt(beta1/rho1)
    k1a = ka * np.sqrt(beta1*rho1)
    jn1 = sp.spherical_jn(n, k1a)
    jn1p = sp.spherical_jn(n, k1a, 1)
    hn = special.spherical_h1(n, ka)
    hnp = special.spherical_h1(n, ka, derivative=1)
    # the Wronskian j_n h_n' - j_n' h_n = i/(ka)^2
    up = 1j / (ka**2)
    down = (jn1 * hnp - gamma * jn1p * hn)
    
    if get_denominator:
        return down
    else:
        return up/down


def acoustics_tmatrix(n, ka, rho_rel, beta_rel):
    """
        Diagonal T-matrix of a sphere: the outgoing amplitude divided by the regular amplitude of the same multipole.
        In the expansion p = 𝒜 sum_jm C_jm z_j(k r) Y_jm(θ, φ),
        the scattered amplitudes (z_j = h_j) are C^sc_jm = T_j C^inc_jm, where the incident amplitudes have z_j = j_j.
        T does not depend on m.
        With the sign of acoustics_mie_a, T_n = a_n.
        The optical T_n has the same sign convention, but there T_n = -a_n, see optics_tmatrix.
        For a lossless sphere |1 + 2 T| = 1.

        n - multipole order
        ka - size parameter in host media
        rho_rel - relative density
        beta_rel - relative compressibility
    """
    return acoustics_mie_a(n, ka, rho_rel, beta_rel)
    
    
def acoustics_scattering_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=None, norm='none'):
    ka = np.asarray(a * k)
    nstop = special.wiscombe_jmax(ka)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2
    elif norm != 'none':
        raise ValueError(f"norm must be 'none' or 'geom', not {norm!r}")

    sigma_sc = np.zeros(ka.size, dtype=np.float64)
    sigma_sc_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, acoustics_mie_a(n, ka, rho_rel, beta_rel), 0)
        sigma_sc_n[n, :] = 4*np.pi / k**2 * (2*n+1) * np.abs(an)**2
        
    sigma_sc = np.sum(sigma_sc_n, axis=0)
    
    return sigma_sc/sigma_norm, sigma_sc_n/sigma_norm
    

def acoustics_pressure_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=None, norm='none'):
    ka = np.asarray(a * k)
    nstop = special.wiscombe_jmax(ka)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2
    elif norm != 'none':
        raise ValueError(f"norm must be 'none' or 'geom', not {norm!r}")

    sigma_pr = np.zeros(ka.size, dtype=np.float64)
    sigma_pr_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, acoustics_mie_a(n, ka, rho_rel, beta_rel), 0)
            an_plus1 = np.where(n+1 <= nstop, acoustics_mie_a(n+1, ka, rho_rel, beta_rel), 0)
        if n == nmax-1:
            # to ensure that pressure cross section is always positive
            sigma_pr_n[n, :] = -4*np.pi / k**2 * (2*n+1) * np.real(an)  
        else:
            sigma_pr_n[n, :] = -4*np.pi / k**2 * ((2*n+1) * np.real(an) + 2*(n+1) * np.real(np.conj(an) * an_plus1)) 
        
    sigma_pr = np.sum(sigma_pr_n, axis=0)
    
    return sigma_pr/sigma_norm, sigma_pr_n/sigma_norm


def acoustics_extinction_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=None, norm='none'):
    ka = np.asarray(a * k)
    nstop = special.wiscombe_jmax(ka)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2
    elif norm != 'none':
        raise ValueError(f"norm must be 'none' or 'geom', not {norm!r}")

    sigma_ext = np.zeros(ka.size, dtype=np.float64)
    sigma_ext_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, acoustics_mie_a(n, ka, rho_rel, beta_rel), 0)
        sigma_ext_n[n, :] = - 4*np.pi / k**2 * (2*n+1) * np.real(an)
        
    sigma_ext = np.sum(sigma_ext_n, axis=0)
    
    return sigma_ext/sigma_norm, sigma_ext_n/sigma_norm


def acoustics_absorption_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=None, norm='none'):
    ka = np.asarray(a * k)
    nstop = special.wiscombe_jmax(ka)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2
    elif norm != 'none':
        raise ValueError(f"norm must be 'none' or 'geom', not {norm!r}")

    sigma_abs = np.zeros(ka.size, dtype=np.float64)
    sigma_abs_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        # the terms above nstop are negligible, and they are nan when h1 overflows
        with np.errstate(invalid='ignore'):
            an = np.where(n <= nstop, acoustics_mie_a(n, ka, rho_rel, beta_rel), 0)
        sigma_abs_n[n, :] = - 4*np.pi / k**2 * (2*n+1) * (np.abs(an)**2 + np.real(an)) 
        
    sigma_abs = np.sum(sigma_abs_n, axis=0)
    
    return sigma_abs/sigma_norm, sigma_abs_n/sigma_norm

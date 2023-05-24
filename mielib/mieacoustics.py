import numpy as np
import scipy.special as sp
from mielib import extraspecial

def acoustics_mie_a(n, ka, rho1, beta1, get_denominator=False):
    """
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
    hn = extraspecial.spherical_h1(n, ka)
    hnp = extraspecial.spherical_h1(n, ka, p=1)
    up = (gamma * jn1p * jn - jn1 * jnp)
    down = (jn1 * hnp - gamma * jn1p * hn)
    
    if get_denominator:
        return down
    else:
        return up/down

def acoustics_mie_c(n, ka, rho1, beta1, get_denominator=False):
    """
        n - multipole order
        ka - size parameter in host media
        rho1 - relative density
        beta1 - relative compressibility
    """
    gamma = np.sqrt(beta1/rho1)
    k1a = ka * np.sqrt(beta1*rho1)
    jn1 = sp.spherical_jn(n, k1a)
    jn1p = sp.spherical_jn(n, k1a, 1)
    hn = extraspecial.spherical_h1(n, ka)
    hnp = extraspecial.spherical_h1(n, ka, p=1)
    up = 1j / (ka**2)
    down = (jn1 * hnp - gamma * jn1p * hn)
    
    if get_denominator:
        return down
    else:
        return up/down
    
    
def acoustics_scattering_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=50, norm='none'):
    ka = a * k
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_sc = np.zeros(ka.size, dtype=np.float64)
    sigma_sc_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        an = acoustics_mie_a(n, ka, rho_rel, beta_rel)
        sigma_sc_n[n, :] = 4*np.pi / k**2 * (2*n+1) * np.abs(an)**2
        
    sigma_sc = np.sum(sigma_sc_n, axis=0)
    
    return sigma_sc/sigma_norm, sigma_sc_n/sigma_norm
    

def acoustics_pressure_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=50, norm='none'):
    ka = a * k
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_pr = np.zeros(ka.size, dtype=np.float64)
    sigma_pr_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        an = mielib.acoustics_mie_a(n, ka, rho_rel, beta_rel)
        an_plus1 = mielib.acoustics_mie_a(n+1, ka, rho_rel, beta_rel)
        if n == nmax-1:
            # to ensure that pressure cross section is always positive
            sigma_pr_n[n, :] = -4*np.pi / k**2 * (2*n+1) * np.real(an)  
        else:
            sigma_pr_n[n, :] = -4*np.pi / k**2 * ((2*n+1) * np.real(an) + 2*(n+1) * np.real(np.conj(an) * an_plus1)) 
        
    sigma_pr = np.sum(sigma_pr_n, axis=0)
    
    return sigma_pr/sigma_norm, sigma_pr_n/sigma_norm


def acoustics_extinction_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=50, norm='none'):
    ka = a * k
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_ext = np.zeros(ka.size, dtype=np.float64)
    sigma_ext_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        an = acoustics_mie_a(n, ka, rho_rel, beta_rel)
        sigma_ext_n[n, :] = - 4*np.pi / k**2 * (2*n+1) * np.real(an)
        
    sigma_ext = np.sum(sigma_ext_n, axis=0)
    
    return sigma_ext/sigma_norm, sigma_ext_n/sigma_norm


def acoustics_absorption_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=50, norm='none'):
    ka = a * k
    
    sigma_norm = 1.0
    if norm == 'geom':
        sigma_norm = np.pi * a**2

    sigma_abs = np.zeros(ka.size, dtype=np.float64)
    sigma_abs_n = np.zeros([nmax, ka.size])
    
    for n in range(nmin, nmax):
        an = acoustics_mie_a(n, ka, rho_rel, beta_rel)
        sigma_abs_n[n, :] = - 4*np.pi / k**2 * (2*n+1) * (np.abs(an)**2 + np.real(an)) 
        
    sigma_abs = np.sum(sigma_abs_n, axis=0)
    
    return sigma_abs/sigma_norm, sigma_abs_n/sigma_norm

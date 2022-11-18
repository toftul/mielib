import numpy as np
import scipy.special as sp
from src import extraspecial

def acoustics_Mie_a(n, ka, rho1, beta1):
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
    extraspecial.spherical_h1(n, ka)
    hn = extraspecial.spherical_h1(n, ka)
    hnp = extraspecial.spherical_h1p(n, ka)
    up = (gamma * jn1p * jn - jn1 * jnp)
    down = (jn1 * hnp - gamma * jn1p * hn)
    
    ans = np.where(
        down == 0,
        0,
        up/down
    )
    return ans
    
    
def acoustics_scattering_cross_section(k, a, rho_rel, beta_rel, nmin=0, nmax=50):
    ka = a * k
    
    sigma_sc = np.zeros(ka.size, dtype=np.float64)
    sigma_sc_n = np.zeros([nmax+1 - nmin, ka.size])
    
    for n in range(nmin, nmax+1):
        an = acoustics_Mie_a(n, ka, rho_rel, beta_rel)
        sigma_sc_n[n, :] = 4*np.pi / k**2 * (2*n+1) * np.abs(an**2)
        
    sigma_sc = np.sum(sigma_sc_n, axis=0)
    
    return sigma_sc, sigma_sc_n
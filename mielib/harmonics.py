import numpy as np
import scipy.special as sp

from mielib import extraspecial

def spherical_radial_function_z(j, rho, p=0, superscript=1):
    """
        j - radial quantum number
        rho - argument
        p = 0,1 - derivative order
        superscript - type of radial function
    """
    zj = np.array([0.0j])
    match superscript:
        case 1:
            zj = sp.spherical_jn(j, rho, p)
        case 2:
            zj = sp.spherical_yn(j, rho, p)
        case 3:
            zj = extraspecial.spherical_h1(j, rho, p)
        case 4:
            zj = extraspecial.spherical_h2(j, rho, p)

    return zj

        
def vsh_bohren_me(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.17), (4.18)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho + 0.0)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    Mt = 0.0
    Mp = 0.0

    zn = spherical_radial_function_z(n, rho, p=0, superscript=superscript)

    Mt = -m / np.sin(theta) * np.sin(m * phi) * sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Mp = -np.cos(m * phi) * dPnm

    Mr = np.zeros(np.shape(Mp*zn))

    return np.array([Mr, Mt*zn, Mp*zn])


def vsh_bohren_mo(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.17), (4.18)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho + 0.0)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    Mt = 0.0
    Mp = 0.0

    zn = spherical_radial_function_z(n, rho, p=0, superscript=superscript)

    Mt = m / np.sin(theta) * np.cos(m * phi) * sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Mp = -np.sin(m * phi) * dPnm

    Mr = np.zeros(np.shape(Mp*zn))

    return np.array([Mr, Mt*zn, Mp*zn])


def vsh_bohren_ne(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.19), (4.20)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho + 0.0)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    Nr = 0.0
    Nt = 0.0
    Np = 0.0

    zn = spherical_radial_function_z(n, rho, p=0, superscript=superscript)
    znp = spherical_radial_function_z(n, rho, p=1, superscript=superscript)

    Pnm = sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Nr = zn/rho * np.cos(m*phi) * n * (n+1) * Pnm
    Nt = np.cos(m * phi) * dPnm * (zn/rho + znp)
    Np = -m * np.sin(m * phi) * Pnm / np.sin(theta) * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def vsh_bohren_no(m, n, rho, theta, phi, superscript=1):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.19), (4.20)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- azimuthal angle
            phi = [0, 2pi] -- polar angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.asarray(rho + 0.0)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    Nr = 0.0
    Nt = 0.0
    Np = 0.0

    zn = spherical_radial_function_z(n, rho, p=0, superscript=superscript)
    znp = spherical_radial_function_z(n, rho, p=1, superscript=superscript)

    Pnm = sp.lpmv(m, n, np.cos(theta))

    # formula for the d_θ Pnm(cos θ) is from here (I got a different sign though...)
    # https://math.stackexchange.com/questions/391672/derivative-of-associated-legendre-polynomials-at-x-pm-1
    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
    dPnm = 1/np.sin(theta) * (np.abs(np.sin(theta)) * sp.lpmv(m+1, n,
                                                              np.cos(theta)) + m * np.cos(theta) * sp.lpmv(m, n, np.cos(theta)))

    # an old way
    # dtheta = 1e-12
    # dPnm = (lpmv(m, n, np.cos(theta + dtheta)) - lpmv(m, n, np.cos(theta - dtheta))) / (2*dtheta)

    Nr = zn/rho * np.sin(m * phi) * n * (n+1) * Pnm
    Nt = np.sin(m * phi) * dPnm * (zn/rho + znp)
    Np = m * np.cos(m * phi) * Pnm / np.sin(theta) * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def vsh_jackson_x(m, n, theta, phi):
    '''
        from Jackson's book paragraph 9.7
    '''
    Ymn = sp.sph_harm(m, n, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm(m+1, n, phi, theta)

    return (-1)/np.sqrt(n*(n+1)) * np.array([
        np.zeros(np.shape(theta * phi)),
        m/np.sin(theta) * Ymn,
        1j * dYmn
    ])


def vsh_toftul_m(m, j, rho, theta, phi, superscript=3):
    """ 
        M vector spherical harmonic

        Arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
    """
    # convert input to np arrays
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj = spherical_radial_function_z(j, rho, p=0, superscript=superscript)

    Mt = 1j * m / np.sin(theta) * zj * sp.sph_harm(m, j, phi, theta)

    dYmn = m/np.tan(theta) * sp.sph_harm(m, j, phi, theta)
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm(m+1, j, phi, theta)

    Mp = -zj * dYmn

    Mr = np.zeros(np.shape(Mp))

    return np.array([Mr, Mt, Mp])


def vsh_toftul_n(m, j, rho, theta, phi, superscript=3):
    """ 
        N vector spherical harmonic

        Arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
    """
    # convert input to np arrays
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj  = spherical_radial_function_z(j, rho, p=0, superscript=superscript)
    zjp = spherical_radial_function_z(j, rho, p=1, superscript=superscript)

    Ymn = sp.sph_harm(m, j, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm(m+1, j, phi, theta)

    Nr = j*(j+1) * zj/rho * Ymn
    Nt = 1/rho * (zj + rho * zjp) * dYmn
    Np = 1j*m/np.sin(theta) * 1/rho * (zj + rho * zjp) * Ymn

    return np.array([Nr, Nt, Np])


def vsh_toftul_l(m, j, rho, theta, phi, superscript=3):
    """ 
        L vector spherical harmonic

        Arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
    """
    # convert input to np arrays
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj  = spherical_radial_function_z(j, rho, p=0, superscript=superscript)
    zjp = spherical_radial_function_z(j, rho, p=1, superscript=superscript)

    Ymn = sp.sph_harm(m, j, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= j:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm(m+1, j, phi, theta)

    Lr = zjp * Ymn
    Lt = zj/rho * dYmn
    Lp = 1j*m/np.sin(theta) * zj/rho * Ymn

    return np.array([Lr, Lt, Lp])


# wrapper functions
def vector_spherical_harmonic_m(m, j, rho, theta, phi, superscript=3, source='toftul', parity='even'):
    """ 
        vector spherical harmonic

        arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
            source:
                'toftul' - based on sm arxiv.org/abs/2210.04021
                'bohren' - based on bohren & huffmann book
            partiy (applicapble only for real vsh):
                'even' - even harmonic
                'odd' - odd harmonic
    """
    match source:
        case 'toftul':
            return vsh_toftul_m(m, j, rho, theta, phi, superscript=superscript)
        case 'bohren':
            match parity:
                case 'even':
                    return vsh_bohren_me(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case 'odd':
                    return vsh_bohren_mo(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case _:
                    return np.array([0.0, 0.0, 0.0])
        case _:
            return np.array([0.0, 0.0, 0.0])
     


def vector_spherical_harmonic_n(m, j, rho, theta, phi, superscript=3, source='toftul', parity='even'):
    """ 
        vector spherical harmonic

        arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
            source:
                'toftul' - based on sm arxiv.org/abs/2210.04021
                'bohren' - based on bohren & huffmann book
            partiy (applicapble only for real vsh):
                'even' - even harmonic
                'odd' - odd harmonic
    """
    match source:
        case 'toftul':
            return vsh_toftul_n(m, j, rho, theta, phi, superscript=superscript)
        case _:
            return np.array([0.0, 0.0, 0.0])
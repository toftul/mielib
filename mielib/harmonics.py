import numpy as np
import scipy.special as sp

from mielib import special

__all__ = [
    "vsh_bohren_me",
    "vsh_bohren_mo",
    "vsh_bohren_ne",
    "vsh_bohren_no",
    "vsh_jackson_x",
    "vsh_toftul_m",
    "vsh_toftul_n",
    "vsh_toftul_l",
    "vector_spherical_harmonic_m",
    "vector_spherical_harmonic_n",
    "vector_spherical_harmonic_l",
]

def _bohren_legendre(m, n, theta):
    """
        P_n^m(cos θ), m P_n^m(cos θ) / sin θ and d_θ P_n^m(cos θ) as in Bohren & Huffman,
        that is, without the Condon-Shortley phase (-1)^m.
        All three are finite on the z axis, θ = 0 and θ = π.

        With this P_n^m, the complex harmonics are W_mj = (-1)^m β_mj (W_emj + i W_omj), W = M, N,
        with β_mj = sqrt((2j+1)/(4π) (j-m)!/(j+m)!).
        Eq. (25) in the SM of arXiv:2210.04021 has no (-1)^m, because there the real harmonics use P_n^m with the phase.
    """
    # sph_legendre_p takes θ and not cos θ. Thus, it keeps the precision near the poles,
    # where cos θ = 1 in floating point for θ < 2^(-26.5) ≈ 1.05e-8.
    # sph_legendre_p has the Condon-Shortley phase and the normalization of Y_n^m.
    scale = (-1)**m * np.sqrt(4*np.pi / (2*n+1)) * np.exp(0.5 * (sp.gammaln(n+m+1) - sp.gammaln(n-m+1)))
    Pnm, dPnm = scale * sp.sph_legendre_p(n, m, theta, diff_n=1)

    sin_theta = np.sin(theta)
    on_axis = np.abs(sin_theta) < 1e-12
    # on the axis, P_n^m / sin θ -> d_θ P_n^m / cos θ (l'Hôpital's rule)
    pinm = m * np.where(on_axis, dPnm / np.cos(theta), Pnm / np.where(on_axis, 1, sin_theta))

    return Pnm, pinm, dPnm


def vsh_bohren_me(m, n, rho, theta, phi, superscript):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.17), (4.18)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- polar angle
            phi = [0, 2pi] -- azimuthal angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.array(rho, dtype=complex)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15

    zn = special.spherical_z(n, rho, superscript=superscript)
    _, pinm, dPnm = _bohren_legendre(m, n, theta)

    Mt = -np.sin(m * phi) * pinm
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
            theta = [0, pi] -- polar angle
            phi = [0, 2pi] -- azimuthal angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.array(rho, dtype=complex)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15

    zn = special.spherical_z(n, rho, superscript=superscript)
    _, pinm, dPnm = _bohren_legendre(m, n, theta)

    Mt = np.cos(m * phi) * pinm
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
            theta = [0, pi] -- polar angle
            phi = [0, 2pi] -- azimuthal angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.array(rho, dtype=complex)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15

    zn = special.spherical_z(n, rho, superscript=superscript)
    znp = special.spherical_z(n, rho, superscript=superscript, derivative=1)
    Pnm, pinm, dPnm = _bohren_legendre(m, n, theta)

    Nr = zn/rho * np.cos(m*phi) * n * (n+1) * Pnm
    Nt = np.cos(m * phi) * dPnm * (zn/rho + znp)
    Np = -np.sin(m * phi) * pinm * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def vsh_bohren_no(m, n, rho, theta, phi, superscript=1):
    '''
        vector Harmonics in spherical coordinates (r, θ, φ)
        from Bohren & Huffman, eq. (4.19), (4.20)
        Input
        -----
            rho = k r, k = k0 n_media
            theta = [0, pi] -- polar angle
            phi = [0, 2pi] -- azimuthal angle
            n = 0, 1, ...
            m = 0, ..., n
            superscript = 1, 2, 3, 4
    '''
    # convert input to np arrays
    rho = np.array(rho, dtype=complex)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15

    zn = special.spherical_z(n, rho, superscript=superscript)
    znp = special.spherical_z(n, rho, superscript=superscript, derivative=1)
    Pnm, pinm, dPnm = _bohren_legendre(m, n, theta)

    Nr = zn/rho * np.sin(m * phi) * n * (n+1) * Pnm
    Nt = np.sin(m * phi) * dPnm * (zn/rho + znp)
    Np = np.cos(m * phi) * pinm * (zn/rho + znp)

    return np.array([Nr, Nt, Np])


def vsh_jackson_x(m, n, theta, phi):
    '''
        from Jackson's book paragraph 9.7
    '''
    # convert input to np arrays
    theta = np.array(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    theta[np.abs(theta) < 1e-15] = 1e-15

    if n == 0:
        # L Y_00 = 0
        return np.zeros((3,) + np.shape(theta * phi), dtype=complex)

    Ymn = sp.sph_harm_y(n, m, theta, phi)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)
                                          ) * sp.sph_harm_y(n, m+1, theta, phi)

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
    rho = np.array(rho, dtype=complex)
    theta = np.array(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj = special.spherical_z(j, rho, superscript=superscript)

    Mt = 1j * m / np.sin(theta) * zj * sp.sph_harm_y(j, m, theta, phi)

    dYmn = m/np.tan(theta) * sp.sph_harm_y(j, m, theta, phi)
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm_y(j, m+1, theta, phi)

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
    rho = np.array(rho, dtype=complex)
    theta = np.array(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj  = special.spherical_z(j, rho, superscript=superscript)
    zjp = special.spherical_z(j, rho, superscript=superscript, derivative=1)

    Ymn = sp.sph_harm_y(j, m, theta, phi)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm_y(j, m+1, theta, phi)

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
    rho = np.array(rho, dtype=complex)
    theta = np.array(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # to prevent devision by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    theta[np.abs(theta) < 1e-15] = 1e-15

    zj  = special.spherical_z(j, rho, superscript=superscript)
    zjp = special.spherical_z(j, rho, superscript=superscript, derivative=1)

    Ymn = sp.sph_harm_y(j, m, theta, phi)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= j:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm_y(j, m+1, theta, phi)

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
                    raise ValueError(f"parity must be 'even' or 'odd', not {parity!r}")
        case _:
            raise ValueError(f"source must be 'toftul' or 'bohren', not {source!r}")
     


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
        case 'bohren':
            match parity:
                case 'even':
                    return vsh_bohren_ne(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case 'odd':
                    return vsh_bohren_no(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case _:
                    raise ValueError(f"parity must be 'even' or 'odd', not {parity!r}")
        case _:
            raise ValueError(f"source must be 'toftul' or 'bohren', not {source!r}")


def vector_spherical_harmonic_l(m, j, rho, theta, phi, superscript=3, source='toftul', parity='even'):
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
            return vsh_toftul_l(m, j, rho, theta, phi, superscript=superscript)
        case _:
            raise ValueError(f"source must be 'toftul' for the L harmonic, not {source!r}")

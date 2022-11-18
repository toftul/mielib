import numpy as np
import scipy.special as sp

from mielib import extraspecial


def VSH_bohren_Me_mn(m, n, rho, theta, phi, superscript):
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

    zn = np.asarray(0.0)
    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extraspecial.spherical_h1(n, rho)
    )
    """
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yv(n, rho)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

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


def VSH_bohren_Mo_mn(m, n, rho, theta, phi, superscript):
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

    zn = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extraspecial.spherical_h1(n, rho))
    """
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

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


def VSH_bohren_Ne(m, n, rho, theta, phi, superscript):
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

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extraspecial.spherical_h1(n, rho))
    znp = np.where(superscript == 1, sp.spherical_jn(
        n, rho, 1), extraspecial.spherical_h1p(n, rho))

    """
    zn = 0
    znp = 0
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
        znp = spherical_jn(n, rho, 1)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
        znp = spherical_yn(n, rho, 1)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
        znp = spherical_h1p(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
        znp = spherical_h2p(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

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


def VSH_bohren_No(m, n, rho, theta, phi, superscript=1):
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

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)
    zn = np.where(superscript == 1, sp.spherical_jn(
        n, rho), extraspecial.spherical_h1(n, rho))
    znp = np.where(superscript == 1, sp.spherical_jn(
        n, rho, 1), extraspecial.spherical_h1p(n, rho))

    """
    zn = 0
    znp = 0
    if superscript == 1:
        # zn = jn -- spherical Bessel 1st kind
        zn = spherical_jn(n, rho)
        znp = spherical_jn(n, rho, 1)
    elif superscript == 2:
        # zn = yn -- sherical Bessel 2nd kind
        zn = spherical_yn(n, rho)
        znp = spherical_yn(n, rho, 1)
    elif superscript == 3:
        # zn = h(1) -- spherical Hankel 1st
        zn = spherical_h1(n, rho)
        znp = spherical_h1p(n, rho)
    elif superscript == 4:
        # zn = h(2) -- spherical Hankel 2nd
        zn = spherical_h2(n, rho)
        znp = spherical_h2p(n, rho)
    else:
        print("ERROR: Superscript invalid!")
    """

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


def VSH_jackson_X(m, n, theta, phi):
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


def VSH_toftul_M(m, j, rho, theta, phi, superscript=3):
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

    zn = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(j, rho),
        extraspecial.spherical_h1(j, rho)
    )

    Mt = 1j * m / np.sin(theta) * zn * sp.sph_harm(m, j, phi, theta)

    dYmn = m/np.tan(theta) * sp.sph_harm(m, j, phi, theta)
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm(m+1, j, phi, theta)

    Mp = -zn * dYmn

    Mr = np.zeros(np.shape(Mp))

    return np.array([Mr, Mt, Mp])


def VSH_toftul_N(m, j, rho, theta, phi, superscript=3):
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

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(j, rho),
        extraspecial.spherical_h1(j, rho)
    )
    znp = np.where(
        superscript == 1,
        sp.spherical_jn(j, rho, 1),
        extraspecial.spherical_h1p(j, rho)
    )

    Ymn = sp.sph_harm(m, j, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= j:
        # sqrt((-m + j) (1 + m + j)) = np.sqrt(sp.gamma(1-m+j)) * np.sqrt(sp.gamma(2+m+j)) / (np.sqrt(sp.gamma(-m+j)) * np.sqrt(sp.gamma(1+m+j)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + j)*(1 + m + j)) * sp.sph_harm(m+1, j, phi, theta)

    Nr = j*(j+1) * zn/rho * Ymn
    Nt = 1/rho * (zn + rho * znp) * dYmn
    Np = 1j*m/np.sin(theta) * 1/rho * (zn + rho * znp) * Ymn

    return np.array([Nr, Nt, Np])


def VSH_toftul_L(m, j, rho, theta, phi, superscript=3):
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

    zn = np.asarray(0.0)
    znp = np.asarray(0.0)

    zn = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho),
        extraspecial.spherical_h1(n, rho)
    )
    znp = np.where(
        superscript == 1,
        sp.spherical_jn(n, rho, 1),
        extraspecial.spherical_h1p(n, rho)
    )

    Ymn = sp.sph_harm(m, n, phi, theta)
    dYmn = m/np.tan(theta) * Ymn
    if m+1 <= n:
        # sqrt((-m + n) (1 + m + n)) = np.sqrt(sp.gamma(1-m+n)) * np.sqrt(sp.gamma(2+m+n)) / (np.sqrt(sp.gamma(-m+n)) * np.sqrt(sp.gamma(1+m+n)))
        dYmn += np.exp(-1j*phi) * np.sqrt((-m + n)*(1 + m + n)) * sp.sph_harm(m+1, n, phi, theta)

    Lr = znp * Ymn
    Lt = zn/rho * dYmn
    Lp = 1j*m/np.sin(theta) * zn/rho * Ymn

    return np.array([Lr, Lt, Lp])


# wrapper functions
def VSH_M(m, j, rho, theta, phi, superscript=3, source='toftul', parity='even'):
    """ 
        M vector spherical harmonic

        Arguments:
            m - projection of total angular momentum 
            j - total angular momentum
            rho, theta, phi - arguments in spherical coordinate system
            superscript:
                1 - spherical bessel
                3 - spherical hankel1
            source:
                'toftul' - based on SM arxiv.org/abs/2210.04021
                'bohren' - based on Bohren & Huffmann book
            partiy:
                'even' - even harmonic
                'odd' - odd harmonic
    """
    match book:
        case 'toftul':
            return VSH_toftul_M(m, j, rho, theta, phi, superscript=superscript)
        case 'bohren':
            match parity:
                case 'even':
                    return VSH_bohren_Me(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case 'odd':
                    return VSH_bohren_Mo(m=m, n=j, rho=rho, theta=theta, phi=phi, superscript=superscript)
                case _:
                    return np.array([0.0, 0.0, 0.0])
        case _:
            return np.array([0.0, 0.0, 0.0])
     


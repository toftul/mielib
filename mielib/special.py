import numpy as np
import scipy.special as sp

__all__ = [
    "spherical_jn",
    "spherical_yn",
    "spherical_h1",
    "spherical_h2",
    "spherical_z",
    "riccati_psi",
    "riccati_xi",
    "wiscombe_jmax",
]


def _check_derivative(derivative):
    if derivative not in (0, 1, 2):
        raise ValueError(f"derivative must be 0, 1 or 2, not {derivative!r}")


def _second_derivative(f, j, z):
    """
        z_j'' = [j z_{j-1}' - (j+1) z_{j+1}'] / (2j+1) for each spherical Bessel function z_j.
        For j = 0 the first term has zero weight, and np.maximum avoids the nan from order -1.
    """
    return j/(2*j+1) * f(np.maximum(j-1, 0), z, True) - (j+1)/(2*j+1) * f(j+1, z, True)


def spherical_jn(j, z, *, derivative=0):
    """
        Spherical Bessel function j_j(z) of the first kind, or its first or second derivative.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    _check_derivative(derivative)
    if derivative == 2:
        return _second_derivative(sp.spherical_jn, j, z)
    return sp.spherical_jn(j, z, derivative == 1)


def spherical_yn(j, z, *, derivative=0):
    """
        Spherical Bessel function y_j(z) of the second kind, or its first or second derivative.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    _check_derivative(derivative)
    if derivative == 2:
        return _second_derivative(sp.spherical_yn, j, z)
    return sp.spherical_yn(j, z, derivative == 1)


def spherical_h1(j, z, *, derivative=0):
    """
        Spherical Hankel function h_j^(1)(z) = j_j(z) + i y_j(z), or its first or second derivative.
        With the time dependence exp(-i omega t), h_j^(1)(kr) is an outgoing wave.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    return spherical_jn(j, z, derivative=derivative) + 1j * spherical_yn(j, z, derivative=derivative)


def spherical_h2(j, z, *, derivative=0):
    """
        Spherical Hankel function h_j^(2)(z) = j_j(z) - i y_j(z), or its first or second derivative.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    return spherical_jn(j, z, derivative=derivative) - 1j * spherical_yn(j, z, derivative=derivative)


def spherical_z(j, rho, *, superscript, derivative=0):
    """
        Radial function z_j^(s)(rho) of the vector spherical harmonics.

        Arguments:
            j - order
            rho - argument, rho = k r
            superscript:
                1 - spherical Bessel j_j
                2 - spherical Bessel y_j
                3 - spherical Hankel h_j^(1)
                4 - spherical Hankel h_j^(2)
            derivative - 0, 1 or 2
    """
    match superscript:
        case 1:
            return spherical_jn(j, rho, derivative=derivative)
        case 2:
            return spherical_yn(j, rho, derivative=derivative)
        case 3:
            return spherical_h1(j, rho, derivative=derivative)
        case 4:
            return spherical_h2(j, rho, derivative=derivative)
        case _:
            raise ValueError(f"superscript must be 1, 2, 3 or 4, not {superscript!r}")


def riccati_psi(j, z, *, derivative=0):
    """
        Riccati-Bessel function psi_j(z) = z j_j(z), or its first or second derivative.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    _check_derivative(derivative)
    match derivative:
        case 0:
            return z * spherical_jn(j, z)
        case 1:
            return spherical_jn(j, z) + z * spherical_jn(j, z, derivative=1)
        case 2:
            return 2 * spherical_jn(j, z, derivative=1) + z * spherical_jn(j, z, derivative=2)


def riccati_xi(j, z, *, derivative=0):
    """
        Riccati-Bessel function xi_j(z) = z h_j^(1)(z), or its first or second derivative.

        Arguments:
            j - order
            z - argument, real or complex
            derivative - 0, 1 or 2
    """
    _check_derivative(derivative)
    match derivative:
        case 0:
            return z * spherical_h1(j, z)
        case 1:
            return spherical_h1(j, z) + z * spherical_h1(j, z, derivative=1)
        case 2:
            return 2 * spherical_h1(j, z, derivative=1) + z * spherical_h1(j, z, derivative=2)


def wiscombe_jmax(x):
    """
        Highest multipole order for a converged Mie series.
        Wiscombe, Appl. Opt. 19, 1505 (1980).

        Arguments:
            x - size parameter in the host medium
    """
    x = np.abs(x)
    return np.floor(x + 4.05 * np.cbrt(x) + 2).astype(int)

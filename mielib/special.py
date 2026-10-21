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


def _log_derivative_jn(jmax, z):
    """
        E_j(z) = j_j'(z) / j_j(z) for the orders j = 0, ..., jmax on axis 0, by the downward recurrence
            E_{j-1} = (j-1)/z - 1 / (E_j + (j+1)/z)
        from E_j = j/z, the limit for large j. Wiscombe, Appl. Opt. 19, 1505 (1980).
        The recurrence is stable for any complex z, also where j_j(z) overflows.
        The error of the start decays to the round-off before j = jmax for |z| up to 3000 at least.
    """
    z = np.asarray(z)
    z = z.astype(np.result_type(z, float))
    size = np.max(np.abs(z), initial=0)
    jstart = int(max(jmax, size) + 10 * np.cbrt(size)) + 16

    E = np.empty((jmax + 1,) + z.shape, dtype=z.dtype)
    e = jstart / z
    for j in range(jstart, 0, -1):
        e = (j - 1) / z - 1 / (e + (j + 1) / z)
        if j <= jmax + 1:
            E[j - 1] = e
    return E


def _spherical_orders(jmax, x):
    """
        j_j(x), j_j'(x), h_j(x) and h_j'(x), with h_j = h_j^(1), for j = 0, 1, ..., jmax in turn and a real x.
        y_j comes from the upward recurrence, which is stable because y_j grows with j,
        and j_j from the Wronskian j_j y_j' - j_j' y_j = 1/x^2 with j_j' = E_j j_j:
            j_j = 1 / [x^2 (y_j' - E_j y_j)].
        The upward recurrence for j_j itself loses the precision for j > x.
        One order at a time keeps the memory small for a large grid of x.
    """
    x = np.asarray(x, dtype=float)
    E = _log_derivative_jn(jmax, x)
    # y_{-1} = j_0 and y_0
    y_previous, y = np.sin(x) / x, -np.cos(x) / x
    for j in range(jmax + 1):
        if j > 0:
            y_previous, y = y, (2*j - 1) / x * y - y_previous
        dy = y_previous - (j + 1) / x * y
        jn = 1 / (x**2 * (dy - E[j] * y))
        djn = E[j] * jn
        yield jn, djn, jn + 1j * y, djn + 1j * dy

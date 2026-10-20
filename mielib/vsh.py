"""
    Vector spherical harmonics in spherical components (r, θ, φ).

    The complex harmonics are
        M_jm = ∇ × (r z_j(ρ) Y_jm),   N_jm = (1/k) ∇ × M_jm,   L_jm = (1/k) ∇ (z_j(ρ) Y_jm),
    with ρ = k r and Y_jm with the Condon-Shortley phase, as in the SM of arXiv:2210.04021.
    The real harmonics M_real and N_real are the even and odd harmonics of Bohren & Huffman, Eqs. (4.17)-(4.20).
"""
import numpy as np
import scipy.special as sp

from mielib import special

__all__ = [
    "M",
    "N",
    "L",
    "X",
    "M_real",
    "N_real",
]


def _radial_input(rho):
    # the copy keeps the array of the caller unchanged
    rho = np.array(rho, dtype=complex)
    # to prevent division by zero
    rho[np.abs(rho) < 1e-15] = 1e-15
    return rho


def _complex_angles(theta, phi):
    # the copy keeps the array of the caller unchanged
    theta = np.array(theta, dtype=float)
    # to prevent division by zero; sph_harm_y takes θ, so the limits on the z axis are correct
    theta[np.abs(theta) < 1e-15] = 1e-15
    return theta, np.asarray(phi, dtype=float)


def _sph_harm_and_dtheta(j, m, theta, phi):
    """Y_jm(θ, φ) and d_θ Y_jm(θ, φ)."""
    Y = sp.sph_harm_y(j, m, theta, phi)
    dY = m / np.tan(theta) * Y
    if m + 1 <= j:
        dY = dY + np.exp(-1j*phi) * np.sqrt((j - m) * (j + m + 1)) * sp.sph_harm_y(j, m + 1, theta, phi)
    return Y, dY


def M(j, m, rho, theta, phi, *, superscript):
    """
        Complex vector spherical harmonic M_jm = ∇ × (r z_j Y_jm).

        Arguments:
            j - total angular momentum
            m - projection of the total angular momentum
            rho, theta, phi - spherical coordinates, rho = k r
            superscript - radial function z_j, see special.spherical_z
    """
    rho = _radial_input(rho)
    theta, phi = _complex_angles(theta, phi)
    z = special.spherical_z(j, rho, superscript=superscript)
    Y, dY = _sph_harm_and_dtheta(j, m, theta, phi)

    Mt = 1j * m / np.sin(theta) * z * Y
    Mp = -z * dY
    Mr = np.zeros(np.shape(Mp))

    return np.array([Mr, Mt, Mp])


def N(j, m, rho, theta, phi, *, superscript):
    """
        Complex vector spherical harmonic N_jm = (1/k) ∇ × M_jm.

        Arguments:
            j - total angular momentum
            m - projection of the total angular momentum
            rho, theta, phi - spherical coordinates, rho = k r
            superscript - radial function z_j, see special.spherical_z
    """
    rho = _radial_input(rho)
    theta, phi = _complex_angles(theta, phi)
    z = special.spherical_z(j, rho, superscript=superscript)
    zp = special.spherical_z(j, rho, superscript=superscript, derivative=1)
    Y, dY = _sph_harm_and_dtheta(j, m, theta, phi)

    # (rho z)' / rho
    d = z / rho + zp
    Nr = j * (j + 1) * z / rho * Y
    Nt = d * dY
    Np = 1j * m / np.sin(theta) * d * Y

    return np.array([Nr, Nt, Np])


def L(j, m, rho, theta, phi, *, superscript):
    """
        Complex vector spherical harmonic L_jm = (1/k) ∇ (z_j Y_jm).

        Arguments:
            j - total angular momentum
            m - projection of the total angular momentum
            rho, theta, phi - spherical coordinates, rho = k r
            superscript - radial function z_j, see special.spherical_z
    """
    rho = _radial_input(rho)
    theta, phi = _complex_angles(theta, phi)
    z = special.spherical_z(j, rho, superscript=superscript)
    zp = special.spherical_z(j, rho, superscript=superscript, derivative=1)
    Y, dY = _sph_harm_and_dtheta(j, m, theta, phi)

    Lr = zp * Y
    Lt = z / rho * dY
    Lp = 1j * m / np.sin(theta) * z / rho * Y

    return np.array([Lr, Lt, Lp])


def X(j, m, theta, phi):
    """
        Vector spherical harmonic X_jm = L Y_jm / sqrt(j (j+1)) of Jackson, Sec. 9.7, with L = -i r × ∇.
        M_jm = -i sqrt(j (j+1)) z_j X_jm.

        Arguments:
            j - total angular momentum
            m - projection of the total angular momentum
            theta, phi - spherical angles
    """
    theta, phi = _complex_angles(theta, phi)

    if j == 0:
        # L Y_00 = 0
        return np.zeros((3,) + np.shape(theta * phi), dtype=complex)

    Y, dY = _sph_harm_and_dtheta(j, m, theta, phi)

    return -1 / np.sqrt(j * (j + 1)) * np.array([
        np.zeros(np.shape(theta * phi)),
        m / np.sin(theta) * Y,
        1j * dY,
    ])


def _bohren_legendre(j, m, theta):
    """
        P_j^m(cos θ), m P_j^m(cos θ) / sin θ and d_θ P_j^m(cos θ) as in Bohren & Huffman,
        that is, without the Condon-Shortley phase (-1)^m.
        All three are finite on the z axis, θ = 0 and θ = π.

        With this P_j^m, the complex harmonics are W_jm = (-1)^m β_jm (W_jm^even + i W_jm^odd), W = M, N,
        with β_jm = sqrt((2j+1)/(4π) (j-m)!/(j+m)!).
        Eq. (25) in the SM of arXiv:2210.04021 has no (-1)^m, because there the real harmonics use P_j^m with the phase.
    """
    # sph_legendre_p takes θ and not cos θ. Thus, it keeps the precision near the poles,
    # where cos θ = 1 in floating point for θ < 2^(-26.5) ≈ 1.05e-8.
    # sph_legendre_p has the Condon-Shortley phase and the normalization of Y_j^m.
    scale = (-1)**m * np.sqrt(4*np.pi / (2*j+1)) * np.exp(0.5 * (sp.gammaln(j+m+1) - sp.gammaln(j-m+1)))
    P, dP = scale * sp.sph_legendre_p(j, m, theta, diff_n=1)

    sin_theta = np.sin(theta)
    on_axis = np.abs(sin_theta) < 1e-12
    # on the axis, P_j^m / sin θ -> d_θ P_j^m / cos θ (l'Hôpital's rule)
    pi = m * np.where(on_axis, dP / np.cos(theta), P / np.where(on_axis, 1, sin_theta))

    return P, pi, dP


def _parity_factors(m, phi, parity):
    """u = cos(mφ) or sin(mφ), and v = (1/m) d_φ u."""
    match parity:
        case 'even':
            return np.cos(m * phi), -np.sin(m * phi)
        case 'odd':
            return np.sin(m * phi), np.cos(m * phi)
        case _:
            raise ValueError(f"parity must be 'even' or 'odd', not {parity!r}")


def M_real(j, m, rho, theta, phi, *, superscript, parity):
    """
        Real vector spherical harmonic M_emj or M_omj of Bohren & Huffman, Eqs. (4.17) and (4.18), m >= 0.

        Arguments:
            j - total angular momentum (n in Bohren & Huffman)
            m - projection of the total angular momentum, m = 0, ..., j
            rho, theta, phi - spherical coordinates, rho = k r
            superscript - radial function z_j, see special.spherical_z
            parity - 'even' or 'odd'
    """
    rho = _radial_input(rho)
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    u, v = _parity_factors(m, phi, parity)
    z = special.spherical_z(j, rho, superscript=superscript)
    _, pi, dP = _bohren_legendre(j, m, theta)

    Mt = v * pi * z
    Mp = -u * dP * z
    Mr = np.zeros(np.shape(Mp))

    return np.array([Mr, Mt, Mp])


def N_real(j, m, rho, theta, phi, *, superscript, parity):
    """
        Real vector spherical harmonic N_emj or N_omj of Bohren & Huffman, Eqs. (4.19) and (4.20), m >= 0.

        Arguments:
            j - total angular momentum (n in Bohren & Huffman)
            m - projection of the total angular momentum, m = 0, ..., j
            rho, theta, phi - spherical coordinates, rho = k r
            superscript - radial function z_j, see special.spherical_z
            parity - 'even' or 'odd'
    """
    rho = _radial_input(rho)
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    u, v = _parity_factors(m, phi, parity)
    z = special.spherical_z(j, rho, superscript=superscript)
    zp = special.spherical_z(j, rho, superscript=superscript, derivative=1)
    P, pi, dP = _bohren_legendre(j, m, theta)

    # (rho z)' / rho
    d = z / rho + zp
    Nr = j * (j + 1) * z / rho * u * P
    Nt = u * dP * d
    Np = v * pi * d

    return np.array([Nr, Nt, Np])

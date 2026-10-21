"""
    Mie theory of a sphere in optics, with the time dependence exp(-i omega t).

    The field is the dual-symmetric pair (sqrt(eps) E, sqrt(mu) H) = 𝒜 (e, h) with
        e = sum_jm (A_jm N_jm + B_jm M_jm),   h = -i sum_jm (B_jm N_jm + A_jm M_jm),
    where A_jm (B_jm) are the electric (magnetic) amplitudes, see mielib.vsh.
    All the functions take the size parameter x = k a in the host and the parameters of the particle relative to the host,
    eps_rel = eps_p / eps_h and mu_rel = mu_p / mu_h. The electromagnetic duality is the swap eps_rel <-> mu_rel:
    it swaps a_j and b_j, the two denominators and the two internal amplitudes.
"""
from dataclasses import dataclass

import numpy as np

from mielib import special

__all__ = [
    "mie_a",
    "mie_b",
    "mie_c",
    "mie_d",
    "denominator_a",
    "denominator_b",
    "internal_amplitudes",
    "tmatrix",
    "cross_sections",
    "CrossSections",
]


def _parts(j, x, eps_rel, mu_rel):
    """
        The numerators and denominators of a_j and b_j, m and sqrt(eps_rel).
        With (s1, s2) = (sqrt(eps_rel), sqrt(mu_rel)) for a_j and (sqrt(mu_rel), sqrt(eps_rel)) for b_j,
            numerator   = s1 ψ_j(mx) ψ_j'(x) - s2 ψ_j(x) ψ_j'(mx),
            denominator = s1 ψ_j(mx) ξ_j'(x) - s2 ξ_j(x) ψ_j'(mx),
        and m = sqrt(eps_rel) sqrt(mu_rel), so k_p a = m x.
    """
    sqrt_eps = np.sqrt(np.asarray(eps_rel, dtype=complex))
    sqrt_mu = np.sqrt(np.asarray(mu_rel, dtype=complex))
    m = sqrt_eps * sqrt_mu

    mx = m * x
    psi_mx, dpsi_mx = special.riccati_psi(j, mx), special.riccati_psi(j, mx, derivative=1)
    psi_x, dpsi_x = special.riccati_psi(j, x), special.riccati_psi(j, x, derivative=1)
    xi_x, dxi_x = special.riccati_xi(j, x), special.riccati_xi(j, x, derivative=1)

    def numerator_denominator(s1, s2):
        return s1 * psi_mx * dpsi_x - s2 * psi_x * dpsi_mx, s1 * psi_mx * dxi_x - s2 * xi_x * dpsi_mx

    return numerator_denominator(sqrt_eps, sqrt_mu), numerator_denominator(sqrt_mu, sqrt_eps), m, sqrt_eps


def mie_a(j, x, eps_rel, mu_rel=1):
    """
        Electric Mie coefficient a_j, Bohren & Huffman, Ch. 4.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    (numerator, denominator), _, _, _ = _parts(j, x, eps_rel, mu_rel)
    return numerator / denominator


def mie_b(j, x, eps_rel, mu_rel=1):
    """
        Magnetic Mie coefficient b_j, Bohren & Huffman, Ch. 4. b_j is a_j with eps_rel <-> mu_rel.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    _, (numerator, denominator), _, _ = _parts(j, x, eps_rel, mu_rel)
    return numerator / denominator


def mie_c(j, x, eps_rel, mu_rel=1):
    """
        Internal magnetic Mie coefficient c_j, Bohren & Huffman, Ch. 4.
        The field inside the particle is E = sum_j E_j (c_j M_o1j - i d_j N_e1j) with the radial function j_j(m k r).
        Bohren & Huffman measure the inside field in E, so c_j = B^in / (sqrt(eps_rel) B^inc), see internal_amplitudes.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    _, (_, denominator), m, sqrt_eps = _parts(j, x, eps_rel, mu_rel)
    return 1j * m**2 / (sqrt_eps * denominator)


def mie_d(j, x, eps_rel, mu_rel=1):
    """
        Internal electric Mie coefficient d_j, Bohren & Huffman, Ch. 4.
        The field inside the particle is E = sum_j E_j (c_j M_o1j - i d_j N_e1j) with the radial function j_j(m k r).
        Bohren & Huffman measure the inside field in E, so d_j = A^in / (sqrt(eps_rel) A^inc), see internal_amplitudes.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    (_, denominator), _, m, sqrt_eps = _parts(j, x, eps_rel, mu_rel)
    return 1j * m**2 / (sqrt_eps * denominator)


def denominator_a(j, x, eps_rel, mu_rel=1):
    """
        Denominator D^a_j of a_j, d_j and A^in / A^inc:
            D^a_j = sqrt(eps_rel) ψ_j(mx) ξ_j'(x) - sqrt(mu_rel) ξ_j(x) ψ_j'(mx).
        Its zeros in the complex x plane are the TM modes.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    (_, denominator), _, _, _ = _parts(j, x, eps_rel, mu_rel)
    return denominator


def denominator_b(j, x, eps_rel, mu_rel=1):
    """
        Denominator D^b_j of b_j, c_j and B^in / B^inc. D^b_j is D^a_j with eps_rel <-> mu_rel:
            D^b_j = sqrt(mu_rel) ψ_j(mx) ξ_j'(x) - sqrt(eps_rel) ξ_j(x) ψ_j'(mx).
        Its zeros in the complex x plane are the TE modes.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
    """
    _, (_, denominator), _, _ = _parts(j, x, eps_rel, mu_rel)
    return denominator


def internal_amplitudes(j, x, eps_rel, mu_rel=1):
    """
        Amplitudes inside the particle divided by the incident amplitudes of the same multipole,
        A^in_jm / A^inc_jm = i m^2 / D^a_j and B^in_jm / B^inc_jm = i m^2 / D^b_j,
        with the expansion of the module docstring and the radial function j_j(m k r) inside.
        Inside, the field is (sqrt(eps_p) E, sqrt(mu_p) H) / 𝒜 with the same 𝒜 as outside.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host

        Returns:
            electric and magnetic amplitude ratios
    """
    (_, denominator_a), (_, denominator_b), m, _ = _parts(j, x, eps_rel, mu_rel)
    return 1j * m**2 / denominator_a, 1j * m**2 / denominator_b


def tmatrix(j, x, eps_rel, mu_rel=1):
    """
        Diagonal T-matrix of a sphere: the outgoing amplitude divided by the regular amplitude of the same multipole,
        A^sc_jm = T^e_j A^inc_jm and B^sc_jm = T^m_j B^inc_jm, with the radial functions h_j^(1) and j_j.
        With the sign of Bohren & Huffman for a_j and b_j, T^e_j = -a_j and T^m_j = -b_j.
        The acoustic T-matrix has the same sign convention, but there T_j = a_j, see mielib.acoustics.tmatrix.
        For a lossless sphere |1 + 2 T| = 1.

        Arguments:
            j - multipole order, the total angular momentum
            x - size parameter k a in the host
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host

        Returns:
            T^e_j, T^m_j
    """
    (numerator_a, denominator_a), (numerator_b, denominator_b), _, _ = _parts(j, x, eps_rel, mu_rel)
    return -numerator_a / denominator_a, -numerator_b / denominator_b


@dataclass(frozen=True)
class CrossSections:
    """
        Cross sections of a sphere for a plane wave.
        The totals have the broadcast shape of the input. The arrays with the suffix _e (_m) have the shape
        (jmax+1,) + that shape, and row j is the electric (magnetic) part of the multipole j.

            sca, ext, abs - scattering, extinction and absorption
            back - backscattering, 4π times the differential scattering cross section at θ = π
            pr - radiation pressure, ext - g sca
            g - asymmetry parameter <cos θ>
    """
    sca: np.ndarray
    ext: np.ndarray
    abs: np.ndarray
    back: np.ndarray
    pr: np.ndarray
    g: np.ndarray
    sca_e: np.ndarray
    sca_m: np.ndarray
    ext_e: np.ndarray
    ext_m: np.ndarray
    abs_e: np.ndarray
    abs_m: np.ndarray


def cross_sections(k, a, eps_rel, mu_rel=1, *, jmin=1, jmax=None, norm='none'):
    """
        Cross sections of a sphere for a plane wave, Bohren & Huffman, Ch. 4.
        The cross sections are not defined in a lossy host, because the scattered field decays.

        Arguments:
            k - wavenumber in the host, real and positive
            a - radius
            eps_rel, mu_rel - permittivity and permeability of the particle relative to the host
            jmin, jmax - the sums include the orders jmin <= j <= jmax and also j <= special.wiscombe_jmax(k a)
                at each point. For jmax=None, jmax is the largest special.wiscombe_jmax(k a) of the input.
            norm - 'none', or 'geom' to divide all the cross sections by π a^2

        Returns:
            CrossSections
    """
    if norm not in ('none', 'geom'):
        raise ValueError(f"norm must be 'none' or 'geom', not {norm!r}")
    k = np.asarray(k)
    if np.any(np.imag(k) != 0) or np.any(np.real(k) <= 0):
        raise ValueError("The cross sections need a lossless host. Use a real and positive k.")
    k, a = np.real(k), np.asarray(a)

    shape = np.broadcast_shapes(k.shape, a.shape, np.shape(eps_rel), np.shape(mu_rel))
    x = np.broadcast_to(k * a, shape)
    jstop = special.wiscombe_jmax(x)
    if jmax is None:
        jmax = int(np.max(jstop))
    j = np.arange(jmax + 1).reshape((-1,) + (1,) * len(shape))

    # the terms above jstop are negligible, and they are nan when h_j overflows
    with np.errstate(all='ignore'):
        (numerator_a, denominator_a), (numerator_b, denominator_b), _, _ = _parts(j, x, eps_rel, mu_rel)
        use = (j >= max(jmin, 1)) & (j <= jstop)
        an = np.where(use, numerator_a / denominator_a, 0)
        bn = np.where(use, numerator_b / denominator_b, 0)

    w = 2 * j + 1
    prefactor = 2 * np.pi / k**2
    sca_e, sca_m = prefactor * w * np.abs(an)**2, prefactor * w * np.abs(bn)**2
    ext_e, ext_m = prefactor * w * np.real(an), prefactor * w * np.real(bn)
    sca, ext = np.sum(sca_e + sca_m, axis=0), np.sum(ext_e + ext_m, axis=0)

    back = np.pi / k**2 * np.abs(np.sum(w * (-1)**j * (an - bn), axis=0))**2
    # g sca, the sums start at j = 1
    jj = j[1:-1]
    g_sca = 2 * prefactor * (
        np.sum(jj * (jj + 2) / (jj + 1) * np.real(an[1:-1] * np.conj(an[2:]) + bn[1:-1] * np.conj(bn[2:])), axis=0)
        + np.sum((2 * j[1:] + 1) / (j[1:] * (j[1:] + 1)) * np.real(an[1:] * np.conj(bn[1:])), axis=0))
    g = np.divide(g_sca, sca, out=np.zeros(np.shape(sca)), where=sca > 0)
    # [()] gives a NumPy scalar for a scalar input, as np.sum gives for the other totals
    g = g[()]

    sigma_norm = np.pi * a**2 if norm == 'geom' else 1.0
    return CrossSections(
        sca=sca / sigma_norm, ext=ext / sigma_norm, abs=(ext - sca) / sigma_norm,
        back=back / sigma_norm, pr=(ext - g_sca) / sigma_norm, g=g,
        sca_e=sca_e / sigma_norm, sca_m=sca_m / sigma_norm,
        ext_e=ext_e / sigma_norm, ext_m=ext_m / sigma_norm,
        abs_e=(ext_e - sca_e) / sigma_norm, abs_m=(ext_m - sca_m) / sigma_norm,
    )

"""
    Mie theory of a fluid sphere in acoustics, with the time dependence exp(-i omega t).
    I. D. Toftul et al., Phys. Rev. Lett. 123, 183901 (2019), Supplemental Material, arXiv:1905.12216.

    The incident pressure is p = sum_j A_j j_j(k r) P_j(cos θ) with A_j = A i^j (2j+1),
    the scattered pressure is p = sum_j A_j a_j h_j(k r) P_j(cos θ),
    and the pressure inside the particle is p = sum_j A_j c_j j_j(k_p r) P_j(cos θ), k_p = k sqrt(rho_rel beta_rel).
    All the functions take the size parameter x = k a in the host and the density and the compressibility
    of the particle relative to the host, rho_rel and beta_rel.
"""
from dataclasses import dataclass

import numpy as np

from mielib import special

__all__ = [
    "mie_a",
    "mie_c",
    "denominator",
    "tmatrix",
    "cross_sections",
    "CrossSections",
]


def _parts(j, x, rho_rel, beta_rel):
    """
        The numerator and the denominator of a_j, with sqrt(rho_rel) and k_p a = sqrt(rho_rel) sqrt(beta_rel) x:
            numerator   = sqrt(beta_rel) j_j'(k_p a) j_j(x) - sqrt(rho_rel) j_j(k_p a) j_j'(x),
            denominator = sqrt(rho_rel) j_j(k_p a) h_j'(x) - sqrt(beta_rel) j_j'(k_p a) h_j(x).
    """
    sqrt_rho = np.sqrt(np.asarray(rho_rel, dtype=complex))
    sqrt_beta = np.sqrt(np.asarray(beta_rel, dtype=complex))

    xp = sqrt_rho * sqrt_beta * x
    jp, djp = special.spherical_jn(j, xp), special.spherical_jn(j, xp, derivative=1)
    jx, djx = special.spherical_jn(j, x), special.spherical_jn(j, x, derivative=1)
    hx, dhx = special.spherical_h1(j, x), special.spherical_h1(j, x, derivative=1)

    numerator = sqrt_beta * djp * jx - sqrt_rho * jp * djx
    denominator = sqrt_rho * jp * dhx - sqrt_beta * djp * hx

    return numerator, denominator, sqrt_rho


def mie_a(j, x, rho_rel, beta_rel):
    """
        Acoustic Mie scattering coefficient a_j, Eq. (4) in the Supplemental Material of
        Phys. Rev. Lett. 123, 183901 (2019).

        Arguments:
            j - multipole order
            x - size parameter k a in the host
            rho_rel - density of the particle relative to the host
            beta_rel - compressibility of the particle relative to the host
    """
    numerator, denominator, _ = _parts(j, x, rho_rel, beta_rel)
    return numerator / denominator


def mie_c(j, x, rho_rel, beta_rel):
    """
        Internal acoustic Mie coefficient c_j, Eq. (4) in the Supplemental Material of
        Phys. Rev. Lett. 123, 183901 (2019).

        Arguments:
            j - multipole order
            x - size parameter k a in the host
            rho_rel - density of the particle relative to the host
            beta_rel - compressibility of the particle relative to the host
    """
    _, denominator, sqrt_rho = _parts(j, x, rho_rel, beta_rel)
    # the numerator is sqrt(rho_rel) times the Wronskian j_j h_j' - j_j' h_j = i / x^2
    return 1j * sqrt_rho / (x**2 * denominator)


def denominator(j, x, rho_rel, beta_rel):
    """
        Denominator of a_j and c_j:
            D_j = sqrt(rho_rel) j_j(k_p a) h_j'(x) - sqrt(beta_rel) j_j'(k_p a) h_j(x),
        with k_p a = sqrt(rho_rel) sqrt(beta_rel) x. Its zeros in the complex x plane are the modes of the sphere.

        Arguments:
            j - multipole order
            x - size parameter k a in the host
            rho_rel - density of the particle relative to the host
            beta_rel - compressibility of the particle relative to the host
    """
    _, denominator, _ = _parts(j, x, rho_rel, beta_rel)
    return denominator


def tmatrix(j, x, rho_rel, beta_rel):
    """
        Diagonal T-matrix of a sphere: the outgoing amplitude divided by the regular amplitude of the same multipole.
        In the expansion p = 𝒜 sum_jm C_jm z_j(k r) Y_jm(θ, φ),
        the scattered amplitudes (z_j = h_j) are C^sc_jm = T_j C^inc_jm, where the incident amplitudes have z_j = j_j.
        With the sign of mie_a, T_j = a_j.
        The optical T-matrix has the same sign convention, but there T_j = -a_j, see mielib.optics.tmatrix.
        For a lossless sphere |1 + 2 T| = 1.

        Arguments:
            j - multipole order
            x - size parameter k a in the host
            rho_rel - density of the particle relative to the host
            beta_rel - compressibility of the particle relative to the host
    """
    return mie_a(j, x, rho_rel, beta_rel)


@dataclass(frozen=True)
class CrossSections:
    """
        Cross sections of a sphere for a plane wave.
        The totals have the broadcast shape of the input. The arrays with the suffix _j have the shape
        (jmax+1,) + that shape, and row j is the part of the multipole j.

            sca, ext, abs - scattering, extinction and absorption
            pr - radiation pressure, ext - g sca
            back - backscattering, 4π times the differential scattering cross section at θ = π
            g - asymmetry parameter <cos θ>
    """
    sca: np.ndarray
    ext: np.ndarray
    abs: np.ndarray
    pr: np.ndarray
    back: np.ndarray
    g: np.ndarray
    sca_j: np.ndarray
    ext_j: np.ndarray
    abs_j: np.ndarray


def cross_sections(k, a, rho_rel, beta_rel, *, jmin=0, jmax=None, norm='none'):
    """
        Cross sections of a sphere for a plane wave.
        The cross sections are not defined in a lossy host, because the scattered field decays.

        Arguments:
            k - wavenumber in the host, real and positive
            a - radius
            rho_rel, beta_rel - density and compressibility of the particle relative to the host
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

    shape = np.broadcast_shapes(k.shape, a.shape, np.shape(rho_rel), np.shape(beta_rel))
    x = np.broadcast_to(k * a, shape)
    jstop = special.wiscombe_jmax(x)
    if jmax is None:
        jmax = int(np.max(jstop))
    j = np.arange(jmax + 1).reshape((-1,) + (1,) * len(shape))

    # the terms above jstop are negligible, and they are nan when h_j overflows
    with np.errstate(all='ignore'):
        numerator, denominator, _ = _parts(j, x, rho_rel, beta_rel)
        an = np.where((j >= jmin) & (j <= jstop), numerator / denominator, 0)

    w = 2 * j + 1
    prefactor = 4 * np.pi / k**2
    sca_j = prefactor * w * np.abs(an)**2
    ext_j = -prefactor * w * np.real(an)
    sca, ext = np.sum(sca_j, axis=0), np.sum(ext_j, axis=0)

    back = prefactor * np.abs(np.sum(w * (-1)**j * an, axis=0))**2
    # g sca = integral of cos θ |f|^2 over the directions
    g_sca = prefactor * np.sum(2 * (j[:-1] + 1) * np.real(an[:-1] * np.conj(an[1:])), axis=0)
    g = np.divide(g_sca, sca, out=np.zeros(np.shape(sca)), where=sca > 0)
    # [()] gives a NumPy scalar for a scalar input, as np.sum gives for the other totals
    g = g[()]

    sigma_norm = np.pi * a**2 if norm == 'geom' else 1.0
    return CrossSections(
        sca=sca / sigma_norm, ext=ext / sigma_norm, abs=(ext - sca) / sigma_norm,
        pr=(ext - g_sca) / sigma_norm, back=back / sigma_norm, g=g,
        sca_j=sca_j / sigma_norm, ext_j=ext_j / sigma_norm, abs_j=(ext_j - sca_j) / sigma_norm,
    )

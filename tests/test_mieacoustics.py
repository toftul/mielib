import numpy as np
import pytest
import scipy.special as sp

import mielib

CROSS_SECTIONS = [
    mielib.acoustics_scattering_cross_section,
    mielib.acoustics_pressure_cross_section,
    mielib.acoustics_extinction_cross_section,
    mielib.acoustics_absorption_cross_section,
]


@pytest.mark.parametrize("rho_rel, beta_rel", [(2.0, 0.5), (0.3, 3.0)])
def test_scattering_matches_rayleigh_limit(rho_rel, beta_rel):
    ka = 0.01
    f1 = 1 - beta_rel
    f2 = 2 * (rho_rel - 1) / (2 * rho_rel + 1)
    sigma_rayleigh = 4 * np.pi * ka**4 * (f1**2 / 9 + f2**2 / 12)

    sigma_sc = mielib.acoustics_scattering_cross_section(np.array([ka]), 1.0, rho_rel, beta_rel)[0]

    np.testing.assert_allclose(sigma_sc, sigma_rayleigh, rtol=1e-3)


def test_default_series_is_converged():
    ka = np.array([20.0])
    rho_rel, beta_rel = 2.0 + 0.1j, 0.5

    n = np.arange(0, 80)
    an = mielib.acoustics_mie_a(n, ka, rho_rel, beta_rel)
    sigma_sc_direct = 4 * np.pi / ka**2 * np.sum((2 * n + 1) * np.abs(an)**2)

    sigma_sc = mielib.acoustics_scattering_cross_section(ka, 1.0, rho_rel, beta_rel)[0]

    np.testing.assert_allclose(sigma_sc, sigma_sc_direct, rtol=1e-12)


@pytest.mark.parametrize("nmax", [None, 300])
@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_one_call_covers_small_and_large_spheres(cross_section, nmax):
    k = np.array([0.01, 0.1, 1.0, 60.0])

    one_call = cross_section(k, 1.0, 2.0 + 0.1j, 0.5, nmax=nmax)[0]
    separate = [cross_section(np.array([ki]), 1.0, 2.0 + 0.1j, 0.5)[0][0] for ki in k]

    assert np.all(np.isfinite(one_call))
    np.testing.assert_allclose(one_call, separate, rtol=1e-12)


@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_scalar_k_gives_the_same_result_as_an_array(cross_section):
    scalar = cross_section(1.5, 1.0, 2.0 + 0.1j, 0.5)
    array = cross_section(np.array([1.5]), 1.0, 2.0 + 0.1j, 0.5)

    for s, a in zip(scalar, array):
        np.testing.assert_allclose(s, a, rtol=1e-14)


@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_unknown_norm_raises(cross_section):
    with pytest.raises(ValueError, match="norm"):
        cross_section(np.array([1.0]), 1.0, 2.0, 0.5, norm='geometric')


@pytest.mark.parametrize("ka, rho_rel, beta_rel", [
    (1.3, 2.0, 0.5),
    (0.4, 0.3, 3.0),
    (2.5, 3 + 0.1j, 2.0),
    (0.05, 1.2e-3, 1.5e4),
    (1.0 - 0.1j, 4.0, 0.7),
])
def test_coefficients_satisfy_boundary_conditions(ka, rho_rel, beta_rel):
    """Eq. (3) in the Supplemental Material of arXiv:1905.12216: the pressure and the normal velocity are continuous at r = a."""
    n = np.arange(0, 7)
    k1a = ka * np.sqrt(rho_rel * beta_rel)
    delta = np.sqrt(beta_rel / rho_rel)

    a = mielib.acoustics_mie_a(n, ka, rho_rel, beta_rel)
    c = mielib.acoustics_mie_c(n, ka, rho_rel, beta_rel)

    # p_in + p_sc = p_part
    np.testing.assert_allclose(
        sp.spherical_jn(n, ka) + a * mielib.spherical_h1(n, ka),
        c * sp.spherical_jn(n, k1a), rtol=1e-12)
    # (1/rho) dp/dr outside = (1/rho_1) dp/dr inside, and k_1 / (k rho_rel) = delta
    np.testing.assert_allclose(
        sp.spherical_jn(n, ka, 1) + a * mielib.spherical_h1(n, ka, p=1),
        delta * c * sp.spherical_jn(n, k1a, 1), rtol=1e-12)


def test_internal_coefficient_shares_the_denominator():
    n = np.arange(0, 5)
    params = dict(ka=1.7, rho1=2.0 + 0.1j, beta1=0.5, get_denominator=True)

    np.testing.assert_allclose(mielib.acoustics_mie_c(n, **params), mielib.acoustics_mie_a(n, **params))


@pytest.mark.parametrize("rho_rel, beta_rel, lossless", [
    (2.5, 0.4, True),
    (0.3, 3.0, True),
    (2.0 + 0.1j, 0.5, False),
    (0.3, 3.0 + 0.2j, False),
])
def test_tmatrix_conserves_the_flux(rho_rel, beta_rel, lossless):
    """j_n = (h_n^(1) + h_n^(2)) / 2, so the outgoing wave has the amplitude 1 + 2T: |1 + 2T| = 1 without loss and < 1 with loss."""
    t = mielib.acoustics_tmatrix(np.arange(0, 5), 1.3, rho_rel, beta_rel)

    if lossless:
        np.testing.assert_allclose(np.abs(1 + 2 * t), 1, rtol=1e-12)
    else:
        assert np.all(np.abs(1 + 2 * t) < 1)


def test_cross_sections_from_the_tmatrix():
    """sigma_ext = -(4 pi / k^2) sum (2n+1) Re T, with the same sign as in optics."""
    k, rho_rel, beta_rel = 1.3, 2.0 + 0.1j, 0.5
    # the same terms as in the cross-section functions
    n = np.arange(0, mielib.wiscombe_nstop(k) + 1)
    t = mielib.acoustics_tmatrix(n, k, rho_rel, beta_rel)

    sigma_ext = -4 * np.pi / k**2 * np.sum((2 * n + 1) * np.real(t))
    sigma_sc = 4 * np.pi / k**2 * np.sum((2 * n + 1) * np.abs(t)**2)

    np.testing.assert_allclose(sigma_ext, mielib.acoustics_extinction_cross_section(k, 1.0, rho_rel, beta_rel)[0], rtol=1e-12)
    np.testing.assert_allclose(sigma_sc, mielib.acoustics_scattering_cross_section(k, 1.0, rho_rel, beta_rel)[0], rtol=1e-12)

import numpy as np
import pytest

from mielib import acoustics, special


@pytest.mark.parametrize("rho_rel, beta_rel", [(2.0, 0.5), (0.3, 3.0)])
def test_scattering_matches_rayleigh_limit(rho_rel, beta_rel):
    ka = 0.01
    f1 = 1 - beta_rel
    f2 = 2 * (rho_rel - 1) / (2 * rho_rel + 1)
    sigma_rayleigh = 4 * np.pi * ka**4 * (f1**2 / 9 + f2**2 / 12)

    np.testing.assert_allclose(acoustics.cross_sections(ka, 1.0, rho_rel, beta_rel).sca, sigma_rayleigh, rtol=1e-3)


def test_default_series_is_converged():
    ka, rho_rel, beta_rel = 20.0, 2.0 + 0.1j, 0.5
    j = np.arange(0, 80)
    sigma_sc_direct = 4 * np.pi / ka**2 * np.sum((2 * j + 1) * np.abs(acoustics.mie_a(j, ka, rho_rel, beta_rel))**2)

    np.testing.assert_allclose(acoustics.cross_sections(ka, 1.0, rho_rel, beta_rel).sca, sigma_sc_direct, rtol=1e-12)


@pytest.mark.parametrize("jmax", [None, 300])
def test_one_call_covers_small_and_large_spheres(jmax):
    k = np.array([0.01, 0.1, 1.0, 60.0])

    one_call = acoustics.cross_sections(k, 1.0, 2.0 + 0.1j, 0.5, jmax=jmax)
    separate = [acoustics.cross_sections(ki, 1.0, 2.0 + 0.1j, 0.5) for ki in k]

    for name in ['sca', 'ext', 'abs', 'pr', 'back', 'g']:
        values = getattr(one_call, name)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, [getattr(s, name) for s in separate], rtol=1e-12, atol=1e-30)


def test_pressure_cross_section_has_the_cross_terms():
    """sigma_pr = -(4 pi / k^2) sum [(2j+1) Re a_j + 2 (j+1) Re(a_j a*_{j+1})] = ext - g sca."""
    k, rho_rel, beta_rel = 1.7, 2.0 + 0.1j, 0.5
    j = np.arange(0, special.wiscombe_jmax(k) + 1)
    a = acoustics.mie_a(j, k, rho_rel, beta_rel)
    sigma_pr = -4 * np.pi / k**2 * (np.sum((2 * j + 1) * np.real(a)) + np.sum(2 * (j[:-1] + 1) * np.real(a[:-1] * np.conj(a[1:]))))

    q = acoustics.cross_sections(k, 1.0, rho_rel, beta_rel)
    np.testing.assert_allclose(q.pr, sigma_pr, rtol=1e-12)
    np.testing.assert_allclose(q.pr, q.ext - q.g * q.sca, rtol=1e-12)


def test_unknown_norm_and_lossy_host_raise():
    with pytest.raises(ValueError, match="norm"):
        acoustics.cross_sections(1.0, 1.0, 2.0, 0.5, norm='geometric')
    with pytest.raises(ValueError, match="lossless host"):
        acoustics.cross_sections(1.0 + 0.1j, 1.0, 2.0, 0.5)


@pytest.mark.parametrize("ka, rho_rel, beta_rel", [
    (1.3, 2.0, 0.5),
    (0.4, 0.3, 3.0),
    (2.5, 3 + 0.1j, 2.0),
    (0.05, 1.2e-3, 1.5e4),
    (1.0 - 0.1j, 4.0, 0.7),
])
def test_coefficients_satisfy_boundary_conditions(ka, rho_rel, beta_rel):
    """Eq. (3) in the Supplemental Material of arXiv:1905.12216: the pressure and the normal velocity are continuous at r = a."""
    j = np.arange(0, 7)
    k1a = ka * np.sqrt(rho_rel) * np.sqrt(beta_rel)
    delta = np.sqrt(beta_rel / rho_rel)

    a = acoustics.mie_a(j, ka, rho_rel, beta_rel)
    c = acoustics.mie_c(j, ka, rho_rel, beta_rel)

    np.testing.assert_allclose(special.spherical_jn(j, ka) + a * special.spherical_h1(j, ka),
                               c * special.spherical_jn(j, k1a), rtol=1e-12)
    np.testing.assert_allclose(special.spherical_jn(j, ka, derivative=1) + a * special.spherical_h1(j, ka, derivative=1),
                               delta * c * special.spherical_jn(j, k1a, derivative=1), rtol=1e-12)


def test_internal_coefficient_and_denominator():
    """c_j D_j = i sqrt(rho_rel) / x^2: the numerator of c_j is the Wronskian."""
    j, x, rho_rel, beta_rel = np.arange(0, 5), 1.7, 2.0 + 0.1j, 0.5
    np.testing.assert_allclose(acoustics.mie_c(j, x, rho_rel, beta_rel) * acoustics.denominator(j, x, rho_rel, beta_rel),
                               1j * np.sqrt(rho_rel) / x**2, rtol=1e-13)


@pytest.mark.parametrize("rho_rel, beta_rel, lossless", [
    (2.5, 0.4, True), (0.3, 3.0, True), (2.0 + 0.1j, 0.5, False), (0.3, 3.0 + 0.2j, False),
])
def test_tmatrix_conserves_the_flux(rho_rel, beta_rel, lossless):
    """j_j = (h_j^(1) + h_j^(2)) / 2, so the outgoing wave has the amplitude 1 + 2T: |1 + 2T| = 1 without loss and < 1 with loss."""
    t = acoustics.tmatrix(np.arange(0, 5), 1.3, rho_rel, beta_rel)
    if lossless:
        np.testing.assert_allclose(np.abs(1 + 2 * t), 1, rtol=1e-12)
    else:
        assert np.all(np.abs(1 + 2 * t) < 1)


def test_cross_sections_from_the_tmatrix():
    """sigma_ext = -(4 pi / k^2) sum (2j+1) Re T, with the same sign as in optics."""
    k, rho_rel, beta_rel = 1.3, 2.0 + 0.1j, 0.5
    j = np.arange(0, special.wiscombe_jmax(k) + 1)
    t = acoustics.tmatrix(j, k, rho_rel, beta_rel)

    q = acoustics.cross_sections(k, 1.0, rho_rel, beta_rel)
    np.testing.assert_allclose(-4 * np.pi / k**2 * np.sum((2 * j + 1) * np.real(t)), q.ext, rtol=1e-12)
    np.testing.assert_allclose(4 * np.pi / k**2 * np.sum((2 * j + 1) * np.abs(t)**2), q.sca, rtol=1e-12)

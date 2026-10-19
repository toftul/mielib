import numpy as np
import pytest

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

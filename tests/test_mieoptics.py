import numpy as np
import pytest
import scipy.special as sp

import mielib

# Q_sca and Q_ext from miepython 3.3.0.
# miepython uses m = n - i kappa. mielib uses exp(-i omega t), so m = n + i kappa.
MIEPYTHON = [
    # m, x, Q_sca, Q_ext
    (1.55, 5.213, 3.1049959150801865, 3.1049959150801865),
    (1.5, 60.0, 2.137281954969279, 2.137281954969279),
    (1.33, 150.0, 2.1143767571353997, 2.1143767571353997),
    (1.5 + 0.01j, 10.0, 2.344131626959545, 2.7706950637975383),
    (0.2 + 3.0j, 5.0, 2.9829374120236696, 3.1913392682413173),
    (4.0 + 0.01j, 50.0, 1.505664155858015, 2.137765425666939),
]


@pytest.mark.parametrize("m, x, q_sca, q_ext", MIEPYTHON)
def test_efficiencies_match_miepython(m, x, q_sca, q_ext):
    k0 = np.array([x])

    q_sca_mielib = mielib.optics_scattering_cross_section(k0, 1.0, eps_p=m**2, norm='geom')[0]
    q_ext_mielib = mielib.optics_extinction_cross_section(k0, 1.0, eps_p=m**2, norm='geom')[0]
    q_abs_mielib = mielib.optics_absorption_cross_section(k0, 1.0, eps_p=m**2, norm='geom')[0]

    np.testing.assert_allclose(q_sca_mielib, q_sca, rtol=1e-9)
    np.testing.assert_allclose(q_ext_mielib, q_ext, rtol=1e-9)
    np.testing.assert_allclose(q_abs_mielib, q_ext - q_sca, atol=1e-9)


@pytest.mark.parametrize("eps_p", [2.25, 4 + 0.5j])
def test_scattering_matches_rayleigh_limit(eps_p):
    x = 0.01
    q_rayleigh = 8 / 3 * x**4 * abs((eps_p - 1) / (eps_p + 2))**2

    q_sca = mielib.optics_scattering_cross_section(np.array([x]), 1.0, eps_p=eps_p, norm='geom')[0]

    np.testing.assert_allclose(q_sca, q_rayleigh, rtol=1e-4)


@pytest.mark.parametrize("nmax", [None, 300])
def test_one_call_covers_small_and_large_spheres(nmax):
    k0 = np.array([0.01, 0.1, 1.0, 60.0])

    one_call = mielib.optics_scattering_cross_section(k0, 1.0, eps_p=2.25, nmax=nmax)[0]
    separate = [mielib.optics_scattering_cross_section(np.array([x]), 1.0, eps_p=2.25)[0][0] for x in k0]

    assert np.all(np.isfinite(one_call))
    np.testing.assert_allclose(one_call, separate, rtol=1e-12)


CROSS_SECTIONS = [
    mielib.optics_scattering_cross_section,
    mielib.optics_extinction_cross_section,
    mielib.optics_absorption_cross_section,
]


@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_lossy_host_raises(cross_section):
    with pytest.raises(ValueError, match="lossless host"):
        cross_section(np.array([1.0]), 1.0, eps_p=4.0, eps_h=1.5 + 0.01j)


@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_host_scales_wavenumber_and_contrast(cross_section):
    """A sphere in a host with index n_h is the same as a sphere in vacuum with k0 -> n_h k0 and eps_p -> eps_p / eps_h."""
    k0 = np.array([0.5, 2.0, 7.0])
    eps_p, eps_h = 6.0 + 0.4j, 1.77

    in_host = cross_section(k0, 1.0, eps_p=eps_p, eps_h=eps_h)[0]
    in_vacuum = cross_section(np.sqrt(eps_h) * k0, 1.0, eps_p=eps_p / eps_h)[0]

    np.testing.assert_allclose(in_host, in_vacuum, rtol=1e-12)


@pytest.mark.parametrize("cross_section", CROSS_SECTIONS)
def test_unknown_norm_raises(cross_section):
    with pytest.raises(ValueError, match="norm"):
        cross_section(np.array([1.0]), 1.0, eps_p=4.0, norm='geometric')


# c_n and d_n from miepython 3.3.0, for n = 1, 2, 3
MIEPYTHON_INTERNAL = [
    # m, x, c_n, d_n
    (1.5 + 0.2j, 2.0,
     [0.548200802114356+0.7618395184353032j, 0.7690918512381497+0.1526681212571882j, 0.42271841106667124-0.07656773540372534j],
     [0.5766010332264746+0.5775777935654849j, 0.7214107437686161+0.19743971046806222j, 0.4207702928492232-0.08898761743330984j]),
    (4.0 + 0.01j, 7.0,
     [-0.11746952746859492+2.535422446194578j, -0.3535357363564826+0.8983044757783928j, 0.7269927205516239+3.0746337719581476j],
     [-0.5994814345995578+0.8189664964491808j, 0.1405179554356766+2.7757000866914225j, 0.05666359016672558+1.0413850085582068j]),
]


@pytest.mark.parametrize("m, x, c_n, d_n", MIEPYTHON_INTERNAL)
def test_internal_coefficients_match_miepython(m, x, c_n, d_n):
    n = np.arange(1, 4)

    np.testing.assert_allclose(mielib.optics_mie_c(n, x, eps_p=m**2), c_n, rtol=1e-12)
    np.testing.assert_allclose(mielib.optics_mie_d(n, x, eps_p=m**2), d_n, rtol=1e-12)


@pytest.mark.parametrize("eps_p, mu_p, eps_h, mu_h, k0a", [
    (2.25, 1, 1, 1, 1.3),
    (6 + 0.4j, 1, 1.77, 1, 2.1),
    (4 + 0.1j, 2.0, 1.5, 1.2, 0.8),
    (2.25, 1, 1.5 + 0.1j, 1, 1.0),
    (16, 1, 1, 1, 1.0 - 0.1j),
])
def test_coefficients_satisfy_boundary_conditions(eps_p, mu_p, eps_h, mu_h, k0a):
    """Continuity of the tangential E and H at r = a, Bohren & Huffman, with mu = mu_p / mu_h."""
    n = np.arange(1, 7)
    n_h = np.sqrt(eps_h * mu_h + 0j)
    x = n_h * k0a
    m = np.sqrt(eps_p * mu_p + 0j) / n_h
    mu = mu_p / mu_h

    jx, jmx = sp.spherical_jn(n, x), sp.spherical_jn(n, m * x)
    hx = mielib.spherical_h1(n, x)
    xjx_p = jx + x * sp.spherical_jn(n, x, 1)
    mxjmx_p = jmx + m * x * sp.spherical_jn(n, m * x, 1)
    xhx_p = hx + x * mielib.spherical_h1(n, x, p=1)

    params = dict(eps_p=eps_p, mu_p=mu_p, eps_h=eps_h, mu_h=mu_h)
    a, b = mielib.optics_mie_a(n, k0a, **params), mielib.optics_mie_b(n, k0a, **params)
    c, d = mielib.optics_mie_c(n, k0a, **params), mielib.optics_mie_d(n, k0a, **params)

    np.testing.assert_allclose(jmx * c + hx * b, jx, rtol=1e-12)
    np.testing.assert_allclose(mxjmx_p * c + mu * xhx_p * b, mu * xjx_p, rtol=1e-12)
    np.testing.assert_allclose(m * jmx * d + mu * hx * a, mu * jx, rtol=1e-12)
    np.testing.assert_allclose(mxjmx_p * d + m * xhx_p * a, m * xjx_p, rtol=1e-12)


def test_internal_coefficients_share_the_denominators():
    n = np.arange(1, 5)
    params = dict(k0a=1.7, eps_p=5 + 0.2j, mu_p=1.3, eps_h=1.4, mu_h=1.1, get_denominator=True)

    np.testing.assert_allclose(mielib.optics_mie_c(n, **params), mielib.optics_mie_b(n, **params))
    np.testing.assert_allclose(mielib.optics_mie_d(n, **params), mielib.optics_mie_a(n, **params))

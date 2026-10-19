import numpy as np
import pytest

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

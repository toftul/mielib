import numpy as np
import pytest

from mielib import optics, special

# Q_sca, Q_ext, Q_back and g from miepython 3.3.0.
# miepython uses m = n - i kappa. mielib uses exp(-i omega t), so m = n + i kappa.
MIEPYTHON = [
    # m, x, Q_sca, Q_ext, Q_back, g
    (1.55, 5.213, 3.1049959150801865, 3.1049959150801865, 2.9242091271819697, 0.6331044159946938),
    (1.5, 60.0, 2.137281954969279, 2.137281954969279, 1.8078667408375773, 0.8014385660200115),
    (1.33, 150.0, 2.1143767571353997, 2.1143767571353997, 0.8239276249556166, 0.8803967605008126),
    (1.5 + 0.01j, 10.0, 2.344131626959545, 2.7706950637975383, 1.3621432845403942, 0.7937231950924977),
    (0.2 + 3.0j, 5.0, 2.9829374120236696, 3.1913392682413173, 0.16464247271093094, 0.5515688860293155),
    (4.0 + 0.01j, 50.0, 1.505664155858015, 2.137765425666939, 1.156682872182073, 0.7135153027342718),
]


@pytest.mark.parametrize("m, x, q_sca, q_ext, q_back, g", MIEPYTHON)
def test_efficiencies_match_miepython(m, x, q_sca, q_ext, q_back, g):
    q = optics.cross_sections(x, 1.0, m**2, norm='geom')

    np.testing.assert_allclose(q.sca, q_sca, rtol=1e-9)
    np.testing.assert_allclose(q.ext, q_ext, rtol=1e-9)
    np.testing.assert_allclose(q.abs, q_ext - q_sca, atol=1e-9)
    np.testing.assert_allclose(q.back, q_back, rtol=1e-9)
    np.testing.assert_allclose(q.g, g, rtol=1e-9)
    np.testing.assert_allclose(q.pr, q.ext - q.g * q.sca, rtol=1e-12)


@pytest.mark.parametrize("eps_rel", [2.25, 4 + 0.5j])
def test_scattering_matches_rayleigh_limit(eps_rel):
    x = 0.01
    q_rayleigh = 8 / 3 * x**4 * abs((eps_rel - 1) / (eps_rel + 2))**2

    np.testing.assert_allclose(optics.cross_sections(x, 1.0, eps_rel, norm='geom').sca, q_rayleigh, rtol=1e-4)


@pytest.mark.parametrize("jmax", [None, 300])
def test_one_call_covers_small_and_large_spheres(jmax):
    k = np.array([0.01, 0.1, 1.0, 60.0])

    one_call = optics.cross_sections(k, 1.0, 2.25, jmax=jmax)
    separate = [optics.cross_sections(ki, 1.0, 2.25) for ki in k]

    for name in ['sca', 'ext', 'abs', 'back', 'pr', 'g']:
        values = getattr(one_call, name)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, [getattr(s, name) for s in separate], rtol=1e-12, atol=1e-30)


def test_input_broadcasts_to_a_grid():
    """A sweep of k and of the radius in one call, with a dispersive eps_rel(k)."""
    k = np.array([0.5, 2.0, 7.0])[:, None]
    a = np.array([0.3, 1.0])[None, :]
    eps_rel = np.array([6.0 + 0.4j, 5.0 + 0.3j, 4.0 + 0.2j])[:, None]

    grid = optics.cross_sections(k, a, eps_rel, 1.2)

    assert grid.sca.shape == (3, 2)
    assert grid.sca_e.shape == (grid.sca_e.shape[0], 3, 2)
    for i in range(3):
        for l in range(2):
            point = optics.cross_sections(k[i, 0], a[0, l], eps_rel[i, 0], 1.2)
            np.testing.assert_allclose(grid.ext[i, l], point.ext, rtol=1e-12)
            np.testing.assert_allclose(grid.g[i, l], point.g, rtol=1e-12)


def test_multipoles_add_up_and_jmax_is_inclusive():
    q = optics.cross_sections(2.0, 1.0, 6.0 + 0.4j, jmax=3)

    assert q.sca_e.shape[0] == 4
    np.testing.assert_allclose(q.sca, np.sum(q.sca_e + q.sca_m), rtol=1e-14)
    np.testing.assert_allclose(q.ext, np.sum(q.ext_e + q.ext_m), rtol=1e-14)
    np.testing.assert_allclose(q.abs, np.sum(q.abs_e + q.abs_m), rtol=1e-12)
    np.testing.assert_array_equal(q.sca_e[0], 0)

    j = np.arange(1, 4)
    a = optics.mie_a(j, 2.0, 6.0 + 0.4j)
    np.testing.assert_allclose(q.sca_e[1:], 2 * np.pi / 2.0**2 * (2 * j + 1) * np.abs(a)**2, rtol=1e-14)


def test_lossy_host_raises():
    with pytest.raises(ValueError, match="lossless host"):
        optics.cross_sections(1.0 + 0.01j, 1.0, 4.0)


def test_unknown_norm_raises():
    with pytest.raises(ValueError, match="norm"):
        optics.cross_sections(1.0, 1.0, 4.0, norm='geometric')


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        optics.cross_sections(1.0, 1.0, 4.0, 1.0, 1)


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
    j = np.arange(1, 4)

    np.testing.assert_allclose(optics.mie_c(j, x, m**2), c_n, rtol=1e-12)
    np.testing.assert_allclose(optics.mie_d(j, x, m**2), d_n, rtol=1e-12)


BOUNDARY_CASES = [
    # x, eps_rel, mu_rel
    (1.3, 2.25, 1),
    (np.sqrt(1.77) * 2.1, (6 + 0.4j) / 1.77, 1),
    (np.sqrt(1.8) * 0.8, (4 + 0.1j) / 1.5, 2.0 / 1.2),
    # lossy host
    (np.sqrt(1.5 + 0.1j) * 1.0, 2.25 / (1.5 + 0.1j), 1),
    # complex frequency
    (1.0 - 0.1j, 16, 1),
    # double-negative sphere
    (np.sqrt(1.3) * 1.9, (-2 + 0.1j) / 1.3, -1 + 0.1j),
]


@pytest.mark.parametrize("x, eps_rel, mu_rel", BOUNDARY_CASES)
def test_amplitudes_satisfy_boundary_conditions(x, eps_rel, mu_rel):
    """The tangential E = e / sqrt(eps) and H = h / sqrt(mu) are continuous at r = a. With psi, xi at x and psi at m x:
        A channel:  psi' + T^e xi' = A^in psi'(mx) / (m sqrt(eps_rel)),  psi + T^e xi = A^in psi(mx) / (m sqrt(mu_rel)),
        B channel:  psi + T^m xi = B^in psi(mx) / (m sqrt(eps_rel)),    psi' + T^m xi' = B^in psi'(mx) / (m sqrt(mu_rel))."""
    j = np.arange(1, 7)
    sqrt_eps, sqrt_mu = np.sqrt(eps_rel + 0j), np.sqrt(mu_rel + 0j)
    m = sqrt_eps * sqrt_mu

    psi, dpsi = special.riccati_psi(j, x), special.riccati_psi(j, x, derivative=1)
    xi, dxi = special.riccati_xi(j, x), special.riccati_xi(j, x, derivative=1)
    psi_in, dpsi_in = special.riccati_psi(j, m * x), special.riccati_psi(j, m * x, derivative=1)

    te, tm = optics.tmatrix(j, x, eps_rel, mu_rel)
    a_in, b_in = optics.internal_amplitudes(j, x, eps_rel, mu_rel)

    np.testing.assert_allclose(dpsi + te * dxi, a_in * dpsi_in / (m * sqrt_eps), rtol=1e-12)
    np.testing.assert_allclose(psi + te * xi, a_in * psi_in / (m * sqrt_mu), rtol=1e-12)
    np.testing.assert_allclose(psi + tm * xi, b_in * psi_in / (m * sqrt_eps), rtol=1e-12)
    np.testing.assert_allclose(dpsi + tm * dxi, b_in * dpsi_in / (m * sqrt_mu), rtol=1e-12)


@pytest.mark.parametrize("x, eps_rel, mu_rel", BOUNDARY_CASES)
def test_bohren_coefficients_relate_to_the_amplitudes(x, eps_rel, mu_rel):
    """T^e = -a, T^m = -b, A^in = sqrt(eps_rel) d, B^in = sqrt(eps_rel) c, and A^in D^a = B^in D^b = i m^2."""
    j = np.arange(1, 7)
    args = (j, x, eps_rel, mu_rel)
    sqrt_eps = np.sqrt(eps_rel + 0j)
    m2 = sqrt_eps**2 * np.sqrt(mu_rel + 0j)**2

    te, tm = optics.tmatrix(*args)
    a_in, b_in = optics.internal_amplitudes(*args)
    da, db = optics.denominators(*args)

    np.testing.assert_allclose(te, -optics.mie_a(*args), rtol=1e-14)
    np.testing.assert_allclose(tm, -optics.mie_b(*args), rtol=1e-14)
    np.testing.assert_allclose(a_in, sqrt_eps * optics.mie_d(*args), rtol=1e-14)
    np.testing.assert_allclose(b_in, sqrt_eps * optics.mie_c(*args), rtol=1e-14)
    np.testing.assert_allclose(a_in * da, 1j * m2, rtol=1e-14)
    np.testing.assert_allclose(b_in * db, 1j * m2, rtol=1e-14)


@pytest.mark.parametrize("x, eps_rel, mu_rel", BOUNDARY_CASES)
def test_duality_swaps_electric_and_magnetic(x, eps_rel, mu_rel):
    """eps_rel <-> mu_rel swaps a and b, the denominators and the internal amplitudes, with no other factor."""
    j = np.arange(1, 7)
    direct, dual = (j, x, eps_rel, mu_rel), (j, x, mu_rel, eps_rel)

    np.testing.assert_allclose(optics.mie_a(*dual), optics.mie_b(*direct), rtol=1e-13)
    np.testing.assert_allclose(optics.mie_b(*dual), optics.mie_a(*direct), rtol=1e-13)
    np.testing.assert_allclose(optics.denominators(*dual), optics.denominators(*direct)[::-1], rtol=1e-13)
    np.testing.assert_allclose(optics.internal_amplitudes(*dual), optics.internal_amplitudes(*direct)[::-1], rtol=1e-13)


def test_dual_sphere_does_not_mix_electric_and_magnetic():
    """eps_rel = mu_rel gives a_j = b_j for each j: the sphere keeps the helicity, and back = 0 (Kerker)."""
    j = np.arange(1, 8)
    np.testing.assert_allclose(optics.mie_a(j, 2.3, 3.0 + 0.2j, 3.0 + 0.2j), optics.mie_b(j, 2.3, 3.0 + 0.2j, 3.0 + 0.2j), rtol=1e-14)
    np.testing.assert_allclose(optics.cross_sections(2.3, 1.0, 3.0 + 0.2j, 3.0 + 0.2j).back, 0, atol=1e-28)


@pytest.mark.parametrize("eps_rel, mu_rel, lossless", [
    (6.0, 1, True),
    (4.0 / 1.5, 2.0 / 1.2, True),
    (6 + 0.4j, 1, False),
    (4.0 / 1.5, (2 + 0.3j) / 1.2, False),
])
def test_tmatrix_conserves_the_flux(eps_rel, mu_rel, lossless):
    """j_j = (h_j^(1) + h_j^(2)) / 2, so the outgoing wave has the amplitude 1 + 2T: |1 + 2T| = 1 without loss and < 1 with loss."""
    j = np.arange(1, 5)

    for t in optics.tmatrix(j, 1.3, eps_rel, mu_rel):
        if lossless:
            np.testing.assert_allclose(np.abs(1 + 2 * t), 1, rtol=1e-12)
        else:
            assert np.all(np.abs(1 + 2 * t) < 1)


def test_cross_sections_from_the_tmatrix():
    """sigma_ext = -(2 pi / k^2) sum (2j+1) Re(T^e + T^m), with the same sign as in acoustics."""
    k, eps_rel, mu_rel = 1.7, (6 + 0.4j) / 1.5, 1.2 / 1.1
    # the same terms as in cross_sections
    j = np.arange(1, special.wiscombe_jmax(k) + 1)
    te, tm = optics.tmatrix(j, k, eps_rel, mu_rel)

    sigma_ext = -2 * np.pi / k**2 * np.sum((2 * j + 1) * np.real(te + tm))
    sigma_sc = 2 * np.pi / k**2 * np.sum((2 * j + 1) * (np.abs(te)**2 + np.abs(tm)**2))

    q = optics.cross_sections(k, 1.0, eps_rel, mu_rel)
    np.testing.assert_allclose(sigma_ext, q.ext, rtol=1e-12)
    np.testing.assert_allclose(sigma_sc, q.sca, rtol=1e-12)


def test_totals_are_scalars_for_a_scalar_input():
    q = optics.cross_sections(1.0, 1.0, 4.0)
    for name in ['sca', 'ext', 'abs', 'back', 'pr', 'g']:
        assert isinstance(getattr(q, name), np.float64), name
    assert q.sca_e.shape == (special.wiscombe_jmax(1.0) + 1,)

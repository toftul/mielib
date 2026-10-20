import numpy as np
import pytest
import scipy.special as sp

import mielib


def y1_and_dtheta(m, theta, phi):
    """Closed-form Y_1^m(theta, phi) and its theta derivative, with the Condon-Shortley phase."""
    if m == 0:
        c = np.sqrt(3 / (4 * np.pi))
        return c * np.cos(theta), -c * np.sin(theta)
    c = -m * np.sqrt(3 / (8 * np.pi)) * np.exp(1j * m * phi)
    return c * np.sin(theta), c * np.cos(theta)


def sph_vector_to_cart(A, theta, phi):
    """Cartesian components of the vector A = (A_r, A_theta, A_phi)."""
    Ar, At, Ap = A
    return np.array([
        Ar * np.sin(theta) * np.cos(phi) + At * np.cos(theta) * np.cos(phi) - Ap * np.sin(phi),
        Ar * np.sin(theta) * np.sin(phi) + At * np.cos(theta) * np.sin(phi) + Ap * np.cos(phi),
        Ar * np.cos(theta) - At * np.sin(theta),
    ])


@pytest.mark.parametrize("theta", [
    np.array([0.4, 0.9, 2.2, 2.9]),
    # on the z axis and near it
    np.array([0.0, 1e-10, np.pi - 1e-10, np.pi]),
])
def test_bohren_vsh_expand_the_plane_wave(theta):
    """Bohren & Huffman, Eqs. (4.37) and (4.38): the x-polarized plane wave E = x exp(ikz) and H = (k / omega mu) y exp(ikz)."""
    rho = np.array([0.3, 1.7, 4.2, 6.0])
    phi = np.array([0.0, 0.4, 3.0, 5.5])

    E, H = 0, 0
    for n in range(1, 40):
        En = 1j**n * (2 * n + 1) / (n * (n + 1))
        E = E + En * (mielib.vsh_bohren_mo(1, n, rho, theta, phi, 1) - 1j * mielib.vsh_bohren_ne(1, n, rho, theta, phi, 1))
        H = H + En * (mielib.vsh_bohren_me(1, n, rho, theta, phi, 1) + 1j * mielib.vsh_bohren_no(1, n, rho, theta, phi, 1))

    wave = np.exp(1j * rho * np.cos(theta))
    zero = 0 * wave
    np.testing.assert_allclose(sph_vector_to_cart(E, theta, phi), [wave, zero, zero], atol=1e-12)
    # H = -(k / omega mu) sum_n E_n (M_e1n + i N_o1n)
    np.testing.assert_allclose(sph_vector_to_cart(H, theta, phi), [zero, -wave, zero], atol=1e-12)


def test_bohren_angular_functions_match_the_recurrence():
    """pi_n = P_n^1 / sin(theta) and tau_n = d P_n^1 / d theta from the upward recurrence of Bohren & Huffman, also on the z axis."""
    theta = np.array([0.0, 1e-10, 0.7, 2.4, np.pi - 1e-10, np.pi])
    mu = np.cos(theta)

    pi_prev, pi_n = np.zeros_like(theta), np.ones_like(theta)
    for n in range(1, 12):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        _, pinm, dPnm = mielib.harmonics._bohren_legendre(1, n, theta)

        np.testing.assert_allclose(pinm, pi_n, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dPnm, tau_n, rtol=1e-12, atol=1e-12)

        pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_toftul_vsh_match_closed_form_for_j1(m):
    rho = np.array([0.4, 1.3, 2.5])
    theta = np.array([0.3, 1.1, 2.6])
    phi = np.array([0.2, 2.0, 4.5])

    Y, dY = y1_and_dtheta(m, theta, phi)
    z = sp.spherical_jn(1, rho)
    zp = sp.spherical_jn(1, rho, 1)

    M = [0 * Y, 1j * m / np.sin(theta) * z * Y, -z * dY]
    N = [2 * z / rho * Y, (z + rho * zp) / rho * dY, 1j * m / np.sin(theta) * (z + rho * zp) / rho * Y]
    L = [zp * Y, z / rho * dY, 1j * m / np.sin(theta) * z / rho * Y]

    np.testing.assert_allclose(mielib.vsh_toftul_m(m, 1, rho, theta, phi, superscript=1), M, atol=1e-14)
    np.testing.assert_allclose(mielib.vsh_toftul_n(m, 1, rho, theta, phi, superscript=1), N, atol=1e-14)
    np.testing.assert_allclose(mielib.vsh_toftul_l(m, 1, rho, theta, phi, superscript=1), L, atol=1e-14)


@pytest.mark.parametrize("j", [1, 2, 3])
def test_jackson_x_is_proportional_to_toftul_m(j):
    """M_jm = -i sqrt(j (j+1)) z_j X_jm, also on the z axis."""
    rho = 1.3
    theta = np.array([0.0, 1e-10, 0.7, 2.4, np.pi])
    phi = np.array([0.3, 1.0, 2.0, 4.0, 5.0])

    for m in range(-j, j + 1):
        X = mielib.vsh_jackson_x(m, j, theta, phi)
        M = mielib.vsh_toftul_m(m, j, rho, theta, phi, superscript=1)

        assert np.all(np.isfinite(X))
        np.testing.assert_allclose(M, -1j * np.sqrt(j * (j + 1)) * sp.spherical_jn(j, rho) * X, atol=1e-14)

    np.testing.assert_array_equal(theta[0], 0.0)


def test_jackson_x_is_zero_for_n0():
    np.testing.assert_array_equal(mielib.vsh_jackson_x(0, 0, np.array([0.0, 0.7]), 0.3), np.zeros((3, 2)))


@pytest.mark.parametrize("parity, direct_m, direct_n", [
    ("even", mielib.vsh_bohren_me, mielib.vsh_bohren_ne),
    ("odd", mielib.vsh_bohren_mo, mielib.vsh_bohren_no),
])
def test_wrappers_return_bohren_vsh(parity, direct_m, direct_n):
    args = dict(m=1, j=2, rho=1.3, theta=0.7, phi=0.4, superscript=1, source='bohren', parity=parity)
    direct_args = dict(m=1, n=2, rho=1.3, theta=0.7, phi=0.4, superscript=1)

    np.testing.assert_allclose(mielib.vector_spherical_harmonic_m(**args), direct_m(**direct_args))
    np.testing.assert_allclose(mielib.vector_spherical_harmonic_n(**args), direct_n(**direct_args))
    assert np.any(mielib.vector_spherical_harmonic_n(**args) != 0)


@pytest.mark.parametrize("vsh", [
    mielib.vsh_bohren_me, mielib.vsh_bohren_mo, mielib.vsh_bohren_ne, mielib.vsh_bohren_no,
    mielib.vsh_toftul_m, mielib.vsh_toftul_n, mielib.vsh_toftul_l,
])
def test_vsh_do_not_change_the_input_arrays(vsh):
    rho = np.array([0.0, 1.0], dtype=complex)
    theta = np.array([0.0, 0.5])

    vsh(1, 2, rho, theta, 0.3, superscript=1)

    np.testing.assert_array_equal(rho, [0.0, 1.0])
    np.testing.assert_array_equal(theta, [0.0, 0.5])


@pytest.mark.parametrize("vsh, options, message", [
    (mielib.vector_spherical_harmonic_m, dict(source='jackson'), "source"),
    (mielib.vector_spherical_harmonic_n, dict(source='jackson'), "source"),
    (mielib.vector_spherical_harmonic_l, dict(source='bohren'), "source"),
    (mielib.vector_spherical_harmonic_m, dict(source='bohren', parity='even '), "parity"),
    (mielib.vector_spherical_harmonic_n, dict(source='bohren', parity='odd '), "parity"),
])
def test_unknown_vsh_option_raises(vsh, options, message):
    with pytest.raises(ValueError, match=message):
        vsh(1, 2, 1.0, 0.5, 0.3, **options)

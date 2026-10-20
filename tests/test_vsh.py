from math import factorial

import numpy as np
import pytest
import scipy.special as sp

from mielib import vsh


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
def test_real_vsh_expand_the_plane_wave(theta):
    """Bohren & Huffman, Eqs. (4.37) and (4.38): the x-polarized plane wave E = x exp(ikz) and H = (k / omega mu) y exp(ikz)."""
    rho = np.array([0.3, 1.7, 4.2, 6.0])
    phi = np.array([0.0, 0.4, 3.0, 5.5])
    args = (rho, theta, phi)

    E, H = 0, 0
    for j in range(1, 40):
        Ej = 1j**j * (2 * j + 1) / (j * (j + 1))
        E = E + Ej * (vsh.M_real(j, 1, *args, superscript=1, parity='odd') - 1j * vsh.N_real(j, 1, *args, superscript=1, parity='even'))
        H = H + Ej * (vsh.M_real(j, 1, *args, superscript=1, parity='even') + 1j * vsh.N_real(j, 1, *args, superscript=1, parity='odd'))

    wave = np.exp(1j * rho * np.cos(theta))
    zero = 0 * wave
    np.testing.assert_allclose(sph_vector_to_cart(E, theta, phi), [wave, zero, zero], atol=1e-12)
    # H = -(k / omega mu) sum_j E_j (M_e1j + i N_o1j)
    np.testing.assert_allclose(sph_vector_to_cart(H, theta, phi), [zero, -wave, zero], atol=1e-12)


def angular_psi_phi(j, m, theta, phi):
    """Psi_jm = r grad Y_jm and Phi_jm = r_hat x Psi_jm, in Cartesian components."""
    Y = sp.sph_harm_y(j, m, theta, phi)
    dY = m / np.tan(theta) * Y
    if m + 1 <= j:
        dY = dY + np.exp(-1j * phi) * np.sqrt((j - m) * (j + m + 1)) * sp.sph_harm_y(j, m + 1, theta, phi)
    psi = [0, dY, 1j * m / np.sin(theta) * Y]
    phi_vec = [0, -1j * m / np.sin(theta) * Y, dY]
    return sph_vector_to_cart(psi, theta, phi), sph_vector_to_cart(phi_vec, theta, phi)


def test_complex_vsh_expand_a_plane_wave_of_any_direction_and_polarization():
    """e exp(i k.r) = sum_jm 4 pi i^j / (j (j+1)) [-(e . Phi*_jm(k_hat)) M_jm - i (e . Psi*_jm(k_hat)) N_jm]."""
    theta_k, phi_k = 1.1, 2.3
    k_hat = sph_vector_to_cart([1, 0, 0], theta_k, phi_k)
    e = (0.6 + 0.2j) * sph_vector_to_cart([0, 1, 0], theta_k, phi_k) + (0.3 - 0.7j) * sph_vector_to_cart([0, 0, 1], theta_k, phi_k)

    rho = np.array([0.5, 2.3, 4.0])
    theta = np.array([0.7, 2.0, 0.0])
    phi = np.array([-1.2, 0.4, 0.0])

    E = 0
    for j in range(1, 40):
        for m in range(-j, j + 1):
            psi, phi_vec = angular_psi_phi(j, m, theta_k, phi_k)
            c = 4 * np.pi * 1j**j / (j * (j + 1))
            E = E + c * (-(e @ np.conj(phi_vec)) * vsh.M(j, m, rho, theta, phi, superscript=1)
                         - 1j * (e @ np.conj(psi)) * vsh.N(j, m, rho, theta, phi, superscript=1))

    r = rho * sph_vector_to_cart([1, 0, 0], theta, phi)
    np.testing.assert_allclose(sph_vector_to_cart(E, theta, phi), e[:, None] * np.exp(1j * k_hat @ r), atol=1e-11)


def test_real_angular_functions_match_the_recurrence():
    """pi_n = P_n^1 / sin(theta) and tau_n = d P_n^1 / d theta from the upward recurrence of Bohren & Huffman, also on the z axis."""
    theta = np.array([0.0, 1e-10, 0.7, 2.4, np.pi - 1e-10, np.pi])
    mu = np.cos(theta)

    pi_prev, pi_n = np.zeros_like(theta), np.ones_like(theta)
    for n in range(1, 12):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        _, pi, dP = vsh._bohren_legendre(n, 1, theta)

        np.testing.assert_allclose(pi, pi_n, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dP, tau_n, rtol=1e-12, atol=1e-12)

        pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n


@pytest.mark.parametrize("j, m", [(1, 0), (2, 1), (3, 2), (3, 3)])
def test_complex_vsh_from_the_real_ones(j, m):
    """W_jm = (-1)^m beta_jm (W_jm^even + i W_jm^odd), W = M, N, with beta_jm = sqrt((2j+1)/(4 pi) (j-m)!/(j+m)!)."""
    args = (1.3, np.array([0.0, 0.8, 2.9]), np.array([0.0, 0.5, 4.0]))
    beta = np.sqrt((2 * j + 1) / (4 * np.pi) * factorial(j - m) / factorial(j + m))

    for W, W_real in [(vsh.M, vsh.M_real), (vsh.N, vsh.N_real)]:
        real = W_real(j, m, *args, superscript=3, parity='even') + 1j * W_real(j, m, *args, superscript=3, parity='odd')
        np.testing.assert_allclose(W(j, m, *args, superscript=3), (-1)**m * beta * real, atol=1e-12)


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_complex_vsh_match_closed_form_for_j1(m):
    rho = np.array([0.4, 1.3, 2.5])
    theta = np.array([0.3, 1.1, 2.6])
    phi = np.array([0.2, 2.0, 4.5])

    Y, dY = y1_and_dtheta(m, theta, phi)
    z = sp.spherical_jn(1, rho)
    zp = sp.spherical_jn(1, rho, 1)

    M = [0 * Y, 1j * m / np.sin(theta) * z * Y, -z * dY]
    N = [2 * z / rho * Y, (z + rho * zp) / rho * dY, 1j * m / np.sin(theta) * (z + rho * zp) / rho * Y]
    L = [zp * Y, z / rho * dY, 1j * m / np.sin(theta) * z / rho * Y]

    np.testing.assert_allclose(vsh.M(1, m, rho, theta, phi, superscript=1), M, atol=1e-14)
    np.testing.assert_allclose(vsh.N(1, m, rho, theta, phi, superscript=1), N, atol=1e-14)
    np.testing.assert_allclose(vsh.L(1, m, rho, theta, phi, superscript=1), L, atol=1e-14)


def test_complex_vsh_match_mathematica():
    """Values from examples/harmonics.ipynb: Grad, Curl of r h_j^(1) Y_jm in Mathematica."""
    np.testing.assert_allclose(vsh.N(3, 1, 1.2 + 1j, 0.3, 0.5, superscript=3),
                               [2.6883 - 6.20557j, -1.02717 + 4.019j, -5.57267 - 1.42426j], rtol=2e-5)
    np.testing.assert_allclose(vsh.M(4, -1, 1.2 + 1j, 1.3, 0.5, superscript=3),
                               [0, -2.89538 + 2.24409j, -4.06637 - 5.24654j], rtol=2e-5, atol=1e-12)


@pytest.mark.parametrize("j", [1, 2, 3])
def test_x_is_proportional_to_m(j):
    """M_jm = -i sqrt(j (j+1)) z_j X_jm, also on the z axis."""
    rho = 1.3
    theta = np.array([0.0, 1e-10, 0.7, 2.4, np.pi])
    phi = np.array([0.3, 1.0, 2.0, 4.0, 5.0])

    for m in range(-j, j + 1):
        X = vsh.X(j, m, theta, phi)
        M = vsh.M(j, m, rho, theta, phi, superscript=1)

        assert np.all(np.isfinite(X))
        np.testing.assert_allclose(M, -1j * np.sqrt(j * (j + 1)) * sp.spherical_jn(j, rho) * X, atol=1e-14)

    np.testing.assert_array_equal(theta[0], 0.0)


def test_x_is_zero_for_j0():
    np.testing.assert_array_equal(vsh.X(0, 0, np.array([0.0, 0.7]), 0.3), np.zeros((3, 2)))


@pytest.mark.parametrize("harmonic, options", [
    (vsh.M, {}), (vsh.N, {}), (vsh.L, {}),
    (vsh.M_real, dict(parity='even')), (vsh.N_real, dict(parity='odd')),
])
def test_vsh_do_not_change_the_input_arrays(harmonic, options):
    rho = np.array([0.0, 1.0], dtype=complex)
    theta = np.array([0.0, 0.5])

    harmonic(2, 1, rho, theta, 0.3, superscript=1, **options)

    np.testing.assert_array_equal(rho, [0.0, 1.0])
    np.testing.assert_array_equal(theta, [0.0, 0.5])


@pytest.mark.parametrize("harmonic", [vsh.M_real, vsh.N_real])
def test_unknown_parity_raises(harmonic):
    with pytest.raises(ValueError, match="parity"):
        harmonic(2, 1, 1.0, 0.5, 0.3, superscript=1, parity='even ')


@pytest.mark.parametrize("harmonic", [vsh.M, vsh.N, vsh.L])
def test_superscript_is_a_required_keyword(harmonic):
    with pytest.raises(TypeError):
        harmonic(2, 1, 1.0, 0.5, 0.3)
    with pytest.raises(TypeError):
        harmonic(2, 1, 1.0, 0.5, 0.3, 3)

"""Differential identities and orthogonality of the vector spherical harmonics, with k = 1 (rho = r)."""
from math import factorial

import numpy as np
import pytest
import scipy.special as sp

import mielib

# points off the z axis, one point in each column
POINTS = np.array([
    [0.7, -1.2, 0.3, 1.5],
    [-0.4, 0.5, 1.6, 0.2],
    [1.1, 0.9, -0.8, 0.4],
])


def cart_field(vsh, m, j, X, superscript):
    """Cartesian components of a harmonic at the Cartesian points X."""
    r, theta, phi = mielib.cart2sph(*X)
    A = vsh(m, j, r, theta, phi, superscript=superscript)
    return np.einsum('ij...,j...->i...', mielib.rotation_matrix_3d_sph(theta, phi), A)


def jacobian(f, X, h=1e-3):
    """J[i, l] = d f_i / d x_l by fourth-order central differences."""
    J = []
    for l in range(3):
        dX = np.zeros((3, 1))
        dX[l] = h
        J.append((-f(X + 2 * dX) + 8 * f(X + dX) - 8 * f(X - dX) + f(X - 2 * dX)) / (12 * h))
    return np.stack(J, axis=1)


def assert_zero_or_equal(actual, desired, scale):
    """Finite differences: the error is relative to the size of the fields."""
    np.testing.assert_allclose(actual, desired, atol=1e-9 * scale)


def div(J):
    return J[0, 0] + J[1, 1] + J[2, 2]


def curl(J):
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


@pytest.mark.parametrize("superscript", [1, 3])
@pytest.mark.parametrize("m, j", [(0, 1), (1, 1), (-1, 2), (2, 3), (-3, 3)])
def test_toftul_vsh_differential_identities(m, j, superscript):
    """div M = div N = 0, curl M = k N, curl N = k M, curl L = 0 and div L = -k z_j Y_jm."""
    def field(vsh):
        return lambda X: cart_field(vsh, m, j, X, superscript)

    M = field(mielib.vsh_toftul_m)
    N = field(mielib.vsh_toftul_n)
    L = field(mielib.vsh_toftul_l)
    J_M, J_N, J_L = jacobian(M, POINTS), jacobian(N, POINTS), jacobian(L, POINTS)

    r, theta, phi = mielib.cart2sph(*POINTS)
    psi = mielib.spherical_radial_function_z(j, r, superscript=superscript) * sp.sph_harm_y(j, m, theta, phi)

    scale = max(np.max(np.abs(F(POINTS))) for F in (M, N, L))
    assert_zero_or_equal(div(J_M), 0, scale)
    assert_zero_or_equal(div(J_N), 0, scale)
    assert_zero_or_equal(curl(J_M), N(POINTS), scale)
    assert_zero_or_equal(curl(J_N), M(POINTS), scale)
    assert_zero_or_equal(curl(J_L), 0, scale)
    assert_zero_or_equal(div(J_L), -psi, scale)


@pytest.mark.parametrize("superscript", [1, 3])
@pytest.mark.parametrize("m, n", [(0, 1), (1, 1), (1, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("vsh_m, vsh_n", [
    (mielib.vsh_bohren_me, mielib.vsh_bohren_ne),
    (mielib.vsh_bohren_mo, mielib.vsh_bohren_no),
])
def test_bohren_vsh_differential_identities(vsh_m, vsh_n, m, n, superscript):
    """div M = div N = 0, curl M = k N and curl N = k M, for each parity."""
    def M(X):
        return cart_field(vsh_m, m, n, X, superscript)

    def N(X):
        return cart_field(vsh_n, m, n, X, superscript)

    J_M, J_N = jacobian(M, POINTS), jacobian(N, POINTS)

    scale = max(np.max(np.abs(F(POINTS))) for F in (M, N))
    assert_zero_or_equal(div(J_M), 0, scale)
    assert_zero_or_equal(div(J_N), 0, scale)
    assert_zero_or_equal(curl(J_M), N(POINTS), scale)
    assert_zero_or_equal(curl(J_N), M(POINTS), scale)


def sphere_quadrature(jmax):
    """Nodes and weights on the unit sphere, exact for the products of harmonics with j <= jmax."""
    x, w_theta = np.polynomial.legendre.leggauss(jmax + 2)
    n_phi = 4 * jmax + 2
    theta, phi = np.meshgrid(np.arccos(x), 2 * np.pi * np.arange(n_phi) / n_phi, indexing='ij')
    weights = np.outer(w_theta, np.full(n_phi, 2 * np.pi / n_phi))
    return theta, phi, weights


def gram(A, B, weights):
    """G[a, b] = integral over the sphere of A_a . conj(B_b)."""
    return np.einsum('aitp,bitp,tp->ab', A, np.conj(B), weights)


def assert_gram(G, expected, scale):
    """The quadrature is exact, so the error is round-off relative to the largest norm."""
    np.testing.assert_allclose(G, expected, rtol=1e-12, atol=1e-13 * scale)


@pytest.mark.parametrize("superscript", [1, 3])
def test_toftul_vsh_are_orthogonal(superscript):
    """On a sphere, M, N and L of different (m, j) are orthogonal, M is orthogonal to N and L, and the norms follow from |Y|^2 and |r grad Y|^2."""
    jmax, rho = 4, 1.7
    theta, phi, weights = sphere_quadrature(jmax)
    labels = [(m, j) for j in range(1, jmax + 1) for m in range(-j, j + 1)]

    def harmonics(vsh):
        return np.array([vsh(m, j, rho, theta, phi, superscript=superscript) for m, j in labels])

    M, N, L = harmonics(mielib.vsh_toftul_m), harmonics(mielib.vsh_toftul_n), harmonics(mielib.vsh_toftul_l)

    j = np.array([j for _, j in labels])
    z = mielib.spherical_radial_function_z(j, rho, superscript=superscript)
    zp = mielib.spherical_radial_function_z(j, rho, p=1, superscript=superscript)
    d = (z + rho * zp) / rho  # (rho z)' / rho
    jj = j * (j + 1)

    norm_N = jj * (jj * np.abs(z / rho)**2 + np.abs(d)**2)
    scale = np.max(norm_N)
    assert_gram(gram(M, M, weights), np.diag(jj * np.abs(z)**2), scale)
    assert_gram(gram(N, N, weights), np.diag(norm_N), scale)
    assert_gram(gram(L, L, weights), np.diag(np.abs(zp)**2 + jj * np.abs(z / rho)**2), scale)
    assert_gram(gram(N, L, weights), np.diag(jj * (z / rho * np.conj(zp) + d * np.conj(z / rho))), scale)
    assert_gram(gram(M, N, weights), 0, scale)
    assert_gram(gram(M, L, weights), 0, scale)


def test_bohren_vsh_are_orthogonal():
    """Bohren & Huffman: the real M_emn, M_omn and N_emn, N_omn are orthogonal on a sphere, and the norm of M is
    (1 + delta_m0) pi n (n+1) 2/(2n+1) (n+m)!/(n-m)! |z_n|^2."""
    nmax, rho = 4, 1.7
    theta, phi, weights = sphere_quadrature(nmax)
    # M_o0n = N_o0n = 0
    labels = [(vsh_m, vsh_n, m, n)
              for n in range(1, nmax + 1) for m in range(0, n + 1)
              for vsh_m, vsh_n in [(mielib.vsh_bohren_me, mielib.vsh_bohren_ne), (mielib.vsh_bohren_mo, mielib.vsh_bohren_no)]
              if m > 0 or vsh_m is mielib.vsh_bohren_me]

    M = np.array([vsh_m(m, n, rho, theta, phi, superscript=1) for vsh_m, _, m, n in labels])
    N = np.array([vsh_n(m, n, rho, theta, phi, superscript=1) for _, vsh_n, m, n in labels])

    norm_M = np.array([(1 + (m == 0)) * np.pi * n * (n + 1) * 2 / (2 * n + 1) * factorial(n + m) / factorial(n - m)
                       * sp.spherical_jn(n, rho)**2 for _, _, m, n in labels])
    G_N = gram(N, N, weights)

    scale = max(np.max(norm_M), np.max(np.abs(G_N)))
    assert_gram(gram(M, M, weights), np.diag(norm_M), scale)
    assert_gram(G_N - np.diag(np.diag(G_N)), 0, scale)
    assert_gram(gram(M, N, weights), 0, scale)

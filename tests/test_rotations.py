import numpy as np
import pytest

import mielib


def test_sph2cart_inverts_cart2sph():
    x = np.array([1.0, -0.5, 0.2, 0.0])
    y = np.array([0.3, 2.0, -1.5, 0.7])
    z = np.array([-0.4, 0.1, 0.9, -2.0])

    np.testing.assert_allclose(mielib.sph2cart(*mielib.cart2sph(x, y, z)), (x, y, z), atol=1e-14)


def test_cart2sph_at_the_origin_and_near_the_z_axis():
    x = np.array([0.0, 1e-10, 1e-10])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 1.0, -1.0])

    r, theta, phi = mielib.cart2sph(x, y, z)

    np.testing.assert_allclose(r, [0.0, 1.0, 1.0], rtol=1e-15)
    np.testing.assert_allclose(theta, [0.0, 1e-10, np.pi - 1e-10], rtol=1e-14)
    np.testing.assert_allclose(phi, [0.0, 0.0, 0.0], atol=1e-15)


def test_cyl2cart_inverts_cart2cyl():
    x = np.array([1.0, -0.5, 0.2, 0.0])
    y = np.array([0.3, 2.0, -1.5, 0.7])
    z = np.array([-0.4, 0.1, 0.9, -2.0])

    np.testing.assert_allclose(mielib.cyl2cart(*mielib.cart2cyl(x, y, z)), (x, y, z), atol=1e-14)


@pytest.mark.parametrize("rotation, angles", [
    (mielib.rotation_matrix_3d_sph, (np.array([0.0, 0.4, 1.9, np.pi]), np.array([0.3, 2.0, 4.0, 6.0]))),
    (mielib.rotation_matrix_3d_sph, (np.array([0.0, 0.4, 1.9, np.pi]), 0.5)),
    (mielib.rotation_matrix_3d_cyl, (np.array([0.3, 2.0, 4.0, 6.0]),)),
    (mielib.rotation_matrix_2d, (np.array([0.3, 2.0, 4.0, 6.0]),)),
])
def test_rotation_matrices_accept_arrays(rotation, angles):
    R = rotation(*angles)
    d = R.shape[0]

    assert R.shape == (d, d, 4)
    for i in range(4):
        np.testing.assert_allclose(R[..., i], rotation(*[np.broadcast_to(a, 4)[i] for a in angles]), atol=1e-15)
    # each matrix is orthogonal
    np.testing.assert_allclose(np.einsum('ki...,kj...->ij...', R, R), np.broadcast_to(np.eye(d)[..., None], (d, d, 4)), atol=1e-15)


def test_unknown_cylinder_axis_raises():
    with pytest.raises(NotImplementedError):
        mielib.rotation_matrix_3d_cyl(0.3, axis='x')
    with pytest.raises(ValueError, match="axis"):
        mielib.rotation_matrix_3d_cyl(0.3, axis='w')


THETA = np.array([0.3, 1.2, 2.0, 2.9])
PHI = np.array([0.1, 1.7, 3.5, 5.9])


def central_difference(f, t, h=1e-6):
    return (np.array(f(t + h)) - np.array(f(t - h))) / (2 * h)


def test_spherical_matrix_columns_are_the_unit_vectors():
    """r_hat = dr/dr, theta_hat = (1/r) dr/dtheta, phi_hat = (1/(r sin theta)) dr/dphi, and r_hat x theta_hat = phi_hat."""
    R = mielib.rotation_matrix_3d_sph(THETA, PHI)

    theta_hat = central_difference(lambda t: mielib.sph2cart(1.0, t, PHI), THETA)
    phi_hat = central_difference(lambda p: mielib.sph2cart(1.0, THETA, p), PHI) / np.sin(THETA)

    np.testing.assert_allclose(R[:, 0], mielib.sph2cart(1.0, THETA, PHI), atol=1e-15)
    np.testing.assert_allclose(R[:, 1], theta_hat, atol=1e-9)
    np.testing.assert_allclose(R[:, 2], phi_hat, atol=1e-9)
    np.testing.assert_allclose(np.cross(R[:, 0], R[:, 1], axis=0), R[:, 2], atol=1e-15)


def test_cylindrical_matrix_columns_are_the_unit_vectors():
    """rho_hat = dr/drho, phi_hat = (1/rho) dr/dphi, z_hat, and rho_hat x phi_hat = z_hat."""
    R = mielib.rotation_matrix_3d_cyl(PHI)
    R2 = mielib.rotation_matrix_2d(PHI)

    phi_hat = central_difference(lambda p: mielib.cyl2cart(1.0, p, np.zeros(4)), PHI)

    np.testing.assert_allclose(R[:, 0], mielib.cyl2cart(1.0, PHI, np.zeros(4)), atol=1e-15)
    np.testing.assert_allclose(R[:, 1], phi_hat, atol=1e-9)
    np.testing.assert_allclose(R[:, 2], np.broadcast_to([[0], [0], [1]], (3, 4)), atol=1e-15)
    np.testing.assert_allclose(np.cross(R[:, 0], R[:, 1], axis=0), R[:, 2], atol=1e-15)
    # the 2D matrix is the upper-left block
    np.testing.assert_allclose(R2, R[:2, :2], atol=1e-15)


def test_position_vector_in_local_components():
    """r = r r_hat in spherical components, and r = rho rho_hat + z z_hat in cylindrical components."""
    X = np.array([[0.7, -1.2, 0.3, 1.5], [-0.4, 0.5, 1.6, 0.2], [1.1, 0.9, -0.8, 0.4]])
    r, theta, phi = mielib.cart2sph(*X)
    rho, phi_cyl, z = mielib.cart2cyl(*X)

    # A^loc = R^T A^cart, because R is orthogonal
    sph = np.einsum('ji...,j...->i...', mielib.rotation_matrix_3d_sph(theta, phi), X)
    cyl = np.einsum('ji...,j...->i...', mielib.rotation_matrix_3d_cyl(phi_cyl), X)

    np.testing.assert_allclose(sph, [r, 0 * r, 0 * r], atol=1e-15)
    np.testing.assert_allclose(cyl, [rho, 0 * rho, z], atol=1e-15)

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

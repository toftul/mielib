import numpy as np
import pytest

import mielib


def test_sph2cart_inverts_cart2sph():
    x = np.array([1.0, -0.5, 0.2, 0.0])
    y = np.array([0.3, 2.0, -1.5, 0.7])
    z = np.array([-0.4, 0.1, 0.9, -2.0])

    np.testing.assert_allclose(mielib.sph2cart(*mielib.cart2sph(x, y, z)), (x, y, z), atol=1e-14)


def test_cyl2cart_inverts_cart2cyl():
    x = np.array([1.0, -0.5, 0.2, 0.0])
    y = np.array([0.3, 2.0, -1.5, 0.7])
    z = np.array([-0.4, 0.1, 0.9, -2.0])

    np.testing.assert_allclose(mielib.cyl2cart(*mielib.cart2cyl(x, y, z)), (x, y, z), atol=1e-14)


def test_unknown_cylinder_axis_raises():
    with pytest.raises(NotImplementedError):
        mielib.rotation_matrix_3d_cyl(0.3, axis='x')
    with pytest.raises(ValueError, match="axis"):
        mielib.rotation_matrix_3d_cyl(0.3, axis='w')

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

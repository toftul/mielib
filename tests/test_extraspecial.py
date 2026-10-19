import numpy as np
import pytest
import scipy.special as sp

import mielib


@pytest.mark.parametrize("n", range(6))
@pytest.mark.parametrize("f, fpp", [
    (sp.spherical_jn, mielib.spherical_jnpp),
    (sp.spherical_yn, mielib.spherical_ynpp),
    (mielib.spherical_h1, mielib.spherical_h1pp),
])
def test_second_derivative_solves_bessel_equation(f, fpp, n):
    """z^2 f'' + 2 z f' + (z^2 - n(n+1)) f = 0 for each spherical Bessel function f."""
    z = np.linspace(0.5, 10, 20)

    lhs = z**2 * fpp(n, z) + 2 * z * f(n, z, 1)
    rhs = -(z**2 - n * (n + 1)) * f(n, z)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

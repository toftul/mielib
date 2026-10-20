import numpy as np
import pytest
import scipy.special as sp

from mielib import special

Z = np.linspace(0.5, 10, 20)
FUNCTIONS = [special.spherical_jn, special.spherical_yn, special.spherical_h1, special.spherical_h2]


@pytest.mark.parametrize("j", range(6))
@pytest.mark.parametrize("f", FUNCTIONS)
def test_second_derivative_solves_bessel_equation(f, j):
    """z^2 f'' + 2 z f' + (z^2 - j(j+1)) f = 0 for each spherical Bessel function f."""
    lhs = Z**2 * f(j, Z, derivative=2) + 2 * Z * f(j, Z, derivative=1)
    rhs = -(Z**2 - j * (j + 1)) * f(j, Z)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_bessel_functions_match_scipy():
    j = np.arange(0, 6)[:, None]
    z = Z + 0.3j

    np.testing.assert_allclose(special.spherical_jn(j, z), sp.spherical_jn(j, z))
    np.testing.assert_allclose(special.spherical_yn(j, z, derivative=1), sp.spherical_yn(j, z, True))
    np.testing.assert_allclose(special.spherical_h1(j, z), sp.spherical_jn(j, z) + 1j * sp.spherical_yn(j, z))
    np.testing.assert_allclose(special.spherical_h2(j, z), sp.spherical_jn(j, z) - 1j * sp.spherical_yn(j, z))


@pytest.mark.parametrize("superscript, f", list(zip([1, 2, 3, 4], FUNCTIONS)))
@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_radial_function_selects_by_superscript(superscript, f, derivative):
    np.testing.assert_array_equal(special.spherical_z(3, Z, superscript=superscript, derivative=derivative),
                                  f(3, Z, derivative=derivative))


@pytest.mark.parametrize("j", range(6))
def test_riccati_bessel_functions(j):
    """psi_j'' + (1 - j(j+1)/z^2) psi_j = 0, the same for xi_j, and the Wronskian psi_j xi_j' - psi_j' xi_j = i."""
    z = Z + 0.2j
    for f in [special.riccati_psi, special.riccati_xi]:
        np.testing.assert_allclose(f(j, z, derivative=2), -(1 - j * (j + 1) / z**2) * f(j, z), rtol=1e-12, atol=1e-12)

    wronskian = (special.riccati_psi(j, z) * special.riccati_xi(j, z, derivative=1)
                 - special.riccati_psi(j, z, derivative=1) * special.riccati_xi(j, z))
    np.testing.assert_allclose(wronskian, 1j, rtol=1e-12)


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="superscript"):
        special.spherical_z(1, 1.0, superscript=5)
    with pytest.raises(ValueError, match="derivative"):
        special.spherical_jn(1, 1.0, derivative=3)
    with pytest.raises(TypeError):
        # superscript has no default
        special.spherical_z(1, 1.0)


def test_wiscombe_jmax():
    """x + 4.05 x^(1/3) + 2, rounded down."""
    np.testing.assert_array_equal(special.wiscombe_jmax(np.array([0.1, 1.0, 10.0, 100.0])), [3, 7, 20, 120])

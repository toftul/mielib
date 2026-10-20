"""
    Spherical (r, θ, φ) and cylindrical (ρ, φ, z) coordinates, and the components of vectors in them.
    https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
"""
import numpy as np

__all__ = [
    "cart2sph",
    "sph2cart",
    "cart2cyl",
    "cyl2cart",
    "rotation_matrix_sph",
    "rotation_matrix_cyl",
    "rotation_matrix_2d",
    "vector_sph2cart",
    "vector_cart2sph",
    "vector_cyl2cart",
    "vector_cart2cyl",
]


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # arccos(z/r) is nan at r = 0 and loses the precision near the z axis
    theta = np.arctan2(np.hypot(x, y), z)
    phi = np.arctan2(y, x)

    return r, theta, phi


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def cart2cyl(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi, z


def cyl2cart(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y, z


def rotation_matrix_sph(theta, phi):
    """
        A^cart = R(θ, φ) A^sph. The columns of R are the unit vectors r, θ and φ.

        For arrays theta and phi, R has the shape (3, 3) + the broadcast shape of theta and phi.
    """
    # all the entries must have the same shape
    theta, phi = np.broadcast_arrays(theta, phi)
    zero = np.zeros(theta.shape)

    return np.array([
        [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
        [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
        [            np.cos(theta),            -np.sin(theta),         zero]
    ])


def rotation_matrix_cyl(phi):
    """
        A^cart = R(φ) A^cyl. The columns of R are the unit vectors ρ, φ and z.

        For an array phi, R has the shape (3, 3) + the shape of phi.
    """
    # all the entries must have the same shape
    zero, one = np.zeros(np.shape(phi)), np.ones(np.shape(phi))

    return np.array([
        [np.cos(phi), -np.sin(phi), zero],
        [np.sin(phi),  np.cos(phi), zero],
        [       zero,         zero,  one],
    ])


def rotation_matrix_2d(phi):
    """
        A^cart = R(φ) A^polar in the plane. The columns of R are the unit vectors ρ and φ.

        For an array phi, R has the shape (2, 2) + the shape of phi.
    """
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])


def vector_sph2cart(A, theta, phi):
    """
        Cartesian components (A_x, A_y, A_z) of the vector with the spherical components A = (A_r, A_θ, A_φ).
        A has the shape (3,) + the shape of the points.
    """
    return np.einsum('ij...,j...->i...', rotation_matrix_sph(theta, phi), A)


def vector_cart2sph(A, theta, phi):
    """
        Spherical components (A_r, A_θ, A_φ) of the vector with the Cartesian components A = (A_x, A_y, A_z).
        A has the shape (3,) + the shape of the points.
    """
    # R is orthogonal, so the inverse is the transpose
    return np.einsum('ji...,j...->i...', rotation_matrix_sph(theta, phi), A)


def vector_cyl2cart(A, phi):
    """
        Cartesian components (A_x, A_y, A_z) of the vector with the cylindrical components A = (A_ρ, A_φ, A_z).
        A has the shape (3,) + the shape of the points.
    """
    return np.einsum('ij...,j...->i...', rotation_matrix_cyl(phi), A)


def vector_cart2cyl(A, phi):
    """
        Cylindrical components (A_ρ, A_φ, A_z) of the vector with the Cartesian components A = (A_x, A_y, A_z).
        A has the shape (3,) + the shape of the points.
    """
    # R is orthogonal, so the inverse is the transpose
    return np.einsum('ji...,j...->i...', rotation_matrix_cyl(phi), A)

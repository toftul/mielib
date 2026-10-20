import numpy as np

def rotation_matrix_3d_sph(theta, phi):
    """
        A^cart = R(theta, phi) A^sph

        For arrays theta and phi, R has the shape (3, 3) + the broadcast shape of theta and phi.

        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    """
    # all the entries must have the same shape
    theta, phi = np.broadcast_arrays(theta, phi)
    zero = np.zeros(theta.shape)

    return np.array([
        [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
        [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
        [            np.cos(theta),            -np.sin(theta),         zero]
    ])


def rotation_matrix_2d(phi):
    """
        A^cart = R(phi) A^cyl

        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    """
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])


def rotation_matrix_3d_cyl(phi, axis='z'):
    """
        A^cart = R(phi) A^cyl

        For an array phi, R has the shape (3, 3) + the shape of phi.

        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    """
    match axis:
        case 'x' | 'y':
            raise NotImplementedError("Only axis='z' is available.")
        case 'z':
            # all the entries must have the same shape
            zero, one = np.zeros(np.shape(phi)), np.ones(np.shape(phi))
            return np.array([
                [np.cos(phi), -np.sin(phi), zero],
                [np.sin(phi),  np.cos(phi), zero],
                [       zero,         zero,  one],
            ])
        case _:
            raise ValueError(f"axis must be 'x', 'y' or 'z', not {axis!r}")


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
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
import numpy as np

def rotation_matrix_3d_sph(theta, phi):
    """
        A^cart = R(theta, phi) A^sph

        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    """
    return np.array([
        [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
        [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
        [            np.cos(theta),            -np.sin(theta),            0]
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

        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    """
    match axis:
        case 'x':
            return np.nan
        case 'y':
            return np.nan
        case 'z':
            return np.array([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi),  np.cos(phi), 0],
                [          0,            0, 1],
            ])
        case _:
            return np.eye(3)


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y/x)

    return r, theta, phi


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    x = r * np.sin(theta) * np.sin(phi)
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
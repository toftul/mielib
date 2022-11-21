import numpy as np

def rotation_matrix_3d(theta, phi):
    """
        A^cart = R(theta, phi) A^sph
    """
    return np.array([
        [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
        [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
        [            np.cos(theta),            -np.sin(theta),            0]
    ])
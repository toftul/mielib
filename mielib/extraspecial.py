import scipy.special as sp


def spherical_h1(n, z, p=0):
    return sp.spherical_jn(n, z, p) + 1j * sp.spherical_yn(n, z, p)


def spherical_jnpp(n, z):
    # slower option is to
    #     spherical_jn(n-1, z, 1) + (n+1)/z**2 * spherical_jn(n, z) - (n+1)/z * spherical_jn(n, z, 1)
    # but this is faster
    return n/(2*n+1) * sp.spherical_jn(n-1, z, 1) - (n+1)/(2*n+1) * sp.spherical_jn(n+1, z, 1)


def spherical_ynpp(n, z):
    return n/(2*n+1) * sp.spherical_yn(n-1, z, 1) - (n+1)/(2*n+1) * sp.spherical_yn(n+1, z, 1)


def spherical_h1pp(n, z):
    return spherical_jnpp(n, z) + 1j * spherical_ynpp(n, z)


def spherical_h2(n, z, p):
    return sp.spherical_jn(n, z, p) - 1j * sp.spherical_yn(n, z, p)
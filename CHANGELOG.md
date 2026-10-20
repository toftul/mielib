# Changelog

This file lists the main changes in each release of mielib.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-10-21

### Fixed

- The Bohren & Huffman harmonics `vsh_bohren_me`, `vsh_bohren_mo`, `vsh_bohren_ne` and `vsh_bohren_no`
  now have the sign of the book for odd `m`.
  Bohren & Huffman use P_n^m without the Condon-Shortley phase (-1)^m, but `scipy.special.lpmv` has it.
  Before, the harmonics were (-1)^m times the book harmonics,
  and the plane-wave expansion of Eqs. (4.37) and (4.38) gave -E and -H.
- The Bohren & Huffman harmonics are now correct on the z axis and near it.
  Before, they were zero for m = 1 when θ < 1.05e-8 or θ = π, because cos θ is then 1 or -1 in floating point.
  Thus, the plane-wave expansion gave zero on the z axis.
  The harmonics now use `scipy.special.sph_legendre_p`, which takes θ and not cos θ,
  and the limit of P_n^m / sin θ on the axis.
- `vsh_jackson_x` no longer returns `nan` at θ = 0 and for n = 0.
  For n = 0, it now returns zeros, because L Y_00 = 0.
- `rotation_matrix_3d_sph` and `rotation_matrix_3d_cyl` now accept arrays of angles.
  The matrix then has the shape (3, 3) + the shape of the angles.
  Before, they raised `ValueError`, because some entries were the scalars 0 and 1.
- `cart2sph` now gives θ = 0 at the origin. Before, it gave `nan`.
  θ is now also accurate near the z axis. Before, it was 0 for x = 1e-10, y = 0, z = 1.

## [0.2.0] - 2026-10-19

### Added

- A test suite in `tests/`. To run it, use `poetry run pytest`.
  The tests stop with an error when the code calls a deprecated function.
- A GitHub Actions workflow runs the tests with Python 3.12 to 3.14 for each push to `main` and each pull request.
- `wiscombe_nstop(x)` gives the highest multipole order that a converged Mie series needs.
- `optics_mie_c` and `optics_mie_d` give the internal optical Mie coefficients c_n and d_n of Bohren & Huffman.
  They have the same arguments as `optics_mie_a` and `optics_mie_b`, including `get_denominator`.

### Changed

- mielib now needs Python 3.12 or later and SciPy 1.18 or later.
- The default of `nmax` in all cross-section functions is now `None`.
  With `None`, each function finds `nmax` from the largest size parameter.
  The shape of the per-multipole arrays now depends on the size parameter.
- The optical cross-section functions now raise `ValueError` for a lossy host.
  The cross sections are not defined in a lossy host.
  Before, the functions removed the imaginary part and gave incorrect values with many warnings.
  `optics_mie_a` and `optics_mie_b` still accept a lossy host.
- A published GitHub release now uploads the package to PyPI with Trusted Publishing.
  The release tag must be equal to the version in `pyproject.toml`, with or without a `v` prefix.
  The old workflow, which used an API token, is removed.
- The docstrings of `acoustics_mie_a` and `acoustics_mie_c` now give the source of the formulas
  ([arXiv:1905.12216](https://arxiv.org/abs/1905.12216)), the field expansion and the time convention.
- An unknown option now raises `ValueError`.
  Before, the functions returned zeros, `nan` or the identity matrix without a warning.
  This applies to `norm` in the cross sections, `source` and `parity` in the harmonics,
  `superscript` in `spherical_radial_function_z`, and `axis` in `rotation_matrix_3d_cyl`.
- `rotation_matrix_3d_cyl` now raises `NotImplementedError` for `axis='x'` and `axis='y'`.
  Before, it returned `nan`.
- The docstring of `optics_mie_b` now says "magnetic" Mie coefficient.
- The docstrings of the Bohren & Huffman harmonics now call θ the polar angle and φ the azimuthal angle.

### Fixed

- The vector spherical harmonics `vsh_toftul_m`, `vsh_toftul_n`, `vsh_toftul_l` and `vsh_jackson_x`
  work again with SciPy 1.17 and later.
  SciPy 1.17 removed `scipy.special.sph_harm`. mielib now uses `scipy.special.sph_harm_y`.
- `sph2cart` no longer raises `NameError`. The function now calculates `y`.
- The cross sections are now correct for large spheres.
  The old default `nmax=50` stopped the series too early when the size parameter was more than approximately 40.
  For example, the error in `Q_sca` was 35 % for `m = 1.5` and `x = 60`.
- The cross sections no longer return `nan` for small spheres when `nmax` is large.
  Each point now uses only the multipoles up to `wiscombe_nstop(x)`.
  Thus, one call can now calculate a sweep from small to large spheres.
- `vector_spherical_harmonic_n(source='bohren')` now returns the Bohren & Huffman harmonic.
  Before, it always returned zeros.
- The vector spherical harmonics no longer change the `rho` and `theta` arrays of the caller.
  Before, they replaced each zero in these arrays with `1e-15`.
- `spherical_jnpp`, `spherical_ynpp` and `spherical_h1pp` no longer return `nan` for `n = 0`.
- The acoustic cross-section functions now accept a scalar `k`.
  Before, they raised `AttributeError`.

## [0.1.12] - 2023-05-24

This is the last release before this changelog.
For older changes, see the git history.

[Unreleased]: https://github.com/toftul/mielib/compare/0.2.1...HEAD
[0.2.1]: https://github.com/toftul/mielib/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/toftul/mielib/compare/0.1.12...0.2.0
[0.1.12]: https://github.com/toftul/mielib/releases/tag/0.1.12

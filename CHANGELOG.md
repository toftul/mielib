# Changelog

This file lists the main changes in each release of mielib.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A test suite in `tests/`. To run it, use `poetry run pytest`.
  The tests stop with an error when the code calls a deprecated function.
- `wiscombe_nstop(x)` gives the highest multipole order that a converged Mie series needs.

### Changed

- mielib now needs SciPy 1.15 or later.
- The default of `nmax` in all cross-section functions is now `None`.
  With `None`, each function finds `nmax` from the largest size parameter.
  The shape of the per-multipole arrays now depends on the size parameter.
- The optical cross-section functions now raise `ValueError` for a lossy host.
  The cross sections are not defined in a lossy host.
  Before, the functions removed the imaginary part and gave incorrect values with many warnings.
  `optics_mie_a` and `optics_mie_b` still accept a lossy host.
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

[Unreleased]: https://github.com/toftul/mielib/compare/0.1.12...HEAD
[0.1.12]: https://github.com/toftul/mielib/releases/tag/0.1.12

# Changelog

This file lists the main changes in each release of mielib.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A test suite in `tests/`. To run it, use `poetry run pytest`.
  The tests stop with an error when the code calls a deprecated function.

### Changed

- mielib now needs SciPy 1.15 or later.

### Fixed

- The vector spherical harmonics `vsh_toftul_m`, `vsh_toftul_n`, `vsh_toftul_l` and `vsh_jackson_x`
  work again with SciPy 1.17 and later.
  SciPy 1.17 removed `scipy.special.sph_harm`. mielib now uses `scipy.special.sph_harm_y`.

## [0.1.12] - 2023-05-24

This is the last release before this changelog.
For older changes, see the git history.

[Unreleased]: https://github.com/toftul/mielib/compare/0.1.12...HEAD
[0.1.12]: https://github.com/toftul/mielib/releases/tag/0.1.12

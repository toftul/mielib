# Changelog

This file lists the main changes in each release of mielib.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `optics.denominators` is now two functions, `optics.denominator_a` and `optics.denominator_b`,
  in the same way as `optics.mie_a` and `optics.mie_b`.
  A search for the modes of one type needs only one of the two.
  For `optics.denominators(j, x, eps_rel, mu_rel)`, use `optics.denominator_a(j, x, eps_rel, mu_rel)`
  and `optics.denominator_b(j, x, eps_rel, mu_rel)`. The values do not change.

## [0.3.0] - 2026-10-21

This release changes the API and does not keep the old names.
The table in "Migration from 0.2" gives the new call for each old call.

### Changed

- The API is now five submodules: `mielib.special`, `mielib.vsh`, `mielib.coordinates`, `mielib.optics` and `mielib.acoustics`.
  `mielib` itself exports only these submodules. Use, for example, `from mielib import optics` and then `optics.mie_a(...)`.
- The multipole order is `j`, the total angular momentum, and its projection is `m`.
  The harmonics take the indices in the order `(j, m)`, as |j m> and `scipy.special.sph_harm_y`.
  Before, the order was `(m, j)` or `(m, n)`.
- The Mie coefficients take the size parameter `x = k a` in the host and the parameters of the particle relative to the host:
  `eps_rel = eps_p / eps_h` and `mu_rel = mu_p / mu_h` in optics, `rho_rel` and `beta_rel` in acoustics.
  The cross sections take the wavenumber `k` in the host, the radius `a` and the same relative parameters, in both physics.
- The options `superscript`, `parity`, `derivative`, `jmin`, `jmax` and `norm` are keyword-only.
  `superscript` and `parity` have no default. Before, the default of `superscript` was 3 in some functions and 1 in others.
  A call that relied on a default now raises `TypeError`, so its result cannot change without a warning.
- `jmax` is inclusive: the sums include `jmin <= j <= jmax`. Before, `nmax` was exclusive.
- Each physics has one function `cross_sections`, which returns a frozen dataclass `CrossSections`.
  It computes all the orders in one call. `k`, `a` and the material parameters broadcast together, so a 2D sweep works.
  Row `j` of each array per multipole is the order `j`, and its shape is `(jmax + 1,)` + the shape of the input.
  For a scalar input, the totals are NumPy scalars. Before, they had the shape `(1,)`.
- A complex `k` (a lossy host) now raises `ValueError` also in acoustics.
- The optical Mie coefficients come from the dual-symmetric form
  a_j = [s1 ψ_j(mx) ψ_j'(x) - s2 ψ_j(x) ψ_j'(mx)] / [s1 ψ_j(mx) ξ_j'(x) - s2 ξ_j(x) ψ_j'(mx)],
  with (s1, s2) = (√eps_rel, √mu_rel) for a_j and (√mu_rel, √eps_rel) for b_j.
  Thus, the swap `eps_rel <-> mu_rel` swaps a_j and b_j. The values agree with 0.2 to 1e-12.
- m = √eps_rel √mu_rel and not √(eps_rel mu_rel).
  For a double-negative sphere (eps_rel < 0 and mu_rel < 0), m now has Re m < 0 and Im m > 0,
  as a passive medium needs with exp(-i omega t). a_j and b_j do not change.
  c_j and d_j change by (-1)^j and (-1)^(j+1), so the field inside does not change.
- The denominators have the normalization of the dual-symmetric form. See `optics.denominators` and `acoustics.denominator`.
  Their zeros, the modes of the sphere, do not change.

### Added

- `optics.tmatrix` and `acoustics.tmatrix` give the diagonal T-matrix of a sphere:
  the outgoing amplitude divided by the regular amplitude of the same multipole.
  They use the same sign convention. Thus, T^e_j = -a_j and T^m_j = -b_j in optics, but T_j = a_j in acoustics,
  because Bohren & Huffman and Phys. Rev. Lett. 123, 183901 (2019) define the scattering coefficients with opposite signs.
- `optics.internal_amplitudes` gives A^in / A^inc = i m² / D^a_j and B^in / B^inc = i m² / D^b_j
  for the field (√eps E, √mu H). These are √eps_rel d_j and √eps_rel c_j. The swap `eps_rel <-> mu_rel` swaps them.
- `optics.denominators` and `acoustics.denominator`.
- The optical cross sections have the backscattering `back`, the radiation pressure `pr` and the asymmetry parameter `g`.
  The acoustic cross sections have `back` and `g`.
- `special.spherical_jn` and `special.spherical_yn` with `derivative=0, 1, 2`,
  and the Riccati-Bessel functions `special.riccati_psi` and `special.riccati_xi`.
- `coordinates.vector_sph2cart`, `vector_cart2sph`, `vector_cyl2cart` and `vector_cart2cyl`
  convert the components of a vector field.
- `mielib.__version__` gives the installed version.
- `docs/theory/mielib_theory.pdf`, a note with the conventions and the formulas.
- New tests: the differential identities and the orthogonality of the harmonics,
  the expansion of a plane wave of any direction and polarization,
  the boundary conditions in the dual-symmetric form, the duality, the T-matrix,
  Q_back and g against miepython 3.3.0, and the rotation matrices.

### Removed

- The flat names in `mielib`, for example `mielib.optics_mie_a`. There are no aliases.
- The modules `extraspecial`, `harmonics`, `rotations`, `mieoptics` and `mieacoustics`.
- The wrappers `vector_spherical_harmonic_m`, `vector_spherical_harmonic_n` and `vector_spherical_harmonic_l`.
- `spherical_jnpp`, `spherical_ynpp` and `spherical_h1pp`. Use `derivative=2`.
- The argument `get_denominator`. Use `optics.denominators` and `acoustics.denominator`.
- The argument `axis` of the cylindrical rotation matrix. Only `axis='z'` worked.
- The split of the acoustic radiation-pressure cross section into multipoles.
  The cross terms a_j a*_(j+1) belong to two multipoles, so the split is not unique.

### Migration from 0.2

| 0.2 | 0.3 |
|---|---|
| `spherical_h1(n, z, p)`, `spherical_h2(n, z, p)` | `special.spherical_h1(j, z, derivative=p)`, `special.spherical_h2(...)` |
| `spherical_jnpp(n, z)`, `spherical_ynpp`, `spherical_h1pp` | `special.spherical_jn(j, z, derivative=2)`, `spherical_yn`, `spherical_h1` |
| `spherical_radial_function_z(j, rho, p, superscript)` | `special.spherical_z(j, rho, superscript=s, derivative=p)` |
| `wiscombe_nstop(x)` | `special.wiscombe_jmax(x)` |
| `vsh_toftul_m(m, j, rho, theta, phi, superscript)`, `_n`, `_l` | `vsh.M(j, m, rho, theta, phi, superscript=s)`, `vsh.N`, `vsh.L` |
| `vsh_jackson_x(m, n, theta, phi)` | `vsh.X(j, m, theta, phi)` |
| `vsh_bohren_me(m, n, rho, theta, phi, s)`, `vsh_bohren_mo` | `vsh.M_real(n, m, rho, theta, phi, superscript=s, parity='even')`, `parity='odd'` |
| `vsh_bohren_ne(m, n, ...)`, `vsh_bohren_no` | `vsh.N_real(n, m, ..., parity='even')`, `parity='odd'` |
| `vector_spherical_harmonic_m(m, j, ..., source, parity)`, `_n`, `_l` | `vsh.M`, `vsh.N`, `vsh.L`, or `vsh.M_real`, `vsh.N_real` |
| `cart2sph`, `sph2cart`, `cart2cyl`, `cyl2cart` | `coordinates.cart2sph`, `sph2cart`, `cart2cyl`, `cyl2cart` |
| `rotation_matrix_3d_sph(theta, phi)` | `coordinates.rotation_matrix_sph(theta, phi)` |
| `rotation_matrix_3d_cyl(phi, axis='z')`, `rotation_matrix_2d(phi)` | `coordinates.rotation_matrix_cyl(phi)`, `coordinates.rotation_matrix_2d(phi)` |
| `optics_mie_a(n, k0a, eps_p, mu_p, eps_h, mu_h)`, `_b`, `_c`, `_d` | `optics.mie_a(j, x, eps_rel, mu_rel)`, `mie_b`, `mie_c`, `mie_d`, with `x = sqrt(eps_h mu_h) k0a`, `eps_rel = eps_p / eps_h`, `mu_rel = mu_p / mu_h` |
| `optics_mie_a(..., get_denominator=True)`, `optics_mie_b(...)` | `optics.denominators(j, x, eps_rel, mu_rel)` → (D^a, D^b). The old values were (√mu_rel / x) D^a and (√mu_rel / (m x)) D^b. |
| `optics_tmatrix(n, k0a, eps_p, ...)` | `optics.tmatrix(j, x, eps_rel, mu_rel)` |
| `optics_scattering_cross_section(k0, a, eps_p, mu_p, eps_h, mu_h, nmin, nmax, norm)` | `q = optics.cross_sections(k, a, eps_rel, mu_rel, jmin=nmin, jmax=nmax - 1, norm=norm)`, then `q.sca`, `q.sca_e`, `q.sca_m`, with `k = sqrt(eps_h mu_h) k0` |
| `optics_extinction_cross_section(...)`, `optics_absorption_cross_section(...)` | `q.ext`, `q.ext_e`, `q.ext_m` and `q.abs`, `q.abs_e`, `q.abs_m` |
| `acoustics_mie_a(n, ka, rho1, beta1)`, `acoustics_mie_c` | `acoustics.mie_a(j, x, rho_rel, beta_rel)`, `acoustics.mie_c` |
| `acoustics_mie_a(..., get_denominator=True)` | `acoustics.denominator(j, x, rho_rel, beta_rel)`, which is √rho_rel times the old value |
| `acoustics_tmatrix(n, ka, rho_rel, beta_rel)` | `acoustics.tmatrix(j, x, rho_rel, beta_rel)` |
| `acoustics_scattering_cross_section(k, a, rho_rel, beta_rel, nmin, nmax, norm)` | `q = acoustics.cross_sections(k, a, rho_rel, beta_rel, jmin=nmin, jmax=nmax - 1, norm=norm)`, then `q.sca`, `q.sca_j` |
| `acoustics_pressure_cross_section(...)` | `q.pr` |
| `acoustics_extinction_cross_section(...)`, `acoustics_absorption_cross_section(...)` | `q.ext`, `q.ext_j` and `q.abs`, `q.abs_j` |

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

[Unreleased]: https://github.com/toftul/mielib/compare/0.3.0...HEAD
[0.3.0]: https://github.com/toftul/mielib/compare/0.2.1...0.3.0
[0.2.1]: https://github.com/toftul/mielib/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/toftul/mielib/compare/0.1.12...0.2.0
[0.1.12]: https://github.com/toftul/mielib/releases/tag/0.1.12

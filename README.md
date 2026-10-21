[![DOI](https://zenodo.org/badge/567526673.svg)](https://zenodo.org/badge/latestdoi/567526673)

# MieLib

MieLib is a Python library for the scattering of waves by a sphere (Mie theory) in optics and in acoustics,
and for the vector spherical harmonics.

- `mielib.optics`: an isotropic sphere, also magnetic, in a magnetic host.
  Mie coefficients, T-matrix, internal amplitudes and the denominators, whose zeros are the modes of the sphere.
- `mielib.acoustics`: a fluid sphere, from the Supplemental Material of
  [Phys. Rev. Lett. 123, 183901 (2019)](https://doi.org/10.1103/PhysRevLett.123.183901).
  The same functions as in optics.
- Cross sections in both physics: scattering, extinction, absorption, backscattering, radiation pressure,
  the asymmetry parameter, and the part of each multipole.
- `mielib.vsh`: the complex vector spherical harmonics $\mathbf{M}$, $\mathbf{N}$, $\mathbf{L}$, the harmonic $\mathbf{X}$ of Jackson,
  and the real harmonics of Bohren & Huffman.
- `mielib.special`: spherical Bessel, Hankel and Riccati–Bessel functions with their first and second derivatives.
- `mielib.coordinates`: spherical and cylindrical coordinates, and the components of vector fields in them.

## Installation

From [PyPI](https://pypi.org/project/mielib/):
```
pip install --upgrade mielib
```
mielib needs Python 3.12 or later and SciPy 1.18 or later.

## Quick start

```python
import numpy as np
from mielib import optics

# a sphere with the radius 100 nm and eps = 16 + 0.3i in vacuum
wavelength = np.linspace(500e-9, 1100e-9, 300)
k = 2 * np.pi / wavelength            # wavenumber in the host
q = optics.cross_sections(k, 100e-9, 16 + 0.3j, norm='geom')

q.sca, q.ext, q.abs                   # cross sections divided by pi a^2
q.sca_e[1], q.sca_m[1]                # the electric and the magnetic dipole parts of q.sca
```

All the Mie functions take the size parameter $x = ka$ in the host and the parameters of the particle relative to the host:
$\varepsilon_\mathrm{rel} = \varepsilon_p / \varepsilon_h$ and $\mu_\mathrm{rel} = \mu_p / \mu_h$ in optics,
the density $\rho_\mathrm{rel}$ and the compressibility $\beta_\mathrm{rel}$ in acoustics.
For example, `optics.mie_a(j, x, eps_rel, mu_rel)` and `acoustics.mie_a(j, x, rho_rel, beta_rel)`.
The harmonics take the indices in the order `(j, m)`, for example `vsh.N(j, m, rho, theta, phi, superscript=1)`.

## Theory

The note [*Vector spherical harmonics and Mie scattering in optics and acoustics: conventions and explicit formulas*](https://github.com/toftul/mielib/blob/main/docs/theory/mielib_theory.pdf)
gives all the definitions and the formulas that mielib uses.
Its notation follows the PhD thesis of I. Toftul,
[*Resonant Metaphotonics: From Optical Manipulation to Chiral Metasurfaces*](https://hdl.handle.net/1885/733812237) (Australian National University, 2026).

## Examples

The [examples](https://github.com/toftul/mielib/tree/main/examples) folder has Jupyter notebooks:

- [cross_sections.ipynb](https://github.com/toftul/mielib/blob/main/examples/cross_sections.ipynb):
  the cross sections of a sphere and their multipole parts, in optics and in acoustics.
- [fields.ipynb](https://github.com/toftul/mielib/blob/main/examples/fields.ipynb):
  the fields for a scattering problem.
- [harmonics.ipynb](https://github.com/toftul/mielib/blob/main/examples/harmonics.ipynb):
  the vector spherical harmonics, and a plane wave from the harmonics.

The changes between the versions are in [CHANGELOG.md](https://github.com/toftul/mielib/blob/main/CHANGELOG.md).

## ToDo

* Search for complex poles of Mie scattering coefficients

## Credits

If you use mielib, please cite it:

```bibtex
@misc{mielib,
	author = {Ivan Toftul},
	title = {{MieLib python library}},
	journal = {GitHub},
	year = {2023},
	url = {https://github.com/toftul/mielib},
	doi = {10.5281/zenodo.8008648}
}
```

The conventions and the theory are from the thesis:

```bibtex
@phdthesis{Toftul2026thesis,
	author = {Ivan Toftul},
	title = {{Resonant Metaphotonics: From Optical Manipulation to Chiral Metasurfaces}},
	school = {Australian National University},
	year = {2026},
	url = {https://hdl.handle.net/1885/733812237}
}
```

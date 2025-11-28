# barc4sr
**BARC** library for **S**ynchrotron **R**adiation

This library was created for facilitating the use of [SRW](https://github.com/ochubar/SRW) 
and [WOFRY](https://github.com/oasys-kit/wofry) for a few routine calculations.

It provides:

- canonical beam and magnet classes (ElectronBeam, MagneticStructure, SynchrotronSource, …)
- magnetic-field generators (bending magnets, arbitrary fields, multi-element fields)
- SRW-powered computations (electron trajectories, wavefronts, power density)
- Wofry-powered computations (CMD)

# Features

### Core modelling
- **ElectronBeam**: canonical storage of second-order moments  
- **MagneticStructure**: bending magnets, arbitrary fields; undulators/ wigglers planned  
- **SynchrotronSource** hierarchy:
  - `BendingMagnetSource`
  - `ArbitraryMagnetSource`

### Magnetic-field generation
- `bm_magnetic_field()` – soft-edge bending magnet model  
- `arb_magnetic_field()` – arbitrary user-defined magnetic fields  
- `multi_bm_magnetic_field()` – composite bending-magnet lattices  
- `multi_arb_magnetic_field()` – multi-field arbitrary lattices  

### SRW-based radiation calculations
- `electron_trajectory()`  
- `wavefront()`  
- `power_density()`  

All return backend-agnostic dictionaries (NumPy arrays + metadata).

### Plotting
Available via `barc4sr.plotting`:
- `plot_electron_trajectory`  
- `plot_wavefront`  
- `plot_power_density`  
- Shared colormaps, styles, and layout helpers  

### I/O utilities
- Save/load: trajectories, wavefronts, power-density maps  
- Simple `.dat` and `.json` formats for interoperability  

---

# Installation

## Basic installation

```
pip install barc4sr
```

This installs the core models, magnetic-field utilities, plotting tools, and I/O helpers.

---

## Enable SRW-based radiation calculations

To perform wavefront, power-density, and trajectory calculations, install SRW bindings:

### Option 1 (recommended if it works on your setup)

```
pip install srwpy
```

### Option 2 (fallback)

```
pip install oasys-srwpy
```

If neither is available for your platform, you can also build SRW from source;  
`barc4sr` will still import, but SRW-based calculations will be unavailable.


## Examples:
Check the examples! You can learn a lot from them.


[![PyPI](https://img.shields.io/pypi/v/barc4sr.svg)](https://pypi.org/project/barc4sr/)
[![Documentation Status](https://readthedocs.org/projects/barc4sr/badge/?version=latest)](https://barc4sr.readthedocs.io/en/latest/)
[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](https://opensource.org/licenses/CECILL-2.1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)](https://doi.org/10.5281/zenodo.0000000)
# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Public API for barc4sr
"""

from __future__ import annotations

from .core.energy import energy_wavelength
from .core.electron_beam import ElectronBeam
from .core.magnetic_structure import MagneticStructure
from .core.sources import (
    SynchrotronSource,
    BendingMagnetSource,
    ArbitraryMagnetSource,
)
from .core.magnetic_fields import (
    bm_magnetic_field,
    arb_magnetic_field,
    multi_bm_magnetic_field,
    multi_arb_magnetic_field
)

from .calculations.trajectory import electron_trajectory

from .calculations.radiation import (
    wavefront,
    power_density,
)

from .calculations.rays import trace_chief_rays


from . import plotting

from .io.rw import (
    write_electron_trajectory,
    read_electron_trajectory,
    write_wavefront,
    read_wavefront,
    write_power_density,
    read_power_density
)

__all__ = [
    # core
    "energy_wavelength",
    "ElectronBeam",
    "MagneticStructure",
    "SynchrotronSource",
    "BendingMagnetSource",
    "ArbitraryMagnetSource",
    "bm_magnetic_field",
    "arb_magnetic_field",
    "multi_bm_magnetic_field",
    "multi_arb_magnetic_field",

    # calculations
    "electron_trajectory",
    "wavefront",
    "power_density",
    "trace_chief_rays",

    # plotting namespace
    "plotting",

    # I/O
    "write_electron_trajectory",
    "read_electron_trajectory",
    "write_wavefront",
    "read_wavefront",
    "write_power_density",
    "read_power_density",
]

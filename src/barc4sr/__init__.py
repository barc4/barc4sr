# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Public API for barc4sr
"""

from __future__ import annotations

from . import plotting
from .calculations.radiation import (
    power_density,
    wavefront,
)
from .calculations.rays import trace_chief_rays
from .calculations.trajectory import electron_trajectory
from .core.electron_beam import ElectronBeam
from .core.energy import energy_wavelength
from .core.magnetic_fields import (
    arb_magnetic_field,
    bm_magnetic_field,
    multi_arb_magnetic_field,
    multi_bm_magnetic_field,
)
from .core.magnetic_structure import MagneticStructure
from .core.sources import (
    ArbitraryMagnetSource,
    BendingMagnetSource,
    SynchrotronSource,
)
from .io.rw import (
    read_electron_trajectory,
    read_power_density,
    read_wavefront,
    write_electron_trajectory,
    write_power_density,
    write_wavefront,
)
from .processing.power import integrate_power_density_window
from .processing.wavefront import integrate_wavefront_window

__all__ = [
    # core
    "ArbitraryMagnetSource",
    "BendingMagnetSource",
    "ElectronBeam",
    "MagneticStructure",
    "SynchrotronSource",
    "arb_magnetic_field",
    "bm_magnetic_field",
    "energy_wavelength",
    "multi_arb_magnetic_field",
    "multi_bm_magnetic_field",

    # calculations
    "electron_trajectory",
    "power_density",
    "trace_chief_rays",
    "wavefront",

    # processing
    "integrate_power_density_window",
    "integrate_wavefront_window",

    # plotting namespace
    "plotting",

    # I/O
    "read_electron_trajectory",
    "read_power_density",
    "read_wavefront",
    "write_electron_trajectory",
    "write_power_density",
    "write_wavefront",
]
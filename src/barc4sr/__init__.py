# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Public API for barc4sr
"""

from __future__ import annotations


from .core.electron_beam import ElectronBeam
from .core.magnetic_structure import MagneticStructure
from .core.sources import (
    SynchrotronSource,
    BendingMagnetSource,
    ArbitraryMagnetSource,
)

from .calculations.trajectory import electron_trajectory

from .calculations.radiation import (
    wavefront,
    power_density,
)


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
    "ElectronBeam",
    "MagneticStructure",
    "SynchrotronSource",
    "BendingMagnetSource",
    "ArbitraryMagnetSource",

    # calculations
    "electron_trajectory",
    "wavefront",
    "power_density",

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

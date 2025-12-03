# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Unified public plotting API for barc4sr.
"""

from __future__ import annotations

from .trajectory import (
    plot_electron_trajectory,
    plot_chief_rays,
    plot_magnetic_field,
    plot_field_and_twiss,
)

from .spatial import (
    plot_wavefront,
    plot_power_density,
    plot_csd,
)

from .spectral import (
    plot_spectrum,
    plot_multiple_spectra,
    plot_beamline_acceptance_scan,
)

__all__ = [
    # trajectory
    "plot_electron_trajectory",
    "plot_chief_rays",
    "plot_magnetic_field",
    "plot_field_and_twiss",

    # spatial
    "plot_wavefront",
    "plot_power_density",
    "plot_csd",

    # spectral
    "plot_spectrum",
    "plot_multiple_spectra",
    "plot_beamline_acceptance_scan",
]
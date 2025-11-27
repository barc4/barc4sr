# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
MagneticStructure: unified container for undulators, wigglers, bending magnets and arbitrary fields.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import physical_constants

from .magnetic_fields import check_magnetic_field_dictionary

CHARGE = physical_constants["atomic unit of charge"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
MASS = physical_constants["electron mass"][0]

class MagneticStructure(object):
    """
    Container for magnetic structure parameters for undulator, wiggler, bending magnet, 
    or arbitrary magnetic field.
    """

    CLASS_NAME = "MagneticStructure"

    def __init__(
        self, magnet_type: str,
        *,
        # generic placement
        center: float = None,
        # arbitrary field only
        magnetic_field: dict = None,
        # undulator / wiggler
        period_length: float = None,
        number_of_periods: float = None,
        K_vertical: float = None, K_horizontal: float = None,
        B_vertical: float = None, B_horizontal: float = None,
        B_vertical_phase: float = 0.0, B_horizontal_phase: float = 0.0,
        B_vertical_symmetry: int = 1, B_horizontal_symmetry: int = 1,
        harmonic: int = 1,
        # bending magnet
        B: float = None, field_length: float = None, 
        edge_length: float = 0.0, extraction_angle: float = None,
    ) -> None:
        """
        Initialize a magnetic structure.

        Parameters
        ----------
        - magnet_type (str):
            - undulator: 'u', 'und', 'undulator'
            - wiggler: 'w', 'wig', 'wiggler'
            - bending magnet: 'bm', 'bending magnet', 'bending_magnet'
            - arbitrary field: 'a', 'arb', 'arbitrary', 'measured'

        - center (float): center position of the magnetin relation to where the electron 
        beam moments are calculated [m].

        ----------------------------------------------------------------            
        Arbitrary magnetic field (magnet_type 'arbitrary' or 'measured')
        ----------------------------------------------------------------            
        - magnetic_field (dict): magnetic field dictionary for arbitrary magnets.

        ----------------------------------------------------------           
        Undulator / wiggler (magnet_type 'undulator' or 'wiggler')
        ----------------------------------------------------------        
        - period_length (float): magnetic period length [m].
        - number_of_periods (int): Number of periods.
        - K_vertical, K_horizontal (float): vertical and horizontal deflection parameters K.
        - B_vertical, B_horizontal (float): vertical and horizontal peak fields [T]. 
        - B_vertical_phase, B_horizontal_phase (float): phase offsets for the vertical
            and horizontal fields [rad].
        - B_vertical_symmetry, B_horizontal_symmetry (int):
            Symmetry flags describing the field parity:
                1  -> symmetric     (B ~ cos(2*pi*n*z/period + phase))
                Z-1  -> antisymmetric (B ~ sin(2*pi*n*z/period + phase))
        - harmonic (int): harmonic index used in source calculations.

        ---------------------------------------------
        Bending magnet (magnet_type 'bending_magnet')
        ---------------------------------------------
        - B (float): magnetic field amplitude [T].
        - field_length (float): length of the region with full field [m].
        - edge_length (float): Soft edge length for field variation (10% to 90%) in meters.
            Assumes a fringe field dependence of B / (1 + ((z - zc) / d)^2)^2.
        - extraction_angle (float): observation angle with respect to the preceeding 
            straight section [rad].

        Raises
        ------
        ValueError
            If magnet_type is invalid, or if required parameters for the
            selected type are missing, or if inconsistent or redundant
            inputs are provided (for example both K and B in one plane).
        """

        mt = magnet_type.lower().strip().replace(" ", "_")
        if mt in ("u", "und", "undulator"):
            self.magnet_type = "undulator"
            self.CLASS_NAME = "Undulator"
        elif mt in ("w", "wig", "wiggler"):
            self.magnet_type = "wiggler"
            self.CLASS_NAME = "Wiggler"
        elif mt in ("bm", "bending_magnet", "bending_magnet"):
            self.magnet_type = "bending_magnet"
            self.CLASS_NAME = "BendingMagnet"
        elif mt in ("a", "arb", "arbitrary", "measured"):
            self.magnet_type = "arbitrary"
            self.CLASS_NAME = "ArbitraryMagnet"
        else:
            raise ValueError(
                f"Invalid magnet_type '{magnet_type}'. "
                "Use undulator, wiggler, bending_magnet, or arbitrary."
            )

        self.center = center

        if self.magnet_type in ("undulator", "wiggler"):
            self._init_undulator(
                period_length=period_length,
                number_of_periods=number_of_periods,
                K_vertical=K_vertical,
                K_horizontal=K_horizontal,
                B_vertical=B_vertical,
                B_horizontal=B_horizontal,
                B_vertical_phase=B_vertical_phase,
                B_horizontal_phase=B_horizontal_phase,
                B_vertical_symmetry=B_vertical_symmetry,
                B_horizontal_symmetry=B_horizontal_symmetry,
                harmonic=harmonic,
            )
        elif self.magnet_type == "bending_magnet":
            self._init_bending_magnet(
                B=B,
                field_length=field_length,
                edge_length=edge_length,
                extraction_angle=extraction_angle,
            )
        else:
            self._init_arbitrary(magnetic_field=magnetic_field)

    def _init_undulator(
        self,
        *,
        period_length: float, number_of_periods: int,
        K_vertical: float, K_horizontal: float,
        B_vertical: float, B_horizontal: float,
        B_vertical_phase: float, B_horizontal_phase: float,
        B_vertical_symmetry: int, B_horizontal_symmetry: int,
        harmonic: int,
    ) -> None:
        """
        Internal initialisation for undulators and wigglers.
        """
        if period_length is None or number_of_periods is None:
            raise ValueError(
                "period_length and number_of_periods are required for "
                "undulators and wigglers."
            )

        self.period_length = period_length
        self.number_of_periods = number_of_periods

        self.B_vertical_symmetry = B_vertical_symmetry
        self.B_horizontal_symmetry = B_horizontal_symmetry

        self.B_vertical_phase = B_vertical_phase
        self.B_horizontal_phase = B_horizontal_phase
        self.harmonic = harmonic

        self.K_vertical = self._resolve_K_plane(
            K=K_vertical,
            B=B_vertical,
            period_length=period_length,
            label="vertical",
        )
        self.K_horizontal = self._resolve_K_plane(
            K=K_horizontal,
            B=B_horizontal,
            period_length=period_length,
            label="horizontal",
        )

    def _init_bending_magnet(
        self,
        *,
        B: float, field_length: float, edge_length: float, extraction_angle: float,
    ) -> None:
        """
        Internal initialisation for bending magnets.
        """
        if B is None or field_length is None:
            raise ValueError(
                "B and field_length are required for a bending magnet."
            )

        self.B = B
        self.field_length = field_length
        self.edge_length = edge_length
        self.extraction_angle = extraction_angle

    def _init_arbitrary(
        self,
        *,
        magnetic_field: dict,
    ) -> None:
        """
        Internal initialisation for arbitrary field magnets.
        """
        if magnetic_field is None:
            raise ValueError(
                "magnetic_field dictionary is required for an arbitrary magnet."
            )

        check_magnetic_field_dictionary(magnetic_field)
        self.magnetic_field = magnetic_field

    @staticmethod
    def _resolve_K_plane(
        *,
        K: float, B: float, period_length: float, label: str,
    ) -> float:
        """
        Internal resolve canonical K.

        Parameters
        ----------
        - K (float): deflection parameter K provided by the user.
        - B (float): peak magnetic field provided by the user [T].
        - period_length (float): undulator period length [m].
        - label (str): text label for error messages ('vertical' or 'horizontal').

        Returns
        -------
        (float): canonical K value for the plane. Returns 0.0 if both K and B are None.

        Raises
        ------
        ValueError
            If both K and B are provided for the same plane.
        """
        if K is not None and B is not None:
            raise ValueError(
                f"Both K and B were provided for the {label} plane. "
                "Provide either K or B, but not both."
            )

        if K is None and B is None:
            return 0.0

        if K is not None:
            return K

        return CHARGE * B * period_length / (2.0 * np.pi * MASS * LIGHT)

    @property
    def B_vertical(self) -> float:
        """
        Vertical peak field [T] derived from K_vertical and period_length.

        Returns
        -------
        (float): vertical peak field in Tesla, or None if this is not an undulator/wiggler
            or if K_vertical is not defined.
        """
        if self.magnet_type not in ("undulator", "wiggler"):
            return None
        if not hasattr(self, "K_vertical") or self.K_vertical is None:
            return None
        return (
            self.K_vertical * (2.0 * np.pi * MASS * LIGHT)
            / (self.period_length * CHARGE)
        )

    @property
    def B_horizontal(self) -> float:
        """
        Horizontal peak field [T] derived from K_horizontal and period_length.

        Returns
        -------
        (float): horizontal peak field in Tesla, or None if this is not an undulator/wiggler
             or if K_horizontal is not defined.
        """
        if self.magnet_type not in ("undulator", "wiggler"):
            return None
        if not hasattr(self, "K_horizontal") or self.K_horizontal is None:
            return None
        return (
            self.K_horizontal * (2.0 * np.pi * MASS * LIGHT)
            / (self.period_length * CHARGE)
        )

    def print_attributes(self) -> None:
        """
        Print all attributes of the magnetic structure instance.
        """
        print(f"\n{self.CLASS_NAME} (magnet_type='{self.magnet_type}'):")
        for name, value in vars(self).items():
            print(f"> {name:20}: {value}")
# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
MagneticStructure: structural container for undulators, wigglers,
bending magnets, and arbitrary magnetic fields.
"""

from __future__ import annotations

from .magnetic_fields import check_magnetic_field_dictionary


class MagneticStructure(object):
    """
    Container for magnetic structure parameters.

    This class stores only structural/device descriptors:
    - magnet family/type
    - geometry or fixed magnet parameters
    - arbitrary field map when relevant

    """

    CLASS_NAME = "MagneticStructure"

    def __init__(
        self,
        magnet_type: str,
        *,
        magnetic_field: dict | None = None,
        period_length: float | None = None,
        number_of_periods: int | None = None,
        B: float | None = None,
        field_length: float | None = None,
        edge_length: float = 0.0,
    ) -> None:
        """
        Initialize a magnetic structure.

        Parameters
        ----------
        magnet_type : str
            Magnetic structure family. Accepted aliases are:
            - undulator: ``"u"``, ``"und"``, ``"undulator"``
            - wiggler: ``"w"``, ``"wig"``, ``"wiggler"``
            - bending magnet: ``"bm"``, ``"bending magnet"``,
              ``"bending_magnet"``
            - arbitrary field: ``"a"``, ``"arb"``, ``"arbitrary"``,
              ``"measured"``
        magnetic_field : dict | None, optional
            Magnetic field dictionary for arbitrary magnets.
            Required when ``magnet_type="arbitrary"``.
        period_length : float | None, optional
            Magnetic period length [m].
            Required for undulators and wigglers.
        number_of_periods : int | None, optional
            Number of magnetic periods.
            Required for undulators and wigglers.
        B : float | None, optional
            Magnetic field amplitude [T].
            Required for bending magnets.
        field_length : float | None, optional
            Length [m] of the region with full field.
            Required for bending magnets.
        edge_length : float, optional
            Soft-edge length [m] for the bending-magnet fringe field.
            Default is 0.0.

        Raises
        ------
        ValueError
            If ``magnet_type`` is invalid, or if required parameters for
            the selected magnetic structure are missing.
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

        if self.magnet_type in ("undulator", "wiggler"):
            self._init_periodic_structure(
                period_length=period_length,
                number_of_periods=number_of_periods,
            )
        elif self.magnet_type == "bending_magnet":
            self._init_bending_magnet(
                B=B,
                field_length=field_length,
                edge_length=edge_length,
            )
        else:
            self._init_arbitrary(magnetic_field=magnetic_field)

    def _init_periodic_structure(
        self,
        *,
        period_length: float | None,
        number_of_periods: int | None,
    ) -> None:
        """
        Internal initialization for undulators and wigglers.

        Parameters
        ----------
        period_length : float | None
            Magnetic period length [m].
        number_of_periods : int | None
            Number of magnetic periods.

        Raises
        ------
        ValueError
            If either parameter is missing.
        """
        if period_length is None or number_of_periods is None:
            raise ValueError(
                "period_length and number_of_periods are required for "
                "undulators and wigglers."
            )

        self.period_length = period_length
        self.number_of_periods = number_of_periods

    def _init_bending_magnet(
        self,
        *,
        B: float | None,
        field_length: float | None,
        edge_length: float,
    ) -> None:
        """
        Internal initialization for bending magnets.

        Parameters
        ----------
        B : float | None
            Magnetic field amplitude [T].
        field_length : float | None
            Length [m] of the region with full field.
        edge_length : float
            Soft-edge length [m] for the fringe field.

        Raises
        ------
        ValueError
            If ``B`` or ``field_length`` is missing.
        """
        if B is None or field_length is None:
            raise ValueError(
                "B and field_length are required for a bending magnet."
            )

        self.B = B
        self.field_length = field_length
        self.edge_length = edge_length

    def _init_arbitrary(
        self,
        *,
        magnetic_field: dict | None,
    ) -> None:
        """
        Internal initialization for arbitrary magnetic fields.

        Parameters
        ----------
        magnetic_field : dict | None
            Magnetic field dictionary.

        Raises
        ------
        ValueError
            If the field dictionary is missing.
        """
        if magnetic_field is None:
            raise ValueError(
                "magnetic_field dictionary is required for an arbitrary magnet."
            )

        check_magnetic_field_dictionary(magnetic_field)
        self.magnetic_field = magnetic_field

    def print_attributes(self) -> None:
        """
        Print all attributes of the magnetic structure instance.
        """
        print(f"\n{self.CLASS_NAME} (magnet_type='{self.magnet_type}'):")
        for name, value in vars(self).items():
            print(f"> {name:20}: {value}")
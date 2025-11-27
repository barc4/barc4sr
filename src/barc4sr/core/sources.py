# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Synchrotron light sources built from ElectronBeam and MagneticStructure.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.constants import physical_constants

from barc4sr.syned.mapping import write_syned_file

from .electron_beam import ElectronBeam
from .magnetic_structure import MagneticStructure

CHARGE = physical_constants["atomic unit of charge"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
MASS = physical_constants["electron mass"][0]
PLANCK = physical_constants["Planck constant"][0]

class SynchrotronSource(object):
    """
    Base container for a synchrotron radiation sources.

    A SynchrotronSource combines an ElectronBeam and a MagneticStructure. 
    """

    def __init__(self, electron_beam: ElectronBeam, magnetic_structure: MagneticStructure) -> None:
        """
        Initialize a synchrotron source container

        Parameters
        ----------
        - electron_beam (ElectronBeam): an isntance of ElectronBeam
        - magnetic_structure (MagneticStructure): an isntance of MagneticStructure
        """
        self.ElectronBeam = electron_beam
        self.MagneticStructure = magnetic_structure

    def __getattr__(self, name: str):
        """
        Retrieves an attribute from either the ElectronBeam or MagneticStructure instances if it exists.

        Parameters
        ----------
        - name (str): Name of the requested attribute.

        Returns
        -------
        - Value of the attribute if it exists on ElectronBeam or MagneticStructure.

        Raises
        ------
        AttributeError
            If the attribute is not found on the source, the electron
            beam, or the magnetic structure.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        if hasattr(self.ElectronBeam, name):
            return getattr(self.ElectronBeam, name)
        if hasattr(self.MagneticStructure, name):
            return getattr(self.MagneticStructure, name)
        raise AttributeError(f"'SynchrotronSource' object has no attribute '{name}'")

    def write_syned_config(self, json_file: str, light_source_name: str = None) -> None:
        """
        Write a SYNED JSON configuration file.

        Parameters
        ----------
        - json_file (str): The path to the JSON file where the dictionary will be written.
        - light_source_name (str): The name of the light source.
        """
        if light_source_name is None:
            light_source_name = json_file.split("/")[-1].replace(".json", "")

        write_syned_file(json_file, light_source_name, self.ElectronBeam, self.MagneticStructure)

    def print_attributes(self) -> None:
        """
        Print attributes of the electron beam and magnetic structure.
        """
        self.ElectronBeam.print_attributes()
        self.MagneticStructure.print_attributes()

# ------------------------------------------------------------------
# Arbitrary/measured magnetic field
# ------------------------------------------------------------------

class ArbitraryMagnetSource(SynchrotronSource):
    """
    SR source based on a user-defined arbitrary magnetic field.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize an arbitrary magnetic source.

        Parameters
        ----------
        - electron_beam (ElectronBeam)
        - magnetic_structure (MagneticStructure)
        """
        super().__init__(**kwargs)

        if getattr(self.MagneticStructure, "magnet_type", None) != "arbitrary":
            raise ValueError(
                "ArbitraryMagnetSource requires MagneticStructure with "
                "magnet_type='arbitrary'."
            )
        self._original_magnetic_field = None

    def configure(self, *, si=-1e23, sf=1e23, dS=None, reset=False, verbose=False) -> None:
        """
        Configure an arbitrary magnetic source.

        Trim and optionally recenter the arbitrary magnetic field.      

        Parameters
        ----------
        - si, sf (float): lower and upper trimming bounds [m]. 
        - dS (float): shift applied to the magnetic-field grid [m].
        - reset (bool): if True, restore the original untrimmed field before
            applying dS and trimming.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If trimming removes all samples, or if the magnetic field
            dictionary is missing mandatory keys.
        """

        self._ensure_original_field()

        if reset:
            self.reset_field()

        if dS is not None:
            self.recenter(dS=dS, verbose=False)
        self.trim(si=si, sf=sf, verbose=False)

        if verbose:
            self._verbose()

    def recenter(self, *, dS=None, verbose=False) -> None:
        """
        Recenter the arbitrary magnetic field.

        The recentering is performed on the current magnetic field stored in
        MagneticStructure.magnetic_field and does not crop any data.

        Parameters
        ----------
        - dS (float): shift applied to the magnetic-field grid [m]. If None, the midpoint
            of the current s-interval is used.
        - verbose (bool): if True, prints to the prompt
        """
        self._ensure_original_field()

        mf = self.MagneticStructure.magnetic_field
        if mf is None:
            raise ValueError(
                "magnetic_field dictionary is not set in MagneticStructure "
                "(magnet_type='arbitrary' expected)."
            )

        try:
            s = np.asarray(mf["s"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "The magnetic_field dictionary must contain key 's'."
            ) from exc

        if s.ndim != 1:
            raise ValueError("'s' must be a 1D array.")

        mid = 0.5 * (s[0] + s[-1])
        center_val = mid if dS is None else float(dS)
        delta = center_val - mid

        mf_shift = {}
        for k, v in mf.items():
            if k == "s":
                mf_shift["s"] = s + delta
            else:
                mf_shift[k] = v

        self.MagneticStructure.magnetic_field = mf_shift
        self.MagneticStructure.center = center_val

        if verbose:
            self._verbose()

    def trim(self, *, si=-1e23, sf=1e23, verbose=False) -> None:
        """
        Trim the arbitrary magnetic field.

        The trimming is performed on the current magnetic field stored in
        MagneticStructure.magnetic_field. If the field has been recentered
        (via recenter or configure), the trimming bounds si and sf are
        interpreted in that recentered coordinate system.

        Parameters
        ----------
        - si, sf (float): lower and upper trimming bounds [m] in the
            current coordinate system (usually recentered).
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If trimming removes all samples, or if the magnetic field
            dictionary is missing mandatory keys.
        """
        self._ensure_original_field()

        mf = self.MagneticStructure.magnetic_field
        if mf is None:
            raise ValueError(
                "magnetic_field dictionary is not set in MagneticStructure "
                "(magnet_type='arbitrary' expected)."
            )

        try:
            s = np.asarray(mf["s"], dtype=float)
            B = np.asarray(mf["B"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "The magnetic_field dictionary must contain keys 's' and 'B'."
            ) from exc

        if s.ndim != 1:
            raise ValueError("'s' must be a 1D array.")
        if B.ndim != 2 or B.shape[1] != 3 or B.shape[0] != s.size:
            raise ValueError(
                "'B' must be a (N, 3) array and its first dimension must "
                "match len(s)."
            )

        if si > sf:
            si, sf = sf, si

        mask = (s >= si) & (s <= sf)
        if not np.any(mask):
            raise ValueError(
                f"Trimming interval [{si}, {sf}] produced an empty dataset."
            )

        s_trim = s[mask]

        mf_trim = {}
        for k, v in mf.items():
            if k == "s":
                mf_trim["s"] = s_trim
                continue
            arr = np.asarray(v)
            if arr.shape[:1] == (s.size,):
                mf_trim[k] = arr[mask, ...]
            else:
                mf_trim[k] = v

        self.MagneticStructure.magnetic_field = mf_trim
        self.MagneticStructure.center = (mf_trim["s"][-1]+mf_trim["s"][0])/2

        if verbose:
            self._verbose()

    def reset_field(self) -> None:
        """
        Reset the magnetic field to its original untrimmed state.
        """
        if self._original_magnetic_field is None:
            raise RuntimeError(
                "reset_field() called before configure(). "
                "There is no stored original magnetic field to restore."
            )

        self.MagneticStructure.magnetic_field = deepcopy(self._original_magnetic_field)
        self.MagneticStructure.center = 0.0

    def _ensure_original_field(self) -> None:
        """
        Internal helper to cache the original magnetic field.

        Makes a deepcopy of MagneticStructure.magnetic_field on first use.
        """
        if self._original_magnetic_field is None:
            if self.MagneticStructure.magnetic_field is None:
                raise ValueError(
                    "magnetic_field dictionary is not set in MagneticStructure "
                    "(magnet_type='arbitrary' expected)."
                )
            self._original_magnetic_field = deepcopy(self.MagneticStructure.magnetic_field)

    def _verbose(self) -> None:
        """
        Internal helper print info on current magnetic field.
        """    
        s_out = self.MagneticStructure.magnetic_field["s"]
        print('\n>>>>>>>>>>> User-defined arbitrary magnetic field <<<<<<<<<<<\n')
        print(f"\t>> Field (re)centered at s = {self.MagneticStructure.center:.6f} m")
        print(f"\t>> Span: [{s_out[0]:.6f}, {s_out[-1]:.6f}] m")
        if s_out.size > 1:
            print(f"\t>> Step size: {s_out[1] - s_out[0]:.6g} m; N = {s_out.size}")
        else:
            print("Only one sample (no trimming).")

# ------------------------------------------------------------------
# Bending magnet
# ------------------------------------------------------------------

class BendingMagnetSource(SynchrotronSource):
    """
    SR bending magnet source
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a bending magnet source.

        Parameters
        ----------
        - electron_beam (ElectronBeam)
        - magnetic_structure (MagneticStructure)
        """
        super().__init__(**kwargs)

        if getattr(self.MagneticStructure, "magnet_type", None) != "bending_magnet":
            raise ValueError(
                "BendingMagnetSource requires MagneticStructure with "
                "magnet_type='bending_magnet'."
            )

    def configure(self,
        *,
        center: float = None,
        extraction_angle: float = None,
        verbose: bool = False,
    ) -> None:
        """
        Configure bending magnet source.

        From B and the electron beam energy, the radius and critical energy can be computed
        on the fly and optionally printed.

        The extraction geometry is defined by either the center position [m] of the magnetic
        field or the extraction angle [rad] taken from the preceeding straight section. 

        Parameters
        ----------
        - center (float): center of the magnetic field [m]. The emission direction (optical
            axis) is taken from the tangent to the trajectory at this position. 
        - extraction_angle (float): extraction angle [rad] taken from the previous straight
            section. 0 [rad] is the magnet entrance and L/R [rad] is the magnet exit. If 
            provided, the center is derived from it and stored in MagneticStructure.center.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If B is not set, or if both center and extraction_angle are
            provided at the same time.
        """
        _ = self.B

        if verbose:
            print('\n>>>>>>>>>>> bending magnet <<<<<<<<<<<\n')
            print("> Properties:")
            print(f"\t>> B = {self.B:.6f} T")
            print(f"\t>> R = {self.radius:.6f} m")
            print(f"\t>> critical energy = {self.critical_energy:.3f} eV")

        if center is not None and extraction_angle is not None:
            raise ValueError("Provide only one of 'center' or 'extraction_angle', not both.")

        if center is None and extraction_angle is None:
            if self.MagneticStructure.center is not None:
                center = self.MagneticStructure.center
            elif self.MagneticStructure.extraction_angle is not None:
                extraction_angle = self.MagneticStructure.extraction_angle
            else:
                center = 0.0

        self._set_geometry(center=center, extraction_angle=extraction_angle, verbose=verbose)

    def _set_geometry(self, *,
        center: float, extraction_angle: float,
        verbose: bool = False,
    ) -> None:
        """
        Internal helper to set extraction geometry.

        Parameters
        ----------
        - center (float): center of the magnetic field [m]. The emission direction (optical
            axis) is taken from the tangent to the trajectory at this position. 
        - extraction_angle (float): extraction angle [rad] taken from the previous straight
            section. 0 [rad] is the magnet entrance and L/R [rad] is the magnet exit. If 
            provided, the center is derived from it and stored in MagneticStructure.center.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If both center and extraction_angle are None, or if the
            resulting angle is out of range.
        """
        if center is None and extraction_angle is None:
            raise ValueError("Provide 'center' (m) or 'extraction_angle' (rad).")

        L = self.MagneticStructure.field_length
        if L is None:
            raise ValueError("MagneticStructure.field_length must be defined.")

        R = self.radius

        half_arc = 0.5 * L / R
        full_arc = L / R

        if extraction_angle is None:
            center_val = float(center)
            extraction_angle = half_arc - center_val / R
        else:
            extraction_angle = float(extraction_angle)
            center_val = (half_arc - extraction_angle) * R

        if not (0.0 <= extraction_angle <= full_arc):
            raise ValueError(
                f"extraction_angle={extraction_angle:.6g} rad out of range [0, {full_arc:.6g}]"
            )

        magnetic_field_center = 0.5 * L - center_val

        self.MagneticStructure.center = center_val
        self.MagneticStructure.extraction_angle = extraction_angle

        if verbose:
            print("> Extraction geometry:")
            print(f"\t>> dist. from BM center    : {center_val:.6f} m")
            print(f"\t>> dist. from BM entrance  : {magnetic_field_center:.6f} m")
            print(f"\t>> extraction angle        : {extraction_angle*1e3:.3f} mrad")
            print(f"\t>> BM arc                  : {2.0*half_arc*1e3:.3f} mrad "
                  f"(L={L:.3f} m, R={R:.3f} m)")

    @property
    def B(self) -> float:
        """
        Magnetic field amplitude [T].
        """
        B = self.MagneticStructure.B
        if B is None:
            raise ValueError("Magnetic field B is not defined in MagneticStructure.")
        return B

    @B.setter
    def B(self, value: float) -> None:
        """
        Set magnetic field amplitude [T].
        """
        self.MagneticStructure.B = value

    @property
    def center(self) -> float:
        """
        Center of the magnetic field [m].
        """
        return self.MagneticStructure.center

    @center.setter
    def center(self, value: float) -> None:
        """
        Set center of the magnetic field [m].
        """
        self.MagneticStructure.center = value

    @property
    def extraction_angle(self) -> float:
        """
        Extraction angle [rad] taken from the previous straight section.
        """
        return self.MagneticStructure.extraction_angle

    @extraction_angle.setter
    def extraction_angle(self, value: float) -> None:
        """
        Set extraction angle [rad] taken from the previous straight section.
        """
        self.MagneticStructure.extraction_angle = value

    @property
    def radius(self) -> float:
        """
        Bending radius [m], computed from B and electron beam energy.
        """
        B = self.B
        gamma = self.gamma()
        e_speed = LIGHT * np.sqrt(1.0 - 1.0 / gamma**2)
        return gamma * MASS * e_speed / (CHARGE * abs(B))

    @property
    def critical_energy(self) -> float:
        """
        Critical photon energy [eV], computed from B and electron beam energy.
        """
        B = self.B
        gamma = self.gamma()
        return (3.0 * PLANCK * B * gamma**2) / (4.0 * np.pi * MASS)
    
# ------------------------------------------------------------------
# Undulator
# ------------------------------------------------------------------

class UndulatorSource(SynchrotronSource):
    """
    Placeholder: SR Undulator source.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a undulator source.

        Parameters
        ----------
        - electron_beam (ElectronBeam)
        - magnetic_structure (MagneticStructure)
        """
        super().__init__(**kwargs)

        if getattr(self.MagneticStructure, "magnet_type", None) != "undulator":
            raise ValueError(
                "WigglerSource requires a MagneticStructure with "
                "magnet_type='undulator'."
            )

    def initialize(self, **kwargs) -> None:
        """
        Configure the undulator source.

        Parameters
        ----------
        **kwargs
            Reserved for future configuration options.

        Raises
        ------
        NotImplementedError
            Always raised until the undulator model is implemented.
        """
        raise NotImplementedError(
            "WigglerSource.initialize() is not implemented yet. "
            "Please implement the undulator radiation model."
        )

# ------------------------------------------------------------------
# Wiggler
# ------------------------------------------------------------------

class WigglerSource(SynchrotronSource):
    """
    Placeholder: SR Wiggler source.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a wiggler source.

        Parameters
        ----------
        - electron_beam (ElectronBeam)
        - magnetic_structure (MagneticStructure)
        """
        super().__init__(**kwargs)

        if getattr(self.MagneticStructure, "magnet_type", None) != "wiggler":
            raise ValueError(
                "WigglerSource requires a MagneticStructure with "
                "magnet_type='wiggler'."
            )

    def initialize(self, **kwargs) -> None:
        """
        Configure the wiggler source.

        Parameters
        ----------
        **kwargs
            Reserved for future configuration options.

        Raises
        ------
        NotImplementedError
            Always raised until the wiggler model is implemented.
        """
        raise NotImplementedError(
            "WigglerSource.initialize() is not implemented yet. "
            "Please implement the wiggler radiation model."
        )
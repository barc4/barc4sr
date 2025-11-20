#!/bin/python

"""
This module provides the barc4sr classes:

- ElectronBeam
- MagneticStructure
- SynchrotronSource
- ArbitraryMagnetSource(SynchrotronSource)
- BendingMagnetSource(SynchrotronSource)
- UndulatorSource(SynchrotronSource)
- Wiggler(SynchrotronSource) -> not implemented

"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '2024.11.25'
__changed__ = '2025.11.19'

import os
from copy import deepcopy

import numpy as np
import scipy.optimize as opt
from numba import njit, prange
from scipy.constants import physical_constants
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import erf, jv

from barc4sr.aux_energy import energy_wavelength, get_gamma
from barc4sr.aux_magnetic_fields import check_magnetic_field_dictionary
from barc4sr.aux_syned import write_syned_file

ALPHA =  physical_constants["fine-structure constant"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
MASS = physical_constants["electron mass"][0]
PI = np.pi
PLANCK = physical_constants["Planck constant"][0]
Z0 = physical_constants["characteristic impedance of vacuum"][0]


class ElectronBeam(object):
    """
    Container for electron beam parameters.

    The canonical state is given by the six transverse second-order moments: 
        <x^2>, <x*xp>, <xp^2>, <y^2>, <y*yp>, <yp^2>.
    """

    CLASS_NAME = "ElectronBeam"

    def __init__(self,
        *,
        energy: float = None, energy_spread: float = None, current: float = None,
        moment_xx: float = None, moment_xxp: float = None, moment_xpxp: float = None,
        moment_yy: float = None, moment_yyp: float = None, moment_ypyp: float = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize an electron beam container.

        Parameters
        ----------
        - energy (float): beam energy in GeV.
        - energy_spread (float): RMS relative energy spread (dE/E).
        - current (float): average beam current in A.
        - moment_xx, moment_xxp, moment_xpxp (float):
            horizontal second-order moments <x^2>, <x*xp>, <xp^2> in units of m^2, m*rad, rad^2.
        - moment_yy, moment_yyp, moment_ypyp (float) 
            vertical second-order moments <y^2>, <y*yp>, <yp^2> in units of m^2, m*rad, rad^2.
        - verbose (bool): if True, prints to the prompt
        """
        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current
        self.number_of_bunches = 1

        self.moment_xx = moment_xx
        self.moment_xxp = moment_xxp
        self.moment_xpxp = moment_xpxp
        self.moment_yy = moment_yy
        self.moment_yyp = moment_yyp
        self.moment_ypyp = moment_ypyp

        moments = (
            self.moment_xx,
            self.moment_xxp,
            self.moment_xpxp,
            self.moment_yy,
            self.moment_yyp,
            self.moment_ypyp,
        )

        self._initialised = all(m is not None for m in moments)

        if verbose and self._initialised:
            self.print_rms()

    def _ensure_not_initialised(self) -> None:
        if self._initialised:
            raise RuntimeError(
                "ElectronBeam is already initialised and cannot be reconfigured."
            )

    def from_twiss(self, energy: float, energy_spread: float, current: float,
        beta_x: float, alpha_x: float, eta_x: float, etap_x: float,
        beta_y: float, alpha_y: float, eta_y: float, etap_y: float,
        *,
        emittance: float = None, coupling: float = None,
        emittance_x: float = None, emittance_y: float = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize ElectronBeam from Twiss parameters and emittances.

        Parameters
        ----------
        - energy (float): beam energy in GeV.
        - energy_spread (float): RMS relative energy spread (dE/E).
        - current (float): average beam current in A.
        - beta_x, beta_y (float): horizontal and vertical beta functions in m.
        - alpha_x, alpha_y (float): horizontal and vertical alpha functions.
        - eta_x, eta_y (float):) horizontal and vertical dispersion functions in m.
        - etap_x, etap_y (float): derivatives of dispersion in rad.
        - emittance (float): total emittance (used with coupling if emittance_x and emittance_y are not given).
        - coupling (float): coupling ratio horizontal and vertical when emittance is given.
        - emittance_x, emittance_y (float): horizontal and vertical emittances in m*rad.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        RuntimeError
            If the beam is already initialized.
        ValueError
            If emittance information is incomplete.
        """
        self._ensure_not_initialised()

        if emittance_x is None or emittance_y is None:
            if emittance is None or coupling is None:
                raise ValueError(
                    "Either (emittance and coupling) or (emittance_x and emittance_y) "
                    "must be provided."
                )
            if emittance_x is None:
                emittance_x = emittance / (1.0 + coupling)
            if emittance_y is None:
                emittance_y = emittance * coupling / (1.0 + coupling)

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current

        sigE2 = energy_spread ** 2

        self.moment_xx = emittance_x * beta_x + sigE2 * eta_x * eta_x
        self.moment_xxp = -emittance_x * alpha_x + sigE2 * eta_x * etap_x
        self.moment_xpxp = (
            emittance_x * (1.0 + alpha_x * alpha_x) / beta_x
            + sigE2 * etap_x * etap_x
        )

        self.moment_yy = emittance_y * beta_y + sigE2 * eta_y * eta_y
        self.moment_yyp = -emittance_y * alpha_y + sigE2 * eta_y * etap_y
        self.moment_ypyp = (
            emittance_y * (1.0 + alpha_y * alpha_y) / beta_y
            + sigE2 * etap_y * etap_y
        )

        self._initialised = True

        if verbose:
            self.print_rms()

    def from_rms(self, energy: float, energy_spread: float, current: float,
        x: float, xp: float, y: float, yp: float, xxp: float = 0.0,  yyp: float = 0.0,
        verbose: bool = False
    ) -> None:
        """
        Initialize ElectronBeam from RMS beam sizes and divergences.

        Parameters
        ----------
        - energy (float): beam energy in GeV.
        - energy_spread (float): RMS relative energy spread (dE/E).
        - current (float): average beam current in A.
        - x, y (float): horizontal and vertical RMS sizes in m.
        - xp, yp (float): horizontal and vertical RMS divergences in rad.
        - xxp, yyp (float): cross-correlation terms <x*xp> and <y*yp> in m*rad.
        - verbose (bool): if True, prints to the prompt

        """
        self._ensure_not_initialised()

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current

        self.moment_xx = x * x
        self.moment_xxp = xxp
        self.moment_xpxp = xp * xp
        self.moment_yy = y * y
        self.moment_yyp = yyp
        self.moment_ypyp = yp * yp

        self._initialised = True

        if verbose:
            self.print_rms()

    def from_moments(
        self, energy: float, energy_spread: float, current: float,
        moment_xx: float, moment_xxp: float, moment_xpxp: float,
        moment_yy: float, moment_yyp: float, moment_ypyp: float,
        verbose: bool = False
    ) -> None:
        """
        Initialize ElectronBeam directly from the six transverse second moments.

        Parameters
        ----------
        - energy (float): beam energy in GeV.
        - energy_spread (float): RMS relative energy spread (dE/E).
        - current (float): average beam current in A.
        - moment_xx, moment_xxp, moment_xpxp (float):
            horizontal second-order moments <x^2>, <x*xp>, <xp^2> in units of m^2, m*rad, rad^2.
        - moment_yy, moment_yyp, moment_ypyp (float): 
            vertical second-order moments <y^2>, <y*yp>, <yp^2> in units of m^2, m*rad, rad^2.
        - verbose (bool): if True, prints to the prompt
        """
        self._ensure_not_initialised()

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current

        self.moment_xx = moment_xx
        self.moment_xxp = moment_xxp
        self.moment_xpxp = moment_xpxp
        self.moment_yy = moment_yy
        self.moment_yyp = moment_yyp
        self.moment_ypyp = moment_ypyp

        self._initialised = True
        if verbose:
            self.print_rms()

    @property
    def e_x(self) -> float:
        """ Horizontal RMS size sigma_x in m."""
        return float(np.sqrt(self.moment_xx)) if self.moment_xx is not None else np.nan

    @property
    def e_y(self) -> float:
        """Vertical RMS size sigma_y in m."""
        return float(np.sqrt(self.moment_yy)) if self.moment_yy is not None else np.nan

    @property
    def e_xp(self) -> float:
        """Horizontal RMS divergence sigma_xp in rad."""
        return (
            float(np.sqrt(self.moment_xpxp))
            if self.moment_xpxp is not None
            else np.nan
        )

    @property
    def e_yp(self) -> float:
        """Vertical RMS divergence sigma_yp in rad."""
        return (
            float(np.sqrt(self.moment_ypyp))
            if self.moment_ypyp is not None
            else np.nan
        )

    def gamma(self) -> float:
        """
        Return the Lorentz factor gamma for the current beam energy.

        Returns
        -------
        float
            Lorentz factor gamma.
        """
        return get_gamma(self.energy_in_GeV)

    def print_rms(self) -> None:
        """
        Print RMS sizes and divergences in both planes.
        """
        print("electron beam:")
        print(f"\t>> x/xp = {self.e_x * 1e6:0.2f} um vs. {self.e_xp * 1e6:0.2f} urad")
        print(f"\t>> y/yp = {self.e_y * 1e6:0.2f} um vs. {self.e_yp * 1e6:0.2f} urad")

    def print_attributes(self) -> None:
        """
        Print all attributes of the electron beam instance.
        """
        print("\nElectronBeam():")
        for name, value in vars(self).items():
            print(f"> {name:16}: {value}")

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

        return CHARGE * B * period_length / (2.0 * PI * MASS * LIGHT)

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
            self.K_vertical * (2.0 * PI * MASS * LIGHT)
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
            self.K_horizontal * (2.0 * PI * MASS * LIGHT)
            / (self.period_length * CHARGE)
        )

    def print_attributes(self) -> None:
        """
        Print all attributes of the magnetic structure instance.
        """
        print(f"\n{self.CLASS_NAME} (magnet_type='{self.magnet_type}'):")
        for name, value in vars(self).items():
            print(f"> {name:20}: {value}")
            
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

    def configure(self, *, si=-1e23, sf=1e23, center=None, verbose=False) -> None:
        """
        Configure an arbitrary magnetic source.

        Trim and optionally recenter the arbitrary magnetic field.      

        Parameters
        ----------
        - si, sf (float): lower and upper trimming bounds [m]. 
        - center (float): new center position [m].
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If trimming removes all samples, or if the magnetic field
            dictionary is missing mandatory keys.
        """


        if self._original_magnetic_field is None:
            if self.MagneticStructure.magnetic_field is None:
                raise ValueError(
                    "magnetic_field dictionary is not set in MagneticStructure "
                    "(magnet_type='arbitrary' expected)."
                )
            self._original_magnetic_field = deepcopy(self.MagneticStructure.magnetic_field)

        mf = self._original_magnetic_field

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

        mid = 0.5 * (s_trim[0] + s_trim[-1])
        center_val = mid if center is None else float(center)

        delta = center_val - mid
        if delta != 0.0:
            mf_trim["s"] = mf_trim["s"] + delta

        self.MagneticStructure.magnetic_field = mf_trim
        self.MagneticStructure.center = center_val

        if verbose:
            s_out = mf_trim["s"]
            print('\n>>>>>>>>>>> User-defined arbitrary magnetic field <<<<<<<<<<<\n')
            print(f"\t>> Field recentered at s = {center_val:.6f} m")
            print(f"\t>> Span: [{s_out[0]:.6f}, {s_out[-1]:.6f}] m")
            if s_out.size > 1:
                print(f"\t>> Step size: {s_out[1] - s_out[0]:.6g} m; N = {s_out.size}")
            else:
                print("Only one sample after trimming.")

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
        return (3.0 * PLANCK * B * gamma**2) / (4.0 * PI * MASS)
    
# ------------------------------------------------------------------
# Undulator
# ------------------------------------------------------------------

class UndulatorSource(SynchrotronSource):
    """
    Class representing an undulator radiation source, which combines an electron beam and 
    a magnetic structure.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes an instance of the UndulatorSource class.

        Args:
            electron_beam (ElectronBeam): An instance of the ElectronBeam class 
               representing the electron beam parameters.
            magnetic_structure (MagneticStructure): An instance of the MagneticStructure 
               class representing the magnetic structure parameters.
        """

        self.wavelength = None
        self.sigma_u = None
        self.sigma_up = None

        self.first_ring = None

        self.on_axis_flux = None
        self.central_cone_flux = None
        self.coherent_flux = None
        self.brilliance = None

        self.coherent_fraction = None

        self.total_power = None
        self.power_though_slit = None

        super().__init__(**kwargs)

    def initialize(self, **kwargs) -> None:
        """
        Sets the parameters of the undulator.

        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            B_horizontal (float): Horizontal magnetic field [T].
            B_vertical (float): Vertical magnetic field [T].
            K_horizontal (float): Horizontal deflection parameter.
            K_vertical (float): Vertical deflection parameter.
            direction (str): Direction of the undulator ('v' for vertical, 'h' for horizontal, 'b' for both).
            wavelength (float): The wavelength of the undulator radiation [m].
            harmonic (int): Harmonic number.
            mth_emittance (int): Method for emittance calculation. Default is 0.
            mth_fillament_emittance (int): Method for filament emittance calculation. Default is 0.
            center_undulator (float): Center position of the undulator. Default is 0.
            center_straight_section (float): Center position of the straight section. Default is 0.
            complete (bool): If True, computes full central cone and power-through-slit characteristics.
        """
        verbose = kwargs.get('verbose', False)
        direction = kwargs.get('direction', None)
        wavelength = kwargs.get('wavelength', None)

        complete = kwargs.get('complete', False)

        self.MagneticStructure.B_horizontal = None
        self.MagneticStructure.K_horizontal = None
        self.MagneticStructure.B_vertical = None
        self.MagneticStructure.K_vertical = None
        
        if 'B_horizontal' in kwargs:
            self.MagneticStructure.B_horizontal = kwargs['B_horizontal']
        if 'B_vertical' in kwargs:
            self.MagneticStructure.B_vertical = kwargs['B_vertical']
        if 'K_horizontal' in kwargs:
            self.MagneticStructure.K_horizontal = kwargs['K_horizontal']
        if 'K_vertical' in kwargs:
            self.MagneticStructure.K_vertical = kwargs['K_vertical']

        piloting = [wavelength, self.B_horizontal, self.B_vertical, self.K_horizontal, self.K_vertical]
        if all(param is None for param in piloting):
            raise ValueError("Please, provide either the wavelength [m], the magnetic fields [T] or the deflection parameters Kx and/or Ky.")

        if wavelength is not None:
            self.MagneticStructure.harmonic = kwargs.get('harmonic', None)
            self.set_resonant_energy(energy=energy_wavelength(wavelength, unity='m'), 
                                     harmonic=self.harmonic, direction=direction, 
                                     verbose=verbose)
            self.wavelength = wavelength
        else:
            self.MagneticStructure.harmonic = kwargs.get('harmonic', 1)
            if self.B_horizontal is not None or self.B_vertical is not None:
                self.set_K_from_magnetic_field(self.B_horizontal, self.B_vertical, verbose=verbose)
            elif self.K_horizontal is not None or self.K_vertical is not None:
                self.set_magnetic_field_from_K(self.K_horizontal, self.K_vertical, verbose=verbose)

            K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)
            gamma = self.gamma()
            self.wavelength = self.period_length/(2 * self.harmonic * gamma ** 2)*(1+(K**2)/2) 
            
        mth_emittance = kwargs.get('mth_emittance', 0)
        mth_fillament_emittance = kwargs.get('mth_fillament_emittance', 0)

        cund = kwargs.get('center_undulator', 0)
        css = kwargs.get('center_straight_section', 0)

        self.MagneticStructure.center = 0

        if verbose:
            print('\n>>>>>>>>>>> beam phase-space characteristics <<<<<<<<<<<')
            self.print_rms()

        self.set_filament_emittance(verbose=verbose, wavelength=wavelength, mth=mth_fillament_emittance)
        self.set_radiation_rings(verbose=verbose)
        self.set_waist(verbose=verbose, center_undulator=cund, center_straight_section=css)
        self.set_emittance(verbose=verbose, mth=mth_emittance)

        if verbose:
            print('\n>>>>>>>>>>> beam spectral characteristics <<<<<<<<<<<')

        self.get_coherent_fraction(verbose=verbose)
        self.set_on_axis_flux(verbose=False)
        self.get_central_cone_flux(mtd=0, verbose=verbose)

        self.get_coherent_flux(verbose=verbose)
        self.get_brilliance(verbose=verbose)

        if verbose:
            print('\n>>>>>>>>>>> power characteristics <<<<<<<<<<<')
        self.set_total_power(verbose=verbose)
        if complete:
            self.get_power_through_slit(verbose=verbose)

    def set_resonant_energy(self, energy: float, direction: str, verbose: bool=False,
                            **kwargs) -> None:
        """
        Sets the undulator K-value based on the specified resonant energy and harmonic.

        This method calculates the undulator parameter K required to achieve the given resonant 
        energy at a specified harmonic. If the harmonic number is not provided, the function 
        searches for the lowest harmonic that meets the K-value constraints.

        Args:
            energy (float): Resonant energy in electron volts (eV).
            direction (str): Direction of the undulator magnetic field.
                - 'v': Vertical polarization
                - 'h': Horizontal polarization
                - 'b': Both (equal distribution in vertical and horizontal)
            verbose (bool, optional): If True, prints additional information. Default is False.
            **kwargs:
                - harmonic (int, optional): The harmonic number to use in the calculation. If None, 
                  the function searches for a valid harmonic.
                - even_harmonics (bool, optional): If True, even harmonics are considered. Default is False.
                - Kmin (float, optional): Minimum allowed value for the K parameter. Default is 0.05.

        Raises:
            ValueError: If no valid harmonic is found within the search limit.
            ValueError: If the provided direction is not one of ['v', 'h', 'b'].

        """
        
        harmonic = kwargs.get('harmonic', None)
        even_harmonics = kwargs.get('even_harmonics', False)
        Kmin = kwargs.get('Kmin', 0.05)

        self.wavelength = energy_wavelength(energy, 'eV')
        gamma = self.gamma()

        if harmonic is not None:
            K = np.sqrt(2)*np.sqrt(((2 * harmonic * self.wavelength * gamma ** 2)/self.period_length)-1)
        else:
            n = starting_harmonic = 1
            harmonic = theta = 0
            while harmonic == 0:
                try:
                    arg_sqrt = 2 * ((2 * n * self.wavelength * gamma ** 2) / self.period_length - 1 - (gamma * theta) ** 2)
                    if arg_sqrt>=0:
                        K = np.sqrt(arg_sqrt)
                    else:
                        K=-1
                    if K >= Kmin:
                        if n % 2 == 0 and even_harmonics:
                            harmonic = int(n)
                        else:
                            harmonic = int(n)
                except ValueError:
                    K = None
                if even_harmonics or (even_harmonics is False and starting_harmonic%2==0):
                    n += 1
                else:
                    n += 2
                if n > 21:
                    raise ValueError("No valid harmonic found.")
                
        self.MagneticStructure.harmonic = harmonic

        if "v" in direction:
            self.MagneticStructure.K_vertical = K
            self.MagneticStructure.K_horizontal = 0
        elif "h" in direction:
            self.MagneticStructure.K_vertical = 0
            self.MagneticStructure.K_horizontal = K
        elif 'b' in direction:
            self.MagneticStructure.K_vertical = K*np.sqrt(1/2)
            self.MagneticStructure.K_horizontal = K*np.sqrt(1/2)
        else:
            raise ValueError("invalid value: direction should be in ['v','h','b']")
        self.set_magnetic_field_from_K(self.MagneticStructure.K_horizontal, self.MagneticStructure.K_vertical)
        if verbose:
            print(f"undulator resonant energy set to {energy:.3f} eV (harm. n°: {harmonic}) with:")
            print(f"\t>> Kh: {self.K_horizontal:.6f}")
            print(f"\t>> Kv: {self.K_vertical:.6f}")

    def get_resonant_energy(self, harmonic: int) -> float:
        """
        Returns the resonant energy based on the provided K-value, harmonic number, and electron beam energy.
        Args:
            harmonic (int): The harmonic number.
        """
        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)
        gamma = self.gamma()
        wavelength = self.period_length/(2 * harmonic * gamma ** 2)*(1+(K**2)/2) 
        energy = energy_wavelength(wavelength, 'm')
        return energy
    
    def print_resonant_energy(self, harmonic: int) -> None:
        """
        Prints the resonant energy based on the provided K-value, harmonic number, and electron beam energy.

        Args:
            harmonic (int): The harmonic number.
        """
        if self.CLASS_NAME.startswith(('B', 'W')):
            raise ValueError("invalid operation for this synchrotron radiation source")
        else:
            energy = self.get_resonant_energy(harmonic)
            print(f">> resonant energy {energy:.2f} eV")

    def set_K_from_magnetic_field(self, B_horizontal: float=None, B_vertical: float=None, verbose: bool=False, **kwargs) -> None:
        """
        Sets the K-value based on the magnetic field strength.

        Args:
            B_horizontal (float): Magnetic field strength in the horizontal direction.
            B_vertical (float): Magnetic field strength in the vertical direction.
            verbose (bool, optional): If True, prints additional information. Default is False.
        **kwargs:
            - harmonic (int, optional): The harmonic number to use in the calculation.
        """
        harmonic = kwargs.get('harmonic', self.harmonic)

        self.MagneticStructure.B_horizontal = 0
        self.MagneticStructure.K_horizontal = 0
        self.MagneticStructure.B_vertical = 0
        self.MagneticStructure.K_vertical = 0

        if B_horizontal is not None:
            self.MagneticStructure.B_horizontal = B_horizontal
            self.MagneticStructure.K_horizontal = CHARGE * B_horizontal * self.period_length / (2 * PI * MASS * LIGHT)
        if B_vertical is not None:
            self.MagneticStructure.B_vertical = B_vertical
            self.MagneticStructure.K_vertical = CHARGE * B_vertical * self.period_length / (2 * PI * MASS * LIGHT)

        self.wavelength = energy_wavelength(self.get_resonant_energy(harmonic), 'eV')

        if verbose:
            print(f"undulator resonant energy set to {self.get_resonant_energy(harmonic):.3f} eV (harm. n°: {harmonic}) with:")
            print(f"\t>> Kh: {self.K_horizontal:.6f}")
            print(f"\t>> Kv: {self.K_vertical:.6f}")

    def set_magnetic_field_from_K(self, K_horizontal: float=None, K_vertical: float=None, verbose: bool=False, **kwargs) -> None:
        """
        Sets the magnetic field strength based on the K-value.

         Args:
            K_horizontal (float): Horizonral deflection parameter.
            K_vertical (float): Vertical deflection parameter.
            verbose (bool, optional): If True, prints additional information. Default is False.
        **kwargs:
            - harmonic (int, optional): The harmonic number to use in the calculation.
        """

        harmonic = kwargs.get('harmonic', self.harmonic)

        self.MagneticStructure.B_horizontal = 0
        self.MagneticStructure.K_horizontal = 0
        self.MagneticStructure.B_vertical = 0
        self.MagneticStructure.K_vertical = 0
        if K_horizontal is not None:
            self.MagneticStructure.K_horizontal = K_horizontal
            self.MagneticStructure.B_horizontal = K_horizontal * (2 * PI * MASS * LIGHT) / (self.period_length * CHARGE)
        if K_vertical is not None:
            self.MagneticStructure.K_vertical = K_vertical
            self.MagneticStructure.B_vertical = K_vertical * (2 * PI * MASS * LIGHT) / (self.period_length * CHARGE)

        self.wavelength = energy_wavelength(self.get_resonant_energy(harmonic), 'eV')

        if verbose:
            print(f"undulator resonant energy set to {self.get_resonant_energy(harmonic):.3f} eV (harm. n°: {harmonic}) with:")
            print(f"\t>> Bh: {self.B_horizontal:.6f}")
            print(f"\t>> Bv: {self.B_vertical:.6f}")

    def set_waist(self, **kwargs) -> None:
        """
        Sets the waist position of the photon beam.

        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            center_undulator (float): Center position of the undulator. Distance straight section/undulator 
               used as origin (positive sence: downstream)
            center_straight_section (float): Center position of the straight section. 
               Distance undulator/straight section used as origin (positive sence: downstream)
        """

        verbose = kwargs.get('verbose', False)
        cund = kwargs.get('center_undulator', 0)
        css = kwargs.get('center_straight_section', 0)

        if cund == 0:
            Zy = np.sqrt(self.e_yp)/(self.e_yp**2 + self.sigma_up**2)*css
            Zx = self.e_xp**2/(self.e_xp**2 + self.sigma_up**2)*css
        else:
            Zy = self.sigma_up**2/(self.e_yp**2 + self.sigma_up**2)*cund
            Zx = self.sigma_up**2/(self.e_xp**2 + self.sigma_up**2)*cund

        self.waist_x = Zx
        self.waist_y = Zy

        if verbose :        
            print("photon beam waist positon:")
            print(f"\t>> hor. x ver. waist position = {Zx:0.3f} m vs. {Zy:0.3f} m")

    def print_central_cone(self, verbose: bool=False, **kwargs) -> float:
        """
        Prints the angular divergence estimates of the undulator central cone using various definitions.

        Args:
            verbose (bool): If True, prints detailed angular divergence estimates.
        """
        L = self.period_length*self.number_of_periods

        divergence_krinsky = np.sqrt(self.wavelength/L)
        divergence_kim = np.sqrt(self.wavelength/2/L)
        divergence_elleaume = 0.69*np.sqrt(self.wavelength/L)

        if verbose:
            print("central cone size:")
            print(f"\t>> {divergence_krinsky*1E6:.2f} µrad (Krinsky's def.)")
            print(f"\t>> {divergence_kim*1E6:.2f} µrad (Kim's def.)")
            print(f"\t>> {divergence_elleaume*1E6:.2f} µrad (Elleaume's approx.)")

    def set_radiation_rings(self, verbose: bool=False) -> float:
        """ 
        Calculate the angular position of the first radiation ring emitted by a planar
        undulator at a given resonant energy and harmonic based on Eq. 38 from K. J. Kim, 
        "Optical and power characteristics of synchrotron radiation sources" 
        [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

        :param verbose: Whether to print results. Defaults to False.
        
        :return: Position of the first radiation ring in radians.
        """
        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)
        gamma = self.gamma()
        l = 1
        theta_nl = 1/gamma * np.sqrt(l/self.harmonic * (1+K**2 /2))
        if verbose:
            print("first radiation ring:")
            print(f"\t>> {theta_nl*1E6:.2f} µrad")

        self.first_ring = theta_nl
    
    def set_filament_emittance(self, **kwargs) -> None:
        """
        Sets the zero-emittance source size and divergence.

        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            wavelength (float): The wavelength of the undulator radiation. Default is the instance wavelength.
            mth (int): Method for filament emittance calculation. 
                mth = 0: based on Elleaume's formulation - doi:10.4324/9780203218235 (Chapter 2.5 and 2.6)
                mth = 1: based on Kim's laser mode approximation - doi:10.1016/0168-9002(86)90048-3

            L (float): Length of the undulator. Defaults to period length times number of periods.
        """

        verbose = kwargs.get('verbose', False)
        mth = kwargs.get('mth', 0)
        L = self.period_length*self.number_of_periods

        # Elleaume - doi:10.4324/9780203218235 (Chapter 2.5 and 2.6)
        if mth == 0:
            self.sigma_u =  2.74*np.sqrt(self.wavelength*L)/(4*PI)
            self.sigma_up = 0.69*np.sqrt(self.wavelength/L)

        # Kim (laser mode approximation) - doi:10.1016/0168-9002(86)90048-3
        elif mth == 1:
            self.sigma_u = np.sqrt(2*self.wavelength*L)/(4*PI)
            self.sigma_up = np.sqrt(self.wavelength/2/L)
        else:
            raise ValueError("Not a valid method for emittance calculation.")
        
        if verbose :        
            print("filament photon beam:")
            print(f"\t>> u/up = {self.sigma_u*1e6:0.2f} µm vs. {self.sigma_up*1e6:0.2f} µrad")
        
    def set_emittance(self, **kwargs) -> None:
        """
        Sets the emittance of the undulator photon beam.

        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            mth (int): Method for emittance calculation. Default is 0.
                mth = 0: Gaussian convolution - doi:10.1016/0168-9002(86)90048-3
                mth = 1: Energy spread effects for at resonant emission - doi:10.1107/S0909049509009479  
            center_undulator (float): Center position of the undulator. Default is 0.
            center_straight_section (float): Center position of the straight section. Default is 0.
        """

        verbose = kwargs.get('verbose', False)
        mth = kwargs.get('mtd', 0)
        cund = kwargs.get('center_undulator', 0)
        css = kwargs.get('center_straight_section', 0)

        if self.sigma_u is None or self.sigma_up is None:
            raise ValueError("Undulator fillament beam emittance needs to be calculated first.")

        # Gaussian Convolution - doi:10.1016/0168-9002(86)90048-3
        if mth == 0:
            sigma_x = np.sqrt(self.sigma_u**2 + self.e_x**2)
            sigma_y = np.sqrt(self.sigma_u**2 + self.e_y**2)
            sigma_xp = np.sqrt(self.sigma_up**2 + self.e_xp**2)
            sigma_yp = np.sqrt(self.sigma_up**2 + self.e_yp**2)

        # Tanaka & Kitamura - doi:10.1107/S0909049509009479
        elif mth == 1:

            def _qa(es:float) -> float:
                if es <= 0:
                    es=1e-10
                numerator   = 2*es**2
                denominator = -1 + np.exp(-2*es**2)+np.sqrt(2*PI)*es*erf(np.sqrt(2)*es)
                return np.sqrt(numerator/denominator)
            
            def _qs(es:float) -> float:
                qs = 2*(_qa(es/4))**(2/3)
                if qs<2: 
                    qs=2
                return qs
            
            sigma_x = np.sqrt(self.sigma_u**2*_qs(self.energy_spread) + self.e_x**2)
            sigma_y = np.sqrt(self.sigma_u**2*_qs(self.energy_spread) + self.e_y**2)
            sigma_xp = np.sqrt(self.sigma_up**2*_qa(self.energy_spread) + self.e_xp**2)
            sigma_yp = np.sqrt(self.sigma_up**2*_qa(self.energy_spread) + self.e_yp**2)

        else:
            raise ValueError("Not a valid method for emittance calculation.")
        
        sqv2 = 1. / ( 1./self.e_yp**2 + 1./self.sigma_up**2)
        sqh2 = 1. / ( 1./self.e_xp**2 + 1./self.sigma_up**2)

        if cund == 0:
            self.sigma_x = np.sqrt(sigma_x**2 + css**2 * sqv2)
            self.sigma_y = np.sqrt(sigma_y**2 + css**2 * sqh2)
        else:
            self.sigma_x = np.sqrt(sigma_x**2 + cund**2 * sqv2)
            self.sigma_y = np.sqrt(sigma_y**2 + cund**2 * sqh2)

        # self.sigma_x = sigma_x
        # self.sigma_y = sigma_y
        self.sigma_xp = sigma_xp
        self.sigma_yp = sigma_yp

        if verbose :        
            print("convolved photon beam:")
            print(f"\t>> x/xp = {sigma_x*1e6:0.2f} µm vs. {sigma_xp*1e6:0.2f} µrad")
            print(f"\t>> y/yp = {sigma_y*1e6:0.2f} µm vs. {sigma_yp*1e6:0.2f} µrad")

    def set_on_axis_flux(self, verbose: bool=False) -> float:
        """ 
        Calculate the peak flux density [ph/s/mrad²/0.1%bw] at a given resonant energy and 
        harmonic based on Eq. 39 (and 40) from K. J. Kim, "Optical and power characteristics 
        of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],  Opt. Eng 34(2), 342 (1995). 

        :param verbose: Whether to print results. Defaults to False.
        
        :return: Peak flux density in [ph/s/mrad²/0.1%bw].
        """
        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)

        if self.K_vertical == 0 or self.K_horizontal == 0:
            und_type = 'planar'
        else:
            und_type = 'helical'

        gamma = self.gamma()
        dw_w = 0.1/100
        N2 = self.number_of_periods**2
        n = self.harmonic
        I = self.current
        self.on_axis_flux = (ALPHA*N2*(gamma**2)*dw_w*I/CHARGE)*Fn(K, n)*1E-6
        
        if verbose:
            print("on axis flux:")
            print(f"\t>> {self.on_axis_flux:.3e} ph/s/mrad²/0.1%bw")

    def get_central_cone_flux(self, mtd=0, verbose: bool=False, **kwargs):
        """ 
        Calculate the total flux integrated over the central cone in [ph/s/0.1%bw] at a given 
        resonant energy and harmonic. If mtd is 0, this is based on Eq. 41 and 42 from K. J. Kim,
        "Optical and  power characteristics of synchrotron radiation sources" 
        [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). If mtd is 1, calculation
        is based on the integration of the convolution between undulator natural divergence and 
        electron ebam divergence as in Elleaume - doi:10.4324/9780203218235 (Chapter 2.5, Eq. 24)

        :param verbose: Whether to print results. Defaults to False.
        
        :return: float: Total photon flux in the central cone [ph/s/0.1%bw].
        """

        if mtd == 0:
            K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)
            dw_w = 0.1/100
            N = self.number_of_periods
            n = self.harmonic
            I = self.current
            Qn = (1+K**2/2)*Fn(K, n)/n
            self.central_cone_flux = PI*ALPHA*N*dw_w*I/CHARGE*Qn

        else:
            nsigma = kwargs.get("nsigma", 6)
            dtn = kwargs.get("off_res_wavelength", self.wavelength)

            divergence = np.amax([self.sigma_xp, self.sigma_yp])
            # natural undulator divergence
            theta = np.linspace(0, nsigma/2, 1001)*divergence
            natural_divergence_1d = np.sinc(0.5*self.harmonic*(theta/divergence)**2 
                                            + self.number_of_periods*(self.wavelength/dtn - self.harmonic))**2
            nat_div_interp_func = interp1d(theta, natural_divergence_1d, 
                                           bounds_error=False, fill_value=0)
            x = np.linspace(-nsigma*divergence/2, nsigma*divergence/2, 2001)
            y = np.linspace(-nsigma*divergence/2, nsigma*divergence/2, 2001)
            xx, yy = np.meshgrid(x, y)
            natural_divergence = nat_div_interp_func(np.sqrt(xx**2 + yy**2))
            # electron beam divergence
            ebam_divergence = np.exp(-((xx**2) / (2 * self.e_xp**2) + 
                                       (yy**2) / (2 * self.e_yp**2)))
            # electron beam divergence
            photon_beam_divergence = fftconvolve(natural_divergence, ebam_divergence/ebam_divergence.sum(), mode='same')
            self.central_cone_flux = np.sum(photon_beam_divergence*self.on_axis_flux)*(x[1]-x[0])*(y[1]-y[0])*1E6

        if verbose:
            print("flux within the central cone:")
            print(f"\t>> {self.central_cone_flux:.3e} ph/s/0.1%bw")

    def get_brilliance(self, verbose: bool=False):
        """
        Calculates and sets the brilliance (spectral brightness) of the undulator source in 
        [ph/s/mrad²/mm²/0.1%bw].

        If the central cone flux is not yet computed, it is calculated using the current configuration.

        Args:
            verbose (bool): If True, prints the calculated brilliance value.
        """
        if self.central_cone_flux is None:
            self.get_central_cone_flux(verbose=verbose)
        self.brilliance = self.central_cone_flux/(self.sigma_x*self.sigma_xp*self.sigma_y*self.sigma_yp*1E12)

        if verbose:
            print("brilliance:")
            print(f"\t>> {self.brilliance:.3e} ph/s/mrad²/mm²/0.1%bw")

    def get_coherent_flux(self, verbose: bool=False):
        """
        Calculates and sets the coherent flux in [ph/s/0.1%bw].

        If the coherent fraction or central cone flux are not already available, they are computed.

        Args:
            verbose (bool): If True, prints the coherent flux.
        """
        if self.coherent_fraction is None:
            self.get_coherent_fraction(verbose=verbose)

        if self.central_cone_flux is None:
            self.get_central_cone_flux(verbose=verbose)

        self.coherent_flux = self.coherent_fraction * self.central_cone_flux / 100

        if verbose:
            print("coherent flux:")
            print(f"\t>> {self.coherent_flux:.3e} ph/s/0.1%bw")

    def get_coherent_fraction(self, verbose: bool=False):
        """
        Estimates and sets the coherent fraction [%] of the undulator beam.

        This is defined as the product of the ratios of filament beam phase space to total 
        beam phase space in both transverse directions.

        Args:
            verbose (bool): If True, prints the coherent fraction estimation.
        """
        CF_x = self.sigma_u * self.sigma_up / (self.sigma_x * self.sigma_xp)
        CF_y = self.sigma_u * self.sigma_up / (self.sigma_y * self.sigma_yp)

        self.coherent_fraction = CF_x*CF_y*100

        if verbose:
            print("coherent fraction (estimation):")
            print(f"\t>> {self.coherent_fraction:.1f} %")   

    def set_total_power(self, verbose: bool=False) -> float:
        """ 
        Calculate the total power emitted by a planar undulator in watts (W) based on Eq. 56 
        from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
        [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

        :param verbose: Whether to print results. Defaults to False.
        
        :return: Total power emitted by the undulator in watts (W).
        """
        gamma = self.gamma()

        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)
        
        tot_pow = (self.number_of_periods*Z0*self.current*CHARGE*2*PI*LIGHT*gamma**2*K**2)/(6*self.period_length)
        if verbose:
            print("total integrated power:")
            print(f"\t>> {tot_pow:.3e} W")

        self.total_power = tot_pow

    def get_power_through_slit(self, verbose: bool=False, **kwargs) -> float:
        """ 
        Calculate the power emitted by a planar undulator passing through a slit in watts (W), 
        based on Eq. 50  from K. J. Kim, "Optical and power characteristics of synchrotron 
        radiation sources" [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995).

        The integration is performed only over one quadrant (positive phi and psi) for 
        computational efficiency, and the result is scaled by 4, leveraging symmetry 
        about zero in both directions.

        :param hor_slit (float): Horizontal slit size [rad].
        :param ver_slit (float): Vertical slit size [rad].
        :param verbose: Whether to print results. Defaults to False.

        :return: Power passing through the slit in watts (W).
        """

        npix = kwargs.get("npix", 251)
        nsigma = kwargs.get("nsigma", 6)
        hor_slit= kwargs.get("hor_slit", nsigma*self.sigma_xp)
        ver_slit= kwargs.get("ver_slit", nsigma*self.sigma_yp)

        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)

        if self.K_vertical == 0 or self.K_horizontal == 0:
            und_type = 'planar'
        else:
            und_type = 'helical'

        gamma = self.gamma()

        # positive quadrant
        dphi = np.linspace(0, hor_slit/2, npix)
        dpsi = np.linspace(0, ver_slit/2, npix)

        gk = G_K(K)

        self.set_total_power(verbose=False)
        d2P_d2phi_0 = self.total_power*(gk*21*gamma**2)/(16*PI*K)

        quadrant = compute_f_K_numba(dphi, dpsi, K, gamma, gk)

        full_quadrant = np.block([
            [np.flip(np.flip(quadrant[1:, 1:], axis=0), axis=1),   # Top-left quadrant
             np.flip(quadrant[1:, :], axis=0)],                    # Top-right quadrant
            [np.flip(quadrant[:, 1:], axis=1),                     # Bottom-left quadrant
             quadrant]                                             # Bottom-right quadrant
        ])

        dphi = np.linspace(-hor_slit / 2, hor_slit / 2, full_quadrant.shape[1])
        dpsi = np.linspace(-ver_slit / 2, ver_slit / 2, full_quadrant.shape[0])

        dphi_step = dphi[1] - dphi[0]
        dpsi_step = dpsi[1] - dpsi[0]

        d2P_d2phi = d2P_d2phi_0*full_quadrant

        CumPow = d2P_d2phi.sum()*dphi_step*dpsi_step

        if verbose:
            print(f"power emitted through a {hor_slit*1E3:.3f} x {ver_slit*1E3:.3f} mrad² slit:")
            print(f"\t>> {CumPow:.3f} W")

        self.power_though_slit = {'power': CumPow, 'slit': {'dh':hor_slit, 'dv':ver_slit}}

    
#***********************************************************************************
# **planar undulator** auxiliary functions 
# K. J. Kim, "Optical and power characteristics of synchrotron radiation sources" 
# [also Erratum 34(4)1243(Apr1995)],  Opt. Eng 34(2), 342 (1995)
#***********************************************************************************

def G_K(K):
    """ Angular function f_k, based on Eq. 52 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    numerator = K*(K**6 + 24/7 * K**4 + 4*K**2 + 16/7)
    denominator = (1+K**2)**(7/2)

    return numerator/denominator

@njit(parallel=True)
def compute_f_K_numba(dphi, dpsi, K, gamma, gk):
    npix = len(dphi)
    angular_function_f_K = np.zeros((npix, npix))
    for j in prange(npix):
        for i in range(npix):
            angular_function_f_K[i, j] = f_K_numba(dphi[j], dpsi[i], K, gamma)
    return angular_function_f_K*16*K/(7*np.pi*gk)

@njit
def f_K_numba(phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    n_points = int(2*np.pi*1001)  # Number of integration points
    return integrate_trapezoidal_numba(f_K_integrand_numba, -np.pi, np.pi, n_points, phi, psi, K, gamma)

@njit
def f_K_integrand_numba(csi, phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    X = gamma * psi
    Y = K * np.cos(csi) - gamma * phi
    D = 1 + X**2 + Y**2
    return (np.sin(csi)**2) * ((1 + X**2 - Y**2)**2 + 4 * (X * Y)**2) / (D**5)

@njit
def integrate_trapezoidal_numba(func, a, b, n, *args):
    """Custom trapezoidal integration."""
    h = (b - a) / n
    s = 0.5 * (func(a, *args) + func(b, *args))
    for i in range(1, n):
        s += func(a + i * h, *args)
    return s * h

def Fn(K, n):
    """
    Function F_n(K), based on Eq. 40 from K. J. Kim, "Optical and power characteristics
    of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995).
    """
    arg_Bessel = 0.25*(n*K**2)/(1 + K**2/2)
    Jnp1 = jv((n+1)/2, arg_Bessel)
    Jnm1 = jv((n-1)/2, arg_Bessel)
    arg_mult = ((K*n)/(1 + K**2/2))**2
    return arg_mult*(Jnp1 - Jnm1)**2

#***********************************************************************************
# **elliptical** auxiliary functions 
# K. J. Kim, "Optical and power characteristics of synchrotron radiation sources" 
# [also Erratum 34(4)1243(Apr1995)],  Opt. Eng 34(2), 342 (1995)
#***********************************************************************************

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
    

#***********************************************************************************
# **other** auxiliary functions 
#***********************************************************************************

def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fit_gaussian(x, fx):
    p0 = [np.max(fx), x[np.argmax(fx)], np.std(x)]
    popt, _ = opt.curve_fit(gaussian, x, fx, p0=p0)
    return popt


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
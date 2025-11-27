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
        return (3.0 * PLANCK * B * gamma**2) / (4.0 * PI * MASS)
    
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
    

if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
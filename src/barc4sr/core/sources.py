# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Synchrotron light sources built from ElectronBeam and MagneticStructure.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.constants import physical_constants
from scipy.special import erf

from barc4sr.syned.mapping import write_syned_file

from .electron_beam import ElectronBeam
from .energy import energy_wavelength
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
        self.dS = 0

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

    def configure(
        self,
        *,
        si: float = -1e23,
        sf: float = 1e23,
        dS: float = None,
        reset: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Configure an arbitrary magnetic source.

        Trim and optionally shift the arbitrary magnetic field.

        Parameters
        ----------
        - si, sf (float): lower and upper trimming bounds [m].
        - dS (float): absolute longitudinal shift [m] applied to the magnetic-field
          grid so that the chosen emission reference is placed at ``s = 0`` for the
          radiation calculation. If ``None``, the current source shift is preserved.
        - reset (bool): if True, restore the original untrimmed field before
          applying ``dS`` and trimming.
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

        print(dS)

        if dS is not None:
            self.recenter(dS=dS, verbose=False)

        self.trim(si=si, sf=sf, verbose=False)

        if verbose:
            self._verbose()

    def recenter(self, *, dS: float = None, verbose: bool = False) -> None:
        """
        Shift the arbitrary magnetic field grid.

        The shift is always applied from the cached original field, not from the
        current shifted field. Therefore repeated calls are not cumulative.

        Parameters
        ----------
        - dS (float): absolute longitudinal shift [m] applied to the original
        magnetic-field grid. This defines the position in the original field
        that becomes s = 0 in the shifted field. If ``None``, use the midpoint
        of the original ``s`` interval.
        - verbose (bool): if True, prints to the prompt
        """
        self._ensure_original_field()

        mf = self._original_magnetic_field
        if mf is None:
            raise ValueError(
                "Original magnetic field dictionary is not available."
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
        dS_val = mid if dS is None else float(dS)

        mf_shift = {}
        for k, v in mf.items():
            if k == "s":
                mf_shift["s"] = s - dS_val
            else:
                mf_shift[k] = deepcopy(v)

        self.MagneticStructure.magnetic_field = mf_shift
        self.dS = dS_val

        if verbose:
            self._verbose()

    def trim(self, *, si: float = -1e23, sf: float = 1e23, verbose: bool = False) -> None:
        """
        Trim the arbitrary magnetic field.

        The trimming is performed on the current magnetic field stored in
        ``MagneticStructure.magnetic_field``. If the field has been shifted
        (via ``recenter`` or ``configure``), the trimming bounds ``si`` and ``sf``
        are interpreted in that shifted coordinate system.

        Parameters
        ----------
        - si, sf (float): lower and upper trimming bounds [m] in the current
          coordinate system.
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

        mf_trim = {}
        for k, v in mf.items():
            if k == "s":
                mf_trim["s"] = s[mask]
                continue
            arr = np.asarray(v)
            if arr.shape[:1] == (s.size,):
                mf_trim[k] = arr[mask, ...]
            else:
                mf_trim[k] = deepcopy(v)

        self.MagneticStructure.magnetic_field = mf_trim

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
        self.dS = 0.0

    def _ensure_original_field(self) -> None:
        """
        Internal helper to cache the original magnetic field.

        Makes a deepcopy of ``MagneticStructure.magnetic_field`` on first use.
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
        print(f"\t>> Field shifted by dS = {self.dS:.6f} m")
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
    SR bending magnet source.
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

        self.extraction_angle = None

    def configure(
        self,
        *,
        dS: float = None,
        extraction_angle: float = None,
        verbose: bool = False,
    ) -> None:
        """
        Configure bending magnet source.

        From B and the electron beam energy, the radius and critical energy can be
        computed on the fly and optionally printed.

        The extraction geometry is defined by either the longitudinal source shift
        ``dS`` [m] or the extraction angle [rad] taken from the preceding straight
        section.

        Parameters
        ----------
        - dS (float): longitudinal shift [m] used to place the chosen emission point
          at ``s = 0`` for the radiation calculation. ``dS = 0`` corresponds to
          emission from the middle of the magnet.
        - extraction_angle (float): extraction angle [rad] taken from the previous
          straight section. ``0`` [rad] is the magnet entrance and ``L / R`` [rad]
          is the magnet exit.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If B is not set, or if both ``dS`` and ``extraction_angle`` are provided
            at the same time.
        """
        _ = self.B

        if verbose:
            print('\n>>>>>>>>>>> bending magnet <<<<<<<<<<<\n')
            print("> Properties:")
            print(f"\t>> B = {self.B:.6f} T")
            print(f"\t>> R = {self.radius:.6f} m")
            print(f"\t>> critical energy = {self.critical_energy:.3f} eV")

        if dS is not None and extraction_angle is not None:
            raise ValueError(
                "Provide only one of 'dS' or 'extraction_angle', not both."
            )

        if dS is None and extraction_angle is None:
            if self.extraction_angle is not None:
                extraction_angle = self.extraction_angle
            else:
                dS = self.dS

        self._set_geometry(dS=dS, extraction_angle=extraction_angle, verbose=verbose)

    def _set_geometry(
        self,
        *,
        dS: float = None,
        extraction_angle: float = None,
        verbose: bool = False,
    ) -> None:
        """
        Internal helper to set extraction geometry.

        Parameters
        ----------
        - dS (float): longitudinal shift [m] used to place the chosen emission point
          at ``s = 0`` for the radiation calculation.
        - extraction_angle (float): extraction angle [rad] taken from the previous
          straight section. ``0`` [rad] is the magnet entrance and ``L / R`` [rad]
          is the magnet exit.
        - verbose (bool): if True, prints to the prompt

        Raises
        ------
        ValueError
            If both ``dS`` and ``extraction_angle`` are None, or if the resulting
            angle is out of range.
        """
        if dS is None and extraction_angle is None:
            raise ValueError("Provide 'dS' (m) or 'extraction_angle' (rad).")

        L = self.MagneticStructure.field_length
        if L is None:
            raise ValueError("MagneticStructure.field_length must be defined.")

        R = self.radius

        half_arc = 0.5 * L / R
        full_arc = L / R

        if extraction_angle is None:
            dS_val = float(dS)
            extraction_angle = half_arc - dS_val / R
        else:
            extraction_angle = float(extraction_angle)
            dS_val = (half_arc - extraction_angle) * R

        if not (0.0 <= extraction_angle <= full_arc):
            raise ValueError(
                f"extraction_angle={extraction_angle:.6g} rad out of range [0, {full_arc:.6g}]"
            )

        distance_from_entrance = 0.5 * L - dS_val

        self.dS = dS_val
        self.extraction_angle = extraction_angle

        if verbose:
            print("> Extraction geometry:")
            print(f"\t>> dS                     : {dS_val:.6f} m")
            print(f"\t>> dist. from BM entrance : {distance_from_entrance:.6f} m")
            print(f"\t>> extraction angle       : {extraction_angle * 1e3:.3f} mrad")
            print(f"\t>> BM arc                 : {2.0 * half_arc * 1e3:.3f} mrad "
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
    SR undulator source.

    The magnetic-structure container stores only immutable structural parameters.
    The undulator operating point is therefore stored in this source instance via
    the peak magnetic fields and their phase offsets.
    """

    _NAMED_POLARIZATIONS = {"LH", "LV", "L45", "L135", "CR", "CL"}
    _ALL_POLARIZATIONS = _NAMED_POLARIZATIONS | {"CUSTOM"}

    def __init__(self, **kwargs) -> None:
        """
        Initialize an undulator source.

        Parameters
        ----------
        - electron_beam (ElectronBeam)
        - magnetic_structure (MagneticStructure)
        """
        super().__init__(**kwargs)

        if getattr(self.MagneticStructure, "magnet_type", None) != "undulator":
            raise ValueError(
                "UndulatorSource requires a MagneticStructure with "
                "magnet_type='undulator'."
            )

        self.B_horizontal = 0.0
        self.B_vertical = 0.0
        self.phase_horizontal = 0.0
        self.phase_vertical = 0.0
        self.harmonic = 1
        self.polarization = "LH"
        self._characteristics: dict | None = None

    def configure(
        self,
        *,
        energy: float | None = None,
        wavelength: float | None = None,
        polarization: str | None = None,
        harmonic: int | None = 1,
        dS: float | None = None,
        B: float | None = None,
        K: float | None = None,
        B_horizontal: float | None = None,
        B_vertical: float | None = None,
        K_horizontal: float | None = None,
        K_vertical: float | None = None,
        phase_horizontal: float | None = None,
        phase_vertical: float | None = None,
        even_harmonics: bool = False,
        K_min: float = 0.05,
        max_harmonic: int = 15,
        verbose: bool = False,
    ) -> None:
        """
        Configure the undulator operating point.

        Three mutually exclusive driving modes are supported:

        1. Resonance-driven mode
           ``energy`` or ``wavelength`` + named ``polarization``.
        2. Constrained field-driven mode
           scalar ``B`` or scalar ``K`` + named ``polarization``.
        3. Expert custom field-driven mode
           plane-specific ``B_*`` or ``K_*`` and optional phases with
           ``polarization='custom'``.

        Parameters
        ----------
        energy : float | None, optional
            Target resonant photon energy [eV]. Mutually exclusive with
            ``wavelength``.
        wavelength : float | None, optional
            Target resonant wavelength [m]. Mutually exclusive with ``energy``.
        polarization : {"LH", "LV", "L45", "L135", "CR", "CL", "custom"} | None, optional
            Polarization flag. Defaults to ``"LH"`` for resonance-driven and
            constrained scalar field-driven modes. Plane-specific field input
            defaults to ``"custom"``.
        harmonic : int | None, optional
            Reference harmonic used in resonance calculations. If ``None`` in
            resonance-driven mode, scan candidate harmonics until a valid one is
            found. Default is 1.
        dS : float | None, optional
            Longitudinal source shift [m]. If omitted, keep the current value.
        B : float | None, optional
            Scalar peak magnetic field amplitude [T] for named polarizations.
        K : float | None, optional
            Scalar deflection parameter for named polarizations.
        B_horizontal, B_vertical : float | None, optional
            Plane-specific peak magnetic fields [T]. Only valid with
            ``polarization='custom'``.
        K_horizontal, K_vertical : float | None, optional
            Plane-specific deflection parameters. Only valid with
            ``polarization='custom'``.
        phase_horizontal, phase_vertical : float | None, optional
            Plane-specific phase offsets [rad]. For named polarizations they are
            imposed by the polarization state. For ``custom`` they are stored as
            provided; if only one field plane is active, missing phases default to
            zero.
        even_harmonics : bool, optional
            If True, even harmonics are allowed during the automatic harmonic scan
            used when ``harmonic is None`` in resonance-driven mode. Default is False.
        K_min : float, optional
            Minimum total deflection parameter accepted during the automatic
            harmonic scan. Default is 0.05.
        max_harmonic : int, optional
            Maximum harmonic tested during the automatic scan. Default is 15.
        verbose : bool, optional
            If True, prints a summary of the configured operating point.

        Raises
        ------
        ValueError
            If the configuration is ambiguous, incomplete, or inconsistent with
            the selected driving mode.
        """
        pol_input = self._normalize_polarization(polarization)

        has_energy = energy is not None
        has_wavelength = wavelength is not None
        has_scalar_B = B is not None
        has_scalar_K = K is not None
        has_plane_B = B_horizontal is not None or B_vertical is not None
        has_plane_K = K_horizontal is not None or K_vertical is not None

        n_modes = sum((
            has_energy or has_wavelength,
            has_scalar_B or has_scalar_K,
            has_plane_B or has_plane_K,
        ))
        if n_modes != 1:
            raise ValueError(
                "Provide exactly one driving mode: resonance-driven "
                "(energy/wavelength), scalar field-driven (B/K), or plane-specific "
                "field-driven (B_horizontal/B_vertical or K_horizontal/K_vertical)."
            )

        if has_energy and has_wavelength:
            raise ValueError("Provide only one of 'energy' or 'wavelength'.")

        if has_scalar_B and has_scalar_K:
            raise ValueError("Provide only one of scalar 'B' or scalar 'K'.")

        if has_plane_B and has_plane_K:
            raise ValueError(
                "Provide plane-specific magnetic fields or plane-specific K values, not both."
            )

        if dS is not None:
            self.dS = float(dS)

        resolved_harmonic: int

        if has_energy or has_wavelength:
            polarization_val = "LH" if pol_input is None else pol_input
            if polarization_val == "CUSTOM":
                raise ValueError(
                    "polarization='custom' is not valid in resonance-driven mode."
                )
            target_wavelength = (
                energy_wavelength(float(energy), "eV")
                if has_energy
                else float(wavelength)
            )
            if harmonic is None:
                resolved_harmonic, Bh, Bv, ph, pv = self._scan_resonant_harmonic(
                    wavelength=target_wavelength,
                    polarization=polarization_val,
                    even_harmonics=even_harmonics,
                    K_min=float(K_min),
                    max_harmonic=int(max_harmonic),
                )
            else:
                resolved_harmonic = self._validate_harmonic(harmonic)
                Bh, Bv, ph, pv = self._fields_from_resonance(
                    wavelength=target_wavelength,
                    harmonic=resolved_harmonic,
                    polarization=polarization_val,
                )

        elif has_scalar_B or has_scalar_K:
            resolved_harmonic = 1 if harmonic is None else self._validate_harmonic(harmonic)
            polarization_val = "LH" if pol_input is None else pol_input
            if polarization_val == "CUSTOM":
                raise ValueError(
                    "Scalar B/K input requires a named polarization, not 'custom'."
                )
            if has_scalar_B:
                Bh, Bv, ph, pv = self._fields_from_scalar_B(
                    B=float(B),
                    polarization=polarization_val,
                )
            else:
                Bh, Bv, ph, pv = self._fields_from_scalar_K(
                    K=float(K),
                    polarization=polarization_val,
                )

        else:
            resolved_harmonic = 1 if harmonic is None else self._validate_harmonic(harmonic)
            polarization_val = "CUSTOM" if pol_input is None else pol_input
            if polarization_val != "CUSTOM":
                raise ValueError(
                    "Plane-specific B/K input is only valid with polarization='custom'."
                )
            Bh, Bv, ph, pv = self._fields_from_custom_inputs(
                B_horizontal=B_horizontal,
                B_vertical=B_vertical,
                K_horizontal=K_horizontal,
                K_vertical=K_vertical,
                phase_horizontal=phase_horizontal,
                phase_vertical=phase_vertical,
            )

        self.B_horizontal = Bh
        self.B_vertical = Bv
        self.phase_horizontal = ph
        self.phase_vertical = pv
        self.harmonic = resolved_harmonic
        self.polarization = polarization_val
        self._characteristics = None

        if verbose:
            self._verbose_configuration()

    def characteristics(
        self,
        *,
        energy_spread: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Compute analytical undulator beam characteristics.

        This first implementation returns only the beam block. The undulator
        must already be configured so that the resonant wavelength can be
        derived from the current magnetic state.

        Parameters
        ----------
        energy_spread : bool, optional
            If True, include energy-spread effects using the Tanaka/Kitamura
            formulation. If False, use Gaussian convolution. Default is False.
        verbose : bool, optional
            If True, print the computed beam characteristics.

        Returns
        -------
        dict
            Structured characteristics dictionary with ``meta`` and ``beam`` blocks.

        Raises
        ------
        ValueError
            If the undulator has not been configured.
        """
        self._ensure_configured_for_characteristics()

        meta = self._characteristics_meta()
        beam = self._characteristics_beam(energy_spread=energy_spread)

        self._characteristics = {
            "meta": meta,
            "beam": beam,
        }

        if verbose:
            self._verbose_characteristics()

        return self._characteristics

    @property
    def length(self) -> float:
        """
        Undulator magnetic length [m].
        """
        return self.period_length * self.number_of_periods

    @property
    def K_horizontal(self) -> float:
        """
        Horizontal deflection parameter.
        """
        return self._B_to_K(self.B_horizontal)

    @property
    def K_vertical(self) -> float:
        """
        Vertical deflection parameter.
        """
        return self._B_to_K(self.B_vertical)

    @property
    def K_total(self) -> float:
        """
        Total deflection parameter magnitude.
        """
        return float(np.hypot(self.K_horizontal, self.K_vertical))

    def resonant_wavelength(self, harmonic: int | None = None) -> float:
        """
        Return the resonant wavelength [m] for the selected harmonic.

        Parameters
        ----------
        harmonic : int | None, optional
            Harmonic index. If omitted, use the currently stored ``self.harmonic``.

        Returns
        -------
        float
            Resonant wavelength [m].
        """
        n = self.harmonic if harmonic is None else self._validate_harmonic(harmonic)
        gamma = self.gamma()
        K2 = self.K_horizontal**2 + self.K_vertical**2
        return self.period_length / (2.0 * n * gamma**2) * (1.0 + 0.5 * K2)

    def resonant_energy(self, harmonic: int | None = None) -> float:
        """
        Return the resonant photon energy [eV] for the selected harmonic.

        Parameters
        ----------
        harmonic : int | None, optional
            Harmonic index. If omitted, use the currently stored ``self.harmonic``.
        """
        return energy_wavelength(self.resonant_wavelength(harmonic=harmonic), "m")

    def power_through_slit(self, *args, **kwargs):
        """
        Placeholder for the analytical slit-power estimate.
        """
        raise NotImplementedError(
            "UndulatorSource.power_through_slit() has not been ported yet."
        )

    def _verbose_configuration(self) -> None:
        """
        Print the configured undulator operating point.
        """
        print("\n>>>>>>>>>>> undulator <<<<<<<<<<<\n")
        print("> Operating point:")
        print(f"\t>> polarization      : {self.polarization}")
        print(f"\t>> harmonic          : {self.harmonic}")
        print(f"\t>> dS                : {self.dS:.6f} m")
        print(f"\t>> (Bh, Bv)          : ({self.B_horizontal:.6f} T , {self.B_vertical:.6f} T)")
        print(f"\t>> (ph, pv)          : ({self.phase_horizontal:.6f} rad , {self.phase_vertical:.6f} rad)")
        print(f"\t>> (Kh, Kv)          : ({self.K_horizontal:.6f} , {self.K_vertical:.6f})")
        print(f"\t>> wavelength (res.) : {self.resonant_wavelength():.3e} m")
        print(f"\t>> energy (res.)     : {self.resonant_energy():.3f} eV")

    def _verbose_characteristics(self) -> None:
        """
        Print the last computed analytical undulator characteristics.
        """
        if self._characteristics is None:
            raise ValueError("No undulator characteristics have been computed yet.")

        beam = self._characteristics["beam"]
        electron = beam["electron"]
        filament = beam["filament"]
        ring = beam["ring"]
        waist = beam["waist"]
        photon = beam["photon"]

        print('\n>>>>>>>>>>> beam phase-space characteristics <<<<<<<<<<<')
        print('electron beam:')
        print(f"\t>> x/xp = {electron['sigma_x'] * 1e6:0.2f} um vs. {electron['sigma_xp'] * 1e6:0.2f} urad")
        print(f"\t>> y/yp = {electron['sigma_y'] * 1e6:0.2f} um vs. {electron['sigma_yp'] * 1e6:0.2f} urad")
        print('filament photon beam:')
        print(f"\t>> u/up = {filament['sigma_u'] * 1e6:0.2f} um vs. {filament['sigma_up'] * 1e6:0.2f} urad")
        print('first radiation ring:')
        print(f"\t>> {ring['first_ring'] * 1e6:.2f} urad")
        print('photon beam waist position:')
        print(f"\t>> hor. x ver. waist position = {waist['waist_x']:0.3f} m vs. {waist['waist_y']:0.3f} m")
        print('convolved photon beam:')
        print(f"\t>> x/xp = {photon['sigma_x'] * 1e6:0.2f} um vs. {photon['sigma_xp'] * 1e6:0.2f} urad")
        print(f"\t>> y/yp = {photon['sigma_y'] * 1e6:0.2f} um vs. {photon['sigma_yp'] * 1e6:0.2f} urad")

    def _ensure_configured_for_characteristics(self) -> None:
        """
        Validate that the undulator operating point is defined.
        """
        if self.harmonic is None or self.polarization is None:
            raise ValueError(
                "UndulatorSource must be configured before calling characteristics()."
            )
        if self.B_horizontal == 0.0 and self.B_vertical == 0.0:
            raise ValueError(
                "UndulatorSource must be configured before calling characteristics()."
            )

    def _characteristics_meta(self) -> dict:
        """
        Build the characteristics metadata block.
        """
        return {
            "harmonic": self.harmonic,
            "polarization": self.polarization,
            "wavelength": self.resonant_wavelength(),
            "energy": self.resonant_energy(),
            "dS": self.dS,
        }

    def _characteristics_beam(self, *, energy_spread: bool) -> dict:
        """
        Build the beam characteristics block.
        """
        return {
            "electron": self._characteristics_electron_beam(),
            "filament": self._characteristics_filament_beam(),
            "ring": self._characteristics_first_ring(),
            "waist": self._characteristics_waist(),
            "photon": self._characteristics_photon_beam(energy_spread=energy_spread),
        }

    def _characteristics_electron_beam(self) -> dict:
        """
        Build the electron-beam characteristics block.
        """
        return {
            "sigma_x": self.e_x,
            "sigma_y": self.e_y,
            "sigma_xp": self.e_xp,
            "sigma_yp": self.e_yp,
        }

    def _characteristics_filament_beam(self) -> dict:
        """
        Build the zero-emittance filament photon-beam block using Elleaume.
        """
        wavelength = self.resonant_wavelength()
        length = self.length
        sigma_u = 2.74 * np.sqrt(wavelength * length) / (4.0 * np.pi)
        sigma_up = 0.69 * np.sqrt(wavelength / length)

        return {
            "sigma_u": sigma_u,
            "sigma_up": sigma_up,
            "model": "elleaume",
        }

    def _characteristics_first_ring(self) -> dict:
        """
        Build the first-radiation-ring block.
        """
        first_ring = (1.0 / self.gamma()) * np.sqrt(
            (1.0 / self.harmonic) * (1.0 + 0.5 * self.K_total**2)
        )
        return {
            "first_ring": first_ring,
        }

    def _characteristics_waist(self) -> dict:
        """
        Build the photon-beam waist block.

        Waist transport is not reintroduced yet. Placeholder values are returned.
        """
        return {
            "waist_x": 0.0,
            "waist_y": 0.0,
            "model": "placeholder",
        }

    def _characteristics_photon_beam(self, *, energy_spread: bool) -> dict:
        """
        Build the convolved photon-beam block.
        """
        filament = self._characteristics_filament_beam()
        sigma_u = filament["sigma_u"]
        sigma_up = filament["sigma_up"]

        if energy_spread:
            sigma_x, sigma_y, sigma_xp, sigma_yp = self._photon_beam_tanaka_kitamura(
                sigma_u=sigma_u,
                sigma_up=sigma_up,
            )
            model = "tanaka_kitamura"
        else:
            sigma_x, sigma_y, sigma_xp, sigma_yp = self._photon_beam_gaussian_convolution(
                sigma_u=sigma_u,
                sigma_up=sigma_up,
            )
            model = "gaussian_convolution"

        return {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_xp": sigma_xp,
            "sigma_yp": sigma_yp,
            "model": model,
        }

    def _photon_beam_gaussian_convolution(
        self,
        *,
        sigma_u: float,
        sigma_up: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute photon-beam emittance using Gaussian convolution.
        """
        sigma_x = np.sqrt(sigma_u**2 + self.e_x**2)
        sigma_y = np.sqrt(sigma_u**2 + self.e_y**2)
        sigma_xp = np.sqrt(sigma_up**2 + self.e_xp**2)
        sigma_yp = np.sqrt(sigma_up**2 + self.e_yp**2)
        return sigma_x, sigma_y, sigma_xp, sigma_yp

    def _photon_beam_tanaka_kitamura(
        self,
        *,
        sigma_u: float,
        sigma_up: float,
    ) -> tuple[float, float, float, float]:
        """
        Compute photon-beam emittance including energy-spread effects.
        """
        def _qa(es: float) -> float:
            if es <= 0:
                es = 1e-10
            numerator = 2.0 * es**2
            denominator = (
                -1.0
                + np.exp(-2.0 * es**2)
                + np.sqrt(2.0 * np.pi) * es * erf(np.sqrt(2.0) * es)
            )
            return np.sqrt(numerator / denominator)

        def _qs(es: float) -> float:
            qs = 2.0 * (_qa(es / 4.0)) ** (2.0 / 3.0)
            return max(qs, 2.0)

        sigma_x = np.sqrt(sigma_u**2 * _qs(self.energy_spread) + self.e_x**2)
        sigma_y = np.sqrt(sigma_u**2 * _qs(self.energy_spread) + self.e_y**2)
        sigma_xp = np.sqrt(sigma_up**2 * _qa(self.energy_spread) + self.e_xp**2)
        sigma_yp = np.sqrt(sigma_up**2 * _qa(self.energy_spread) + self.e_yp**2)
        return sigma_x, sigma_y, sigma_xp, sigma_yp

    def _normalize_polarization(self, polarization: str | None) -> str | None:
        """
        Normalize the polarization flag.
        """
        if polarization is None:
            return None
        pol = polarization.strip().upper()
        if pol not in self._ALL_POLARIZATIONS:
            raise ValueError(
                f"Invalid polarization '{polarization}'. Use one of {sorted(self._ALL_POLARIZATIONS)}."
            )
        return pol

    def _validate_harmonic(self, harmonic: int) -> int:
        """
        Validate the harmonic index.
        """
        try:
            n = int(harmonic)
        except (TypeError, ValueError) as exc:
            raise ValueError("harmonic must be a positive integer.") from exc
        if n < 1:
            raise ValueError("harmonic must be >= 1.")
        return n

    def _fields_from_scalar_B(
        self,
        *,
        B: float,
        polarization: str,
    ) -> tuple[float, float, float, float]:
        """
        Build a named polarization state from a scalar magnetic-field amplitude.
        """
        if B < 0:
            raise ValueError("Scalar B must be >= 0.")
        return self._named_state_from_scalar(amplitude=float(B), polarization=polarization)

    def _fields_from_scalar_K(
        self,
        *,
        K: float,
        polarization: str,
    ) -> tuple[float, float, float, float]:
        """
        Build a named polarization state from a scalar deflection parameter.
        """
        if K < 0:
            raise ValueError("Scalar K must be >= 0.")
        return self._named_state_from_scalar(
            amplitude=self._K_to_B(float(K)),
            polarization=polarization,
        )

    def _fields_from_custom_inputs(
        self,
        *,
        B_horizontal: float | None,
        B_vertical: float | None,
        K_horizontal: float | None,
        K_vertical: float | None,
        phase_horizontal: float | None,
        phase_vertical: float | None,
    ) -> tuple[float, float, float, float]:
        """
        Build a custom two-plane state from explicit fields or K values.
        """
        if B_horizontal is not None or B_vertical is not None:
            Bh = 0.0 if B_horizontal is None else float(B_horizontal)
            Bv = 0.0 if B_vertical is None else float(B_vertical)
        else:
            Kh = 0.0 if K_horizontal is None else float(K_horizontal)
            Kv = 0.0 if K_vertical is None else float(K_vertical)
            if Kh < 0 or Kv < 0:
                raise ValueError("Plane-specific K values must be >= 0.")
            Bh = self._K_to_B(Kh)
            Bv = self._K_to_B(Kv)

        if Bh < 0 or Bv < 0:
            raise ValueError("Plane-specific B values must be >= 0.")
        if Bh == 0.0 and Bv == 0.0:
            raise ValueError("At least one field plane must be non-zero.")

        ph = 0.0 if phase_horizontal is None else float(phase_horizontal)
        pv = 0.0 if phase_vertical is None else float(phase_vertical)
        return Bh, Bv, ph, pv

    def _fields_from_resonance(
        self,
        *,
        wavelength: float,
        harmonic: int,
        polarization: str,
    ) -> tuple[float, float, float, float]:
        """
        Solve the named polarization state from a resonant wavelength.
        """
        if wavelength <= 0:
            raise ValueError("wavelength must be strictly positive.")

        gamma = self.gamma()
        arg = 2.0 * ((2.0 * harmonic * wavelength * gamma**2) / self.period_length - 1.0)
        if arg < 0:
            raise ValueError(
                "The requested wavelength/energy is not reachable for the selected harmonic."
            )
        K_total = float(np.sqrt(arg))

        if polarization in {"LH", "LV"}:
            return self._named_state_from_scalar(
                amplitude=self._K_to_B(K_total),
                polarization=polarization,
            )

        return self._named_state_from_scalar(
            amplitude=self._K_to_B(K_total / np.sqrt(2.0)),
            polarization=polarization,
        )

    def _scan_resonant_harmonic(
        self,
        *,
        wavelength: float,
        polarization: str,
        even_harmonics: bool,
        K_min: float,
        max_harmonic: int,
    ) -> tuple[int, float, float, float, float]:
        """
        Scan harmonics in resonance-driven mode until a valid operating point is found.
        """
        if K_min < 0:
            raise ValueError("K_min must be >= 0.")
        if max_harmonic < 1:
            raise ValueError("max_harmonic must be >= 1.")

        for n in range(1, max_harmonic + 1):
            if not even_harmonics and n % 2 == 0:
                continue
            try:
                Bh, Bv, ph, pv = self._fields_from_resonance(
                    wavelength=wavelength,
                    harmonic=n,
                    polarization=polarization,
                )
            except ValueError:
                continue

            K_total = np.hypot(self._B_to_K(Bh), self._B_to_K(Bv))
            if K_total < K_min:
                continue

            return n, Bh, Bv, ph, pv

        raise ValueError(
            "No valid harmonic found for the requested wavelength/energy and polarization "
            f"within 1..{max_harmonic}."
        )

    def _named_state_from_scalar(
        self,
        *,
        amplitude: float,
        polarization: str,
    ) -> tuple[float, float, float, float]:
        """
        Build one of the constrained named polarization states from a scalar amplitude.
        """
        if amplitude < 0:
            raise ValueError("Amplitude must be >= 0.")

        if polarization == "LH":
            return 0.0, amplitude, 0.0, 0.0
        if polarization == "LV":
            return amplitude, 0.0, 0.0, 0.0
        if polarization == "L45":
            return amplitude, amplitude, 0.0, 0.0
        if polarization == "L135":
            return amplitude, amplitude, 0.0, np.pi
        if polarization == "CR":
            return amplitude, amplitude, np.pi / 4.0, -np.pi / 4.0
        if polarization == "CL":
            return amplitude, amplitude, -np.pi / 4.0, np.pi / 4.0

        raise ValueError(
            f"Named scalar state requires one of {sorted(self._NAMED_POLARIZATIONS)}."
        )

    def _B_to_K(self, B: float) -> float:
        """
        Convert peak magnetic field [T] to deflection parameter.
        """
        return CHARGE * float(B) * self.period_length / (2.0 * np.pi * MASS * LIGHT)

    def _K_to_B(self, K: float) -> float:
        """
        Convert deflection parameter to peak magnetic field [T].
        """
        return float(K) * (2.0 * np.pi * MASS * LIGHT) / (self.period_length * CHARGE)
    
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
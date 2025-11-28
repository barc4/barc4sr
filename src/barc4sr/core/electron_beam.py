# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
ElectronBeam class
"""

from __future__ import annotations

import numpy as np

from .energy import get_gamma


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

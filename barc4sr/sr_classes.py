#!/bin/python

"""
This module provides the barc4sr classes:

- ElectronBeam
- MagneticStructure
- SynchrotronSource
- UndulatorSource(SynchrotronSource)
- BendingMagnetSource(SynchrotronSource)

"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '18/FEB/2025'

import numpy as np
import scipy.optimize as opt
from numba import njit, prange
from scipy.constants import physical_constants
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import erf, jv

from barc4sr.aux_energy import energy_wavelength, get_gamma
from barc4sr.aux_syned import read_syned_file, write_syned_file  #, write_spectra_file

ALPHA =  physical_constants["fine-structure constant"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
MASS = physical_constants["electron mass"][0]
PI = np.pi
PLANCK = physical_constants["Planck constant"][0]
Z0 = physical_constants["characteristic impedance of vacuum"][0]

class ElectronBeam(object):
    """
    Class for entering the electron beam parameters - this is based on the SRWLPartBeam class.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes an instance of the ElectronBeam class.

        Args:
            **kwargs: Parameters for the electron beam:
                - energy (float): Energy of the electron beam in GeV.
                - energy_spread (float): RMS energy spread of the electron beam.
                - current (float): Average current of the electron beam in Amperes.
                - number_of_bunches (int): Number of bunches in the electron beam (default: 1).
                - moment_xx (float): Second order moment: <(x-<x>)^2>.
                - moment_xxp (float): Second order moment: <(x-<x>)(x'-<x'>)>.
                - moment_xpxp (float): Second order moment: <(x'-<x'>)^2>.
                - moment_yy (float): Second order moment: <(y-<y>)^2>.
                - moment_yyp (float): Second order moment: <(y-<y>)(y'-<y'>)>.
                - moment_ypyp (float): Second order moment: <(y'-<y'>)^2>.
        """
        self.CLASS_NAME = "ElectronBeam"
        self.energy_in_GeV = kwargs.get("energy", None)
        self.energy_spread = kwargs.get("energy_spread", None)
        self.current = kwargs.get("current", None)
        self.number_of_bunches = kwargs.get("number_of_bunches", 1)

        self.moment_xx = kwargs.get("moment_xx", None)
        self.moment_xxp = kwargs.get("moment_xxp", None)
        self.moment_xpxp = kwargs.get("moment_xpxp", None)
        self.moment_yy = kwargs.get("moment_yy", None)
        self.moment_yyp = kwargs.get("moment_yyp", None)
        self.moment_ypyp = kwargs.get("moment_ypyp", None)

        moments_list = [
            self.moment_xx, self.moment_xxp, self.moment_xpxp,
            self.moment_yy, self.moment_yyp, self.moment_ypyp,
        ]
        if all(moment is not None for moment in moments_list):
            self.to_rms()

    def from_twiss(self, energy: float, energy_spread: float, current: float, 
                   beta_x: float, alpha_x: float, eta_x: float, etap_x: float, 
                   beta_y: float, alpha_y: float, eta_y: float, etap_y: float,
                   **kwargs) -> None:
        """
        Sets up electron beam internal data from Twiss parameters.

        Args:
            energy (float): Energy of the electron beam in GeV.
            energy_spread (float): RMS energy spread of the electron beam.
            current (float): Average current of the electron beam in Amperes.
            beta_x (float): Horizontal beta-function in meters.
            alpha_x (float): Horizontal alpha-function in radians.
            eta_x (float): Horizontal dispersion function in meters.
            etap_x (float): Horizontal dispersion function derivative in radians.
            beta_y (float): Vertical beta-function in meters.
            alpha_y (float): Vertical alpha-function in radians.
            eta_y (float): Vertical dispersion function in meters.
            etap_y (float): Vertical dispersion function derivative in radians.

        Keyword Args for emittance and coupling or emittance_x and emittance_y:
            emittance (float): Emittance of the electron beam.
            coupling (float): Coupling coefficient between horizontal and vertical emittances.
            emittance_x (float): Horizontal emittance in meters.
            emittance_y (float): Vertical emittance in meters.
        """

        emittance = kwargs.get('emittance')
        coupling = kwargs.get('coupling')
        emittance_x = kwargs.get('emittance_x')
        emittance_y = kwargs.get('emittance_y')

        if emittance_x is None or emittance_y is None:
            if emittance is None or coupling is None:
                raise ValueError("Either emittance and coupling, or both emittance_x and emittance_y must be provided.")
            emittance_x = emittance * (1 / (coupling + 1)) if emittance_x is None else emittance_x
            emittance_y = emittance * (coupling / (coupling + 1)) if emittance_y is None else emittance_y

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current
        sigE2 = energy_spread**2
        # <(x-<x>)^2>
        self.moment_xx = emittance_x*beta_x + sigE2*eta_x*eta_x 
        # <(x-<x>)(x'-<x'>)>          
        self.moment_xxp = -emittance_x*alpha_x + sigE2*eta_x*etap_x  
        # <(x'-<x'>)^2>      
        self.moment_xpxp = emittance_x*(1 + alpha_x*alpha_x)/beta_x + sigE2*etap_x*etap_x 
        #<(y-<y>)^2>    
        self.moment_yy = emittance_y*beta_y + sigE2*eta_y*eta_y 
        #<(y-<y>)(y'-<y'>)>          
        self.moment_yyp = -emittance_y*alpha_y + sigE2*eta_y*etap_y
        #<(y'-<y'>)^2>        
        self.moment_ypyp = emittance_y*(1 + alpha_y*alpha_y)/beta_y + sigE2*etap_y*etap_y     

        self.to_rms()

    def from_rms(self, energy: float, energy_spread: float, current: float, x: float, xp: float,
                 y: float, yp: float, xxp: float = 0, yyp: float = 0) -> None:
        """
        Sets up electron beam internal data from RMS values.

        Args:
            energy (float): Energy of the electron beam in GeV.
            energy_spread (float): RMS energy spread of the electron beam.
            current (float): Average current of the electron beam in Amperes.
            x (float): Horizontal RMS size in meters.
            xp (float): Horizontal RMS divergence in radians.
            y (float): Vertical RMS size in meters.
            yp (float): Vertical RMS divergence in radians.
            xxp (float): Cross-correlation term between x and xp in meters. Defaults to 0.
            yyp (float): Cross-correlation term between y and yp in meters. Defaults to 0.
        """

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current

        self.moment_xx = x*x        # <(x-<x>)^2>
        self.moment_xxp = xxp       # <(x-<x>)(x'-<x'>)>          
        self.moment_xpxp = xp*xp    # <(x'-<x'>)^2>      
        self.moment_yy = y*y        # <(y-<y>)^2>    
        self.moment_yyp = yyp       # <(y-<y>)(y'-<y'>)>          
        self.moment_ypyp = yp*yp    # <(y'-<y'>)^2>        

        self.to_rms()

    def from_syned(self, json_file: str) -> None:
        """
        Reads and sets up electron beam internal data from a SYNED JSON configuration file.

        Parameters:
            json_file (str): The path to the JSON file where the dictionary is written.
        """
        syned  = read_syned_file(json_file)
        self.energy_in_GeV = syned["electron_beam"]["energy_in_GeV"]
        self.energy_spread = syned["electron_beam"]["energy_spread"]
        self.current = syned["electron_beam"]["current"]
        self.number_of_bunches = syned["electron_beam"]["number_of_bunches"]
        self.moment_xx = syned["electron_beam"]["moment_xx"]
        self.moment_xxp = syned["electron_beam"]["moment_xxp"]
        self.moment_xpxp = syned["electron_beam"]["moment_xpxp"]
        self.moment_yy = syned["electron_beam"]["moment_yy"]
        self.moment_yyp = syned["electron_beam"]["moment_yyp"]
        self.moment_ypyp = syned["electron_beam"]["moment_ypyp"]

        self.to_rms()

    # def from_spectra(self):
    #     pass

    def propagate(self, dist: float) -> None:
        """
        Propagates electron beam statistical moments over a distance in free space.

        Args:
            dist (float): Distance the beam has to be propagated over in meters.
        """
        self.moment_xx  += (self.moment_xxp + self.moment_xpxp)*dist**2
        self.moment_xxp += (self.moment_xpxp)*dist
        self.moment_yy  += (self.moment_yyp + self.moment_ypyp)*dist**2
        self.moment_yyp += (self.moment_ypyp)*dist

        self.to_rms()

    def to_rms(self) -> None:
        """
        Computes the RMS sizes and divergences of the electron beam based on its 
        second-order statistical moments. The calculated values are stored back in the 
        object attributes (e_x, e_y, e_xp, e_yp).
        """
        self.e_x = np.sqrt(self.moment_xx)
        self.e_y = np.sqrt(self.moment_yy)
        self.e_xp = np.sqrt(self.moment_xpxp)
        self.e_yp = np.sqrt(self.moment_ypyp)

    def print_rms(self) -> None:
        """
        Prints electron beam rms sizes and divergences 
        """
        print("electron beam:")
        print(f"\t>> x/xp = {self.e_x*1e6:0.2f} um vs. {self.e_xp*1e6:0.2f} µrad")
        print(f"\t>> y/yp = {self.e_y*1e6:0.2f} um vs. {self.e_yp*1e6:0.2f} µrad")
            
    def get_gamma(self) -> float:
        """Calculate the Lorentz factor (γ) based on the energy of electrons in GeV."""
        return get_gamma(self.energy_in_GeV)

    def print_attributes(self) -> None:
        """
        Prints all attribute of object
        """
        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))


class MagneticStructure(object):
    """
    Class for defining magnetic structure parameters, including undulators, wigglers, and bending magnets.
    """
    def __init__(self, 
                 B_vertical: float = None, 
                 B_horizontal: float = None, 
                 mag_structure: str = None,
                 **kwargs) -> None:
        """
        Initializes the MagneticStructure class with parameters representing the magnetic field and geometry.

        Args:
            B_vertical (float): Vertical magnetic field component (in Tesla).
            B_horizontal (float): Horizontal magnetic field component (in Tesla).
            mag_structure (str): Type of magnetic structure, which can be:
                - "u", "und", "undulator" for undulator (default).
                - "w", "wig", "wiggler" for wiggler.
                - "bm", "bending magnet", "bending_magnet" for bending magnet.
            **kwargs: Additional optional parameters based on the magnetic structure type:
                - centre (float): centre of the mag. structure (in meters) in relation to where the electron beam moments are calculated. Defaults to 0.
                For undulators and wigglers:
                    - K_vertical (float): Vertical deflection parameter (K-value).
                    - B_vertical_phase (float): Phase offset for the vertical magnetic field in radians. Defaults to 0.
                    - B_horizontal_symmetry (int): Symmetry type for the vertical deflection parameter. Defaults to 1.
                            1 for symmetric      (B ~ cos(2*Pi*n*z/per + ph)),
                           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)).
                    - K_horizontal (float): Horizontal deflection parameter (K-value).
                    - B_horizontal_phase (float): Phase offset for the horizontal magnetic field in radians. Defaults to 0.
                    - B_vertical_symmetry (int): Symmetry type for the horizontal deflection parameter. Defaults to 1.
                            1 for symmetric      (B ~ cos(2*Pi*n*z/per + ph)),
                           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)).
                    - harmonic (int): Harmonic number. Defaults to 1.
                    - period_length (float): Length of one period (in meters).
                    - number_of_periods (int): Number of periods.
                For bending magnets:
                    - radius (float): Radius of curvature (in meters). Effective if > 0.
                    - length (float): Effective length of the magnet (in meters). Effective if > 0.
                    - length_edge (float): Soft edge length for field variation (10% to 90%) in meters. 
                                           Defaults to 0. Assumes a fringe field dependence of 
                                           G / (1 + ((z - zc) / d)^2)^2.
                    - extraction_angle (float):
                    - critical_energy (float):

        Raises:
            ValueError: If an invalid magnetic structure type is provided.
        """
        # Define valid magnetic structure types
        und = ['u', 'und', 'undulator']
        wig = ['w', 'wig', 'wiggler']
        bm = ['bm', 'bending magnet', 'bending_magnet']
        allowed_mag_structure = [ms for group in [und, wig, bm] for ms in group]

        if mag_structure not in allowed_mag_structure:
            raise ValueError(f"Invalid magnetic structure: {mag_structure}. Must be one of {allowed_mag_structure}")

        self.B_vertical = B_vertical
        self.B_horizontal = B_horizontal

        self.centre = kwargs.get("centre", 0)

        if mag_structure in und + wig:
            self.K_vertical = kwargs.get("K_vertical", None)
            self.B_vertical_phase = kwargs.get("B_vertical_phase", 0)
            self.B_vertical_symmetry = kwargs.get("B_vertical_symmetry", 1)
            self.K_horizontal = kwargs.get("K_horizontal", None)
            self.B_horizontal_phase = kwargs.get("B_horizontal_phase", 0)
            self.B_horizontal_symmetry = kwargs.get("B_horizontal_symmetry", 1)
            self.harmonic = kwargs.get("harmonic", 1)
            self.period_length = kwargs.get("period_length", None)
            self.number_of_periods = kwargs.get("number_of_periods", None)
            self.CLASS_NAME = "Undulator" if mag_structure in und else "Wiggler"

        if mag_structure in bm:
            self.CLASS_NAME = "BendingMagnet"
            self.magnetic_field = self.B_vertical
            self.radius = kwargs.get("radius", None)
            self.length = kwargs.get("length", None)
            self.length_edge = kwargs.get("length_edge", 0)
            self.extraction_angle = kwargs.get("extraction_angle", None)
            self.critical_energy = kwargs.get("critical_energy", None)

    def print_attributes(self) -> None:
        """
        Prints all attribute of object
        """
        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))


class SynchrotronSource(object):
    """
    Class representing a synchrotron radiation source, which combines an electron beam and a magnetic structure.
    """
    def __init__(self, electron_beam: ElectronBeam, magnetic_structure: MagneticStructure) -> None:
        """
        Initializes an instance of the SynchrotronSource class.

        Args:
            electron_beam (ElectronBeam): An instance of the ElectronBeam class representing the electron beam parameters.
            magnetic_structure (MagneticStructure): An instance of the MagneticStructure class representing the magnetic structure parameters.
        """
        self.ElectronBeam = electron_beam
        self.MagneticStructure = magnetic_structure
    
    def __getattr__(self, name):
        """
        Retrieves an attribute from either the ElectronBeam or MagneticStructure instances if it exists.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            The value of the attribute from either the ElectronBeam or MagneticStructure instance.

        """
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.ElectronBeam, name):
            return getattr(self.ElectronBeam, name)
        elif hasattr(self.MagneticStructure, name):
            return getattr(self.MagneticStructure, name)
        else:
            raise AttributeError(f"'SynchrotronSource' object has no attribute '{name}'")
        
    def write_syned_config(self, json_file: str, light_source_name: str=None):
        """
        Writes a SYNED JSON configuration file.

        Parameters:
            json_file (str): The path to the JSON file where the dictionary will be written.
            light_source_name (str): The name of the light source.
        """
        if light_source_name is None:
            light_source_name = json_file.split('/')[-1].replace('.json','')

        write_syned_file(json_file, light_source_name, self.ElectronBeam, self.MagneticStructure)

    # def write_spectra_config(self, json_file: str, light_source_name: str=None):
    #     """
    #     Writes a SPECTRA JSON configuration file.

    #     Parameters:
    #         json_file (str): The path to the JSON file where the dictionary will be written.
    #         light_source_name (str): The name of the light source.
    #     """
    #     if light_source_name is None:
    #         light_source_name = json_file.split('/')[-1].replace('.json','')

    #     write_spectra_file(json_file, light_source_name, self.ElectronBeam, self.MagneticStructure)

    def print_attributes(self) -> None:
        """
        Prints all attribute of object
        """
        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))

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

        # initializes attributes for a frozen object
        self.wavelength = None
        self.sigma_u = None
        self.sigma_up = None

        self.first_ring = None

        self.on_axis_flux = None
        self.central_cone_flux = None

        self.total_power = None
        self.power_though_slit = None

        super().__init__(**kwargs)

    def set_undulator(self, **kwargs) -> None:
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
            gamma = self.get_gamma()
            self.wavelength = self.period_length/(2 * self.harmonic * gamma ** 2)*(1+(K**2)/2) 
            
        mth_emittance = kwargs.get('mth_emittance', 0)
        mth_fillament_emittance = kwargs.get('mth_fillament_emittance', 0)

        cund = kwargs.get('center_undulator', 0)
        css = kwargs.get('center_straight_section', 0)

        if verbose:
            self.print_rms()

        self.set_filament_emittance(verbose=verbose, wavelength=wavelength, mth=mth_fillament_emittance)
        self.set_waist(verbose=verbose, center_undulator=cund, center_straight_section=css)
        self.set_emittance(verbose=verbose, mth=mth_emittance)

        self.set_radiation_rings(verbose=verbose)
        self.set_on_axis_flux(verbose=verbose)

        if complete:
            self.get_central_cone_flux(mtd=1, verbose=True)
        else:
            self.get_central_cone_flux(mtd=0, verbose=True)

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
        gamma = self.get_gamma()

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
        gamma = self.get_gamma()
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
        gamma = self.get_gamma()
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
            sigma_x_div = np.sqrt(self.sigma_up**2 + self.e_xp**2)
            sigma_y_div = np.sqrt(self.sigma_up**2 + self.e_yp**2)

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
            sigma_x_div = np.sqrt(self.sigma_up**2*_qa(self.energy_spread) + self.e_xp**2)
            sigma_y_div = np.sqrt(self.sigma_up**2*_qa(self.energy_spread) + self.e_yp**2)

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
        self.sigma_x_div = sigma_x_div
        self.sigma_y_div = sigma_y_div

        if verbose :        
            print("convolved photon beam:")
            print(f"\t>> x/xp = {sigma_x*1e6:0.2f} um vs. {sigma_x_div*1e6:0.2f} urad")
            print(f"\t>> y/yp = {sigma_y*1e6:0.2f} um vs. {sigma_y_div*1e6:0.2f} urad")

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

        gamma = self.get_gamma()
        dw_w = 0.1/100
        N2 = self.number_of_periods**2
        n = self.harmonic
        I = self.current
        d2Fn_d2phi = (ALPHA*N2*(gamma**2)*dw_w*I/CHARGE)*Fn(K, n)*1E-6
        
        # RC20250219 - sanity check
        # d2Fn_d2phi_bis = 1.74*1E14*(self.number_of_periods**2)*(self.energy_in_GeV**2)*self.current*Fn(K, self.harmonic)
        if verbose:
            print("on axis flux:")
            print(f"\t>> {d2Fn_d2phi:.3e} ph/s/mrad²/0.1%bw")
        self.on_axis_flux = d2Fn_d2phi

    def get_central_cone_flux(self, mtd=0, verbose: bool=False, **kwargs) -> float:
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
            flux = PI*ALPHA*N*dw_w*I/CHARGE*Qn
        else:
            nsigma = kwargs.get("nsigma", 6)
            dtn = kwargs.get("off_res_wavelength", self.wavelength)

            divergence = np.amax([self.sigma_x_div, self.sigma_y_div])
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
            flux = np.sum(photon_beam_divergence*self.on_axis_flux)*(x[1]-x[0])*(y[1]-y[0])*1E6

        if verbose:
            print("flux within the central cone:")
            print(f"\t>> {flux:.3e} ph/s/0.1%bw")

        self.central_cone_flux = flux

    def set_total_power(self, verbose: bool=False) -> float:
        """ 
        Calculate the total power emitted by a planar undulator in watts (W) based on Eq. 56 
        from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
        [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

        :param verbose: Whether to print results. Defaults to False.
        
        :return: Total power emitted by the undulator in watts (W).
        """
        gamma = self.get_gamma()

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

        npix = kwargs.get("npix", 501)
        nsigma = kwargs.get("nsigma", 6)
        hor_slit= kwargs.get("hor_slit", nsigma*self.sigma_x_div)
        ver_slit= kwargs.get("ver_slit", nsigma*self.sigma_y_div)

        K = np.sqrt(self.K_vertical**2 + self.K_horizontal**2)

        if self.K_vertical == 0 or self.K_horizontal == 0:
            und_type = 'planar'
        else:
            und_type = 'helical'

        gamma = self.get_gamma()

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
            print(f"Power emitted through a {hor_slit*1E3:.3f} x {ver_slit*1E3:.3f} mrad² slit: {CumPow:.3f} W")

        self.power_though_slit = {'power': CumPow, 'slit': {'dh':hor_slit, 'dv':ver_slit}}


class BendingMagnetSource(SynchrotronSource):
    """
    Class representing an bending magnet source, which combines an electron beam and 
    a magnetic structure.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes an instance of the BendingMagnet class.

        Args:
            electron_beam (ElectronBeam): An instance of the ElectronBeam class 
               representing the electron beam parameters.
            magnetic_structure (MagneticStructure): An instance of the MagneticStructure 
               class representing the magnetic structure parameters.
        """
        super().__init__(**kwargs)  

    def set_bending_magnet(self, **kwargs) -> None:
        """
        Sets the parameters of the bending magnet.

        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            critical_wavelength (float): The critical wavelength in meters. If provided,
               the magnetic field and radius of curvature are recalculated.

        """
        verbose = kwargs.get('verbose', False)
        critical_wavelength = kwargs.get('critical_wavelength', None)

        piloting = [critical_wavelength, self.critical_energy, self.magnetic_field, self.radius]

        if all(param is None for param in piloting):
            raise ValueError("Please, provide either the critical wavelength [m], the magnetic field [T] or bending radius [m]")

        if critical_wavelength is not None:
            self.set_bm_B_from_critical_energy(energy=energy_wavelength(critical_wavelength, unity='m'), 
                                               verbose=verbose)
            self.MagneticStructure.critical_energy = energy_wavelength(critical_wavelength, unity='m')
        else:
            if self.magnetic_field is not None:
                self.set_bm_radius_from_B(magnetic_field=self.magnetic_field, 
                                          verbose=verbose)
            else:
                self.set_bm_B_from_radius(radius=self.radius, 
                                          verbose=verbose)
                
        self.get_B_central_position(**kwargs)
                
    def set_bm_B_from_critical_energy(self, energy: float, verbose: bool=False) -> None:
        """
        Sets the magnetic field for a bending magnet based on the critical energy and  
        on the electron beam energy. Updates the bending radius.

        Args:
            critical energy (float): critical energy for a bending magnet based (in eV).
            verbose (bool): If True, prints additional information. Default is False.
        """
        self.MagneticStructure.critical_energy = energy
        gamma = self.get_gamma()
        self.MagneticStructure.magnetic_field = (energy*4*PI*MASS)/(3*PLANCK*gamma**2)
        self.set_bm_radius_from_B(self.magnetic_field, verbose)

        if verbose:
            print(f"Bending magnet critial energy energy set to {self.critical_energy:.3f} eV with:\n\
        >> B: {self.magnetic_field:.6f}\n\
        >> R: {self.radius:.6f}")

    def set_bm_radius_from_B(self, magnetic_field: float, verbose: bool=False) -> None:
        """
        Sets the radius of curvature from the magnetic field for a bending magnet based 
        on the electron beam energy.

        Args:
            magnetic_field (float): Vertical magnetic field component (in Tesla).
            eBeamEnergy (float): Energy of the electron beam in GeV.
            verbose (bool): If True, prints additional information. Default is False.
        """

        self.MagneticStructure.magnetic_field = magnetic_field
        gamma = self.get_gamma()
        e_speed = LIGHT * np.sqrt(1-1/gamma)
        self.MagneticStructure.radius = gamma * MASS * e_speed /(CHARGE * magnetic_field)
        self.MagneticStructure.critical_energy = self.get_critical_energy()

        if verbose:
            print(f"Bending magnet critial energy energy set to {self.critical_energy:.3f} eV with:\n\
        >> B: {self.magnetic_field:.6f}\n\
        >> R: {self.radius:.6f}")

    def set_bm_B_from_radius(self, radius:float, verbose: bool=False) -> None:
        """
        Sets the magnetic field from the radius of curvature for a bending magnet based 
        on the electron beam energy.

        Args:
            eBeamEnergy (float): Energy of the electron beam in GeV.
            verbose (bool): If True, prints additional information. Default is False.

        """
        self.MagneticStructure.radius = radius
        gamma = self.get_gamma()
        e_speed = LIGHT * np.sqrt(1-1/gamma)
        self.MagneticStructure.magnetic_field = gamma * MASS * e_speed /(CHARGE * radius)
        self.MagneticStructure.critical_energy = self.get_critical_energy()

        if verbose:
            print(f"Bending magnet critial energy energy set to {self.critical_energy:.3f} eV with:\n\
        >> B: {self.magnetic_field:.6f}\n\
        >> R: {self.radius:.6f}")

    def get_critical_energy(self):
        """
        Returns the critical energy for a bending magnet based on the electron beam energy
        """
        return (3*PLANCK*self.magnetic_field*self.get_gamma()**2)/(4*PI*MASS)

    def print_critical_energy(self) -> None:
        """
        Prints the critical energy for a bending magnet based on the electron beam energy

        Args:
            eBeamEnergy (float): Energy of the electron beam in GeV.
        """

        energy = self.get_critical_energy()
        print(f">> critical energy {energy:.2f} eV")

    def get_B_central_position(self, **kwargs):
        """
        Returns magnetic field central position to be used for the x-ray extraction.
        Extraction at 0 rad is towards the end of the magnetic field, while the largest
        angle is given by BM (length/radius) [radians] and defines the beginning of the 
        magnetic field.
        Keyword Args:
            verbose (bool): If True, prints additional information. Default is False.
            extraction_angle (floar): Extraction angle in radians. If provided, overwrites 
                self.MagneticStructure.extraction_angle
        """
        verbose = kwargs.get('verbose', False)

        self.MagneticStructure.extraction_angle = kwargs.get('extraction_angle',
                                                              self.MagneticStructure.extraction_angle)

        half_arc = 0.5*self.length/self.radius
        if self.extraction_angle is None:
            self.MagneticStructure.extraction_angle = half_arc
        zpos = (half_arc - self.extraction_angle)*self.radius
        if verbose:
            print(f"Arc segment of {2*half_arc:.3f} radians (L={self.length:.3f} m/R={self.radius:.3f} m).")
            print(f"Extraction at {-zpos:.3f} m from the centre of the bending magnet.")
            print(f">> {(0.5*self.length - zpos):.3f} m from the BM entrance.")
        return zpos

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
# **other** auxiliary functions 
#***********************************************************************************

def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fit_gaussian(x, fx):
    p0 = [np.max(fx), x[np.argmax(fx)], np.std(x)]
    popt, _ = opt.curve_fit(gaussian, x, fx, p0=p0)
    return popt
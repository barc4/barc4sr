
#!/bin/python

"""
This module provides ...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MAR/2024'
__changed__ = '22/MAR/2024'


import json
from time import time
from typing import Any, Dict

import numpy as np
from scipy.constants import physical_constants

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]
PI = np.pi

#***********************************************************************************
# auxiliary functions accelerator functions
#***********************************************************************************

def energy_wavelength(value: float, unity: str) -> float:
    """
    Converts energy to wavelength and vice versa.
    
    Parameters:
        value (float): The value of either energy or wavelength.
        unity (str): The unit of 'value'. Can be 'eV', 'meV', 'keV', 'm', 'nm', or 'A'. Case sensitive. 
        
    Returns:
        float: Converted value in meters if the input is energy, or in eV if the input is wavelength.
        
    Raises:
        ValueError: If an invalid unit is provided.
    """
    factor = 1.0
    
    # Determine the scaling factor based on the input unit
    if unity.endswith('eV') or unity.endswith('meV') or unity.endswith('keV'):
        prefix = unity[:-2]
        if prefix == "m":
            factor = 1e-3
        elif prefix == "k":
            factor = 1e3
    elif unity.endswith('m'):
        prefix = unity[:-1]
        if prefix == "n":
            factor = 1e-9
    elif unity.endswith('A'):
        factor = 1e-10
    else:
        raise ValueError("Invalid unit provided: {}".format(unity))

    return PLANCK * LIGHT / CHARGE / (value * factor)


def get_gamma(E: float) -> float:
    """
    Calculate the Lorentz factor (γ) based on the energy of electrons in GeV.

    Parameters:
        E (float): Energy of electrons in GeV.

    Returns:
        float: Lorentz factor (γ).
    """
    return E * 1e9 / (MASS * LIGHT ** 2) * CHARGE

#***********************************************************************************
# time stamp
#***********************************************************************************

def print_elapsed_time(start0: float) -> None:
    """
    Prints the elapsed time since the start of computation.

    Args:
        start0 (float): The start time of computation (in seconds since the epoch).
    """

    deltaT = time() - start0
    if deltaT < 1:
        print(f'>> Total elapsed time: {deltaT * 1000:.2f} ms')
    else:
        hours, rem = divmod(deltaT, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours >= 1:
            print(f'>> Total elapsed time: {int(hours)} h {int(minutes)} min {seconds:.2f} s')
        elif minutes >= 1:
            print(f'>> Total elapsed time: {int(minutes)} min {seconds:.2f} s')
        else:
            print(f'>> Total elapsed time: {seconds:.2f} s')

#***********************************************************************************
# potpourri
#***********************************************************************************

def generate_logarithmic_energy_values(emin: float, emax: float, resonant_energy: float, stepsize: float) -> np.ndarray:
    """
    Generate logarithmically spaced energy values within a given energy range.

    Args:
        emin (float): Lower energy range.
        emax (float): Upper energy range.
        resonant_energy (float): Resonant energy.
        stepsize (float): Step size.

    Returns:
        np.ndarray: Array of energy values with logarithmic spacing.
    """

    # Calculate the number of steps for positive and negative energy values
    n_steps_pos = np.ceil(np.log(emax / resonant_energy) / stepsize)
    n_steps_neg = max(0, np.floor(np.log(emin / resonant_energy) / stepsize))

    # Calculate the total number of steps
    n_steps = int(n_steps_pos - n_steps_neg)
    print(f"generate_logarithmic_energy_values - number of steps: {n_steps}")

    # Generate the array of steps with logarithmic spacing
    steps = np.linspace(n_steps_neg, n_steps_pos, n_steps + 1)

    # Compute and return the array of energy values
    return resonant_energy * np.exp(steps * stepsize)


#***********************************************************************************
# SYNED interface functions
#***********************************************************************************

class ElectronBeam:
    """
    Class for entering the electron beam parameters - this is based on SRWLPartBeam class
    """
    def __init__(self, energy: float = None, energy_spread: float = None, current: float = None,
                 number_of_bunches: int = 1, moment_xx: float = None, moment_xxp: float = None,
                 moment_xpxp: float = None, moment_yy: float = None, moment_yyp: float = None,
                 moment_ypyp: float = None, class_name: str = "ElectronBeam") -> None:
        """
        Initializes an instance of the ElectronBeam class.

        Args:
            energy (float): Energy of the electron beam in GeV. 
            energy_spread (float): RMS energy spread of the electron beam. 
            current (float): Average current of the electron beam in Amperes. 
            number_of_bunches (int): Number of bunches in the electron beam. 
            moment_xx (float): Second order moment: <(x-<x>)^2>. 
            moment_xxp (float): Second order moment: <(x-<x>)(x'-<x'>)>. 
            moment_xpxp (float): Second order moment: <(x'-<x'>)^2>. 
            moment_yy (float): Second order moment: <(y-<y>)^2>. 
            moment_yyp (float): Second order moment: <(y-<y>)(y'-<y'>)>. 
            moment_ypyp (float): Second order moment: <(y'-<y'>)^2>. 
            class_name (str): Name of the class instance. Defaults to "ElectronBeam".
        """  
        self.CLASS_NAME = class_name
        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current
        self.number_of_bunches = number_of_bunches
        self.moment_xx = moment_xx
        self.moment_xxp = moment_xxp
        self.moment_xpxp = moment_xpxp
        self.moment_yy = moment_yy
        self.moment_yyp = moment_yyp
        self.moment_ypyp = moment_ypyp

    def from_twiss(self, energy: float, energy_spread: float, current: float, emittance: float,
                   coupling: float, emittance_x: float, beta_x: float, alpha_x: float, eta_x: float,
                   etap_x: float, emittance_y: float, beta_y: float, alpha_y: float, eta_y: float,
                   etap_y: float) -> None:
        """
        Sets up electron beam internal data from Twiss parameters.

        Args:
            energy (float): Energy of the electron beam in GeV.
            energy_spread (float): RMS energy spread of the electron beam.
            current (float): Average current of the electron beam in Amperes.
            emittance (float): Emittance of the electron beam.
            coupling (float): Coupling coefficient between horizontal and vertical emittances.
            emittance_x (float): Horizontal emittance in meters.
            beta_x (float): Horizontal beta-function in meters.
            alpha_x (float): Horizontal alpha-function in radians.
            eta_x (float): Horizontal dispersion function in meters.
            etap_x (float): Horizontal dispersion function derivative in radians.
            emittance_y (float): Vertical emittance in meters.
            beta_y (float): Vertical beta-function in meters.
            alpha_y (float): Vertical alpha-function in radians.
            eta_y (float): Vertical dispersion function in meters.
            etap_y (float): Vertical dispersion function derivative in radians.
        """
        if emittance_x is None:
            emittance_x = emittance*(1/(coupling+1))
        if emittance_y is None:
            emittance_y = emittance*(coupling/(coupling+1))
            
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

class MagneticStructure:
    """
    Class for entering the undulator parameters
    """
    def __init__(self, K_vertical: float = None, K_horizontal: float = None,
                 period_length: float = None, number_of_periods: int = None,
                 class_name: str = "Undulator") -> None:
        """
        Initializes an instance of the MagneticStructure class.

        Args:
            K_vertical (float): Vertical magnetic parameter (K-value) of the undulator.
            K_horizontal (float): Horizontal magnetic parameter (K-value) of the undulator.
            period_length (float): Length of one period of the undulator in meters.
            number_of_periods (int): Number of periods of the undulator.
            class_name (str): Name of the class instance. Defaults to "Undulator".
        """
        self.CLASS_NAME = class_name
        self.K_vertical = K_vertical
        self.K_horizontal = K_horizontal
        self.period_length = period_length
        self.number_of_periods = number_of_periods

    def set_resonant_energy(self, energy: float, harmonic: int, eBeamEnergy: float, direction: str) -> None:
        """
        Sets the K-value based on the resonant energy and harmonic.

        Args:
            energy (float): Resonant energy in electron volts (eV).
            harmonic (int): Harmonic number.
            eBeamEnergy (float): Energy of the electron beam in GeV.
            direction (str): Direction of the undulator ('v' for vertical, 'h' for horizontal, 'b' for both).

        """
        wavelength = energy_wavelength(energy, 'eV')
        gamma = get_gamma(eBeamEnergy)

        if "v" in direction:
            self.K_vertical = np.sqrt((2 * (2 * harmonic * wavelength * gamma ** 2) / self.period_length  - 1))
        elif "h" in direction:
            self.K_horizontal = np.sqrt((2 * (2 * harmonic * wavelength * gamma ** 2) / self.period_length  - 1))
        else:
            K = np.sqrt((2 * (2 * harmonic * wavelength * gamma ** 2) / self.period_length  - 1))
            self.K_vertical = np.sqrt(K/2)
            self.K_horizontal = np.sqrt(K/2)

    def set_magnetic_field(self, Bx: float=None, By: float=None) -> None:
        """
        Sets the K-value based on the magnetic field strength.

        Args:
            Bx (float): Magnetic field strength in the horizontal direction.
                Defaults to None.
            By (float): Magnetic field strength in the vertical direction.
                Defaults to None.

        """
        if Bx is not None:
           self.K_horizontal = CHARGE * Bx * self.period_length / (2 * PI * MASS * LIGHT)
        if By is not None:
           self.K_horizontal = CHARGE * By * self.period_length / (2 * PI * MASS * LIGHT)


def write_syned_file(json_file: str, light_source_name: str, ElectronBeamClass: type, 
                     MagneticStructureClass: type) -> None:
    """
    Writes a Python dictionary into a SYNED JSON configuration file.

    Parameters:
        json_file (str): The path to the JSON file where the dictionary will be written.
        light_source_name (str): The name of the light source.
        ElectronBeamClass (type): The class representing electron beam parameters.
        MagneticStructureClass (type): The class representing magnetic structure parameters.
    """

    data = {
        "CLASS_NAME": "LightSource",
        "name": light_source_name,
        "electron_beam": vars(ElectronBeamClass),
        "magnetic_structure": vars(MagneticStructureClass)
    }

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def read_syned_file(json_file: str) -> Dict[str, Any]:
    """
    Reads a SYNED JSON configuration file and returns its contents as a dictionary.

    Parameters:
        json_file (str): The path to the SYNED JSON configuration file.

    Returns:
        dict: A dictionary containing the contents of the JSON file.
    """
    with open(json_file) as f:
        data = json.load(f)
    return data
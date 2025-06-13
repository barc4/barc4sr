#!/bin/python

"""
This module provides auxiliary functions for energy relations
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

import numpy as np
from scipy.constants import physical_constants

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]

#***********************************************************************************
# functions
#***********************************************************************************

def get_gamma(E: float) -> float:
    """
    Calculate the Lorentz factor (γ) based on the energy of electrons in GeV.

    Parameters:
        E (float): Energy of electrons in GeV.

    Returns:
        float: Lorentz factor (γ).
    """
    return E * 1e9 / (MASS * LIGHT ** 2) * CHARGE


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
    n_steps_neg = min(0, np.floor(np.log(emin / resonant_energy) / stepsize))

    # Calculate the total number of steps
    n_steps = int(n_steps_pos - n_steps_neg)
    print(f">>> generate_logarithmic_energy_values - number of steps: {n_steps} ({n_steps_neg} and {n_steps_pos}) around E0 with step size {stepsize:.3e}")

    # Generate the array of steps with logarithmic spacing
    steps = np.linspace(n_steps_neg, n_steps_pos, n_steps + 1)

    # Compute and return the array of energy values
    return resonant_energy * np.exp(steps * stepsize)
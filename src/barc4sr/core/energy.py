# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
energy.py - helper functions for energy, wavelength and related SR quantities.
"""

import os

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
    Calculate the Lorentz factor (gamma) based on the energy of electrons in GeV.

    Parameters:
        E (float): Energy of electrons in GeV.

    Returns:
        float: Lorentz factor (gamma).
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

def generate_logarithmic_energy_array(emin: float, emax: float, resonant_energy: float, 
                                      stepsize: float, verbose: bool=True) -> np.ndarray:
    """
    Generate logarithmically spaced energy values within a given energy range.

    Args:
        emin (float): Lower energy range.
        emax (float): Upper energy range.
        resonant_energy (float): Resonant energy.
        stepsize (float): Step size.
        verbose (bool): If True, print log information.
    Returns:
        np.ndarray: Array of energy values with logarithmic spacing.
    """

    n_steps_pos = np.ceil(np.log(emax / resonant_energy) / stepsize)
    n_steps_neg = min(0, np.floor(np.log(emin / resonant_energy) / stepsize))
    n_steps = int(n_steps_pos - n_steps_neg)
    if verbose: print(f">>> generate_logarithmic_energy_array - number of steps: {n_steps} ({n_steps_neg} and {n_steps_pos}) around E0 with step size {stepsize:.3e}")
    steps = np.linspace(n_steps_neg, n_steps_pos, n_steps + 1)
    return resonant_energy * np.exp(steps * stepsize)

def smart_split_energy(energy_array, num_cores):
        """
        Splits an array of energy values into chunks based on a weighted distribution.

        Parameters:
        energy_array (numpy.ndarray): The array of energy values to be split.
        num_cores (int): The number of chunks to split the energy array into.

        Returns:
        List[numpy.ndarray]: A list of numpy arrays, where each array is a chunk of the original energy array.
        """
        energy_array = np.sort(energy_array)
        weights = np.exp(-np.linspace(0, 2, num_cores))
        weights /= weights.sum()
        
        split_indices = np.cumsum(weights * len(energy_array)).astype(int)
        split_indices = np.insert(split_indices, 0, 0)
        split_indices[-1] = len(energy_array)

        energy_chunks = [energy_array[split_indices[i]:split_indices[i+1]] for i in range(num_cores)]
        return energy_chunks
    
def get_undulator_emission_energy(und_per: float, K: float, ring_e: float, n: int = 1, theta: float = 0) -> float:
    """
    Calculate the energy of an undulator emission in a storage ring.

    Parameters:
        und_per (float): Undulator period in meters.
        K (float): Undulator parameter.
        ring_e (float): Energy of electrons in GeV.
        n (int, optional): Harmonic number (default is 1).
        theta (float, optional): Observation angle in radians (default is 0).

    Returns:
        float: Emission energy in electron volts.
    """
    gamma = get_gamma(ring_e)
    emission_wavelength = und_per * (1 + (K ** 2) / 2 + (gamma * theta) ** 2) / (2 * n * gamma ** 2)

    return energy_wavelength(emission_wavelength, "m")

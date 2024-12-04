
"""
This module provides auxiliary for wavefronts
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

import numpy as np
from barc4sr.aux_energy import energy_wavelength
from skimage.restoration import unwrap_phase

#***********************************************************************************
# Stokes parameters
#***********************************************************************************

#***********************************************************************************
# phase unwrap
#***********************************************************************************

def unwrap_wft_phase(phase: np.array, x_axis: np.array, y_axis: np.array, 
                     observation_point: float, photon_energy: float) -> np.array:
    """
    Unwraps the wavefront phase by correcting for the quadratic phase term.

    This function corrects the wavefront phase by computing and subtracting 
    the quadratic phase term (QPT), then unwrapping the phase, and finally adding 
    the QPT back. The central values of the phase and QPT are adjusted to ensure 
    proper unwrapping.

    Parameters:
    phase (np.array): The 2D array representing the wavefront phase.
    x_axis (np.array): The 1D array representing the x-axis coordinates.
    y_axis (np.array): The 1D array representing the y-axis coordinates.
    observation_point (float): The distance to the observation point.
    photon_energy (float): The energy of the photons in electron volts (eV).

    Returns:
    np.array: The unwrapped wavefront phase.
    """
    
    # calculation of the quadratic phase term (QPT)
    X, Y = np.meshgrid(x_axis, y_axis)
    k = 2 * np.pi / energy_wavelength(photon_energy, 'eV')
    qpt = np.mod(k * (X**2 + Y**2) / (2 * observation_point), 2 * np.pi)
    qpt -= central_value(qpt)

    # Centering the phase and QPT
    phase -= central_value(phase)
    phase = np.mod(phase, 2 * np.pi) - qpt
    
    # Unwrapping the phase
    phase = unwrap_phase(phase)
    
    # Adding back the QPT
    phase += k * (X**2 + Y**2) / observation_point

    return phase


def central_value(arr: np.ndarray) -> float:
    """
    Calculate the central value of a 2D numpy array.
    
    If the number of rows and columns are both odd, return the central element.
    If one dimension is odd and the other is even, return the average of the two central elements.
    If both dimensions are even, return the average of the four central elements.

    Parameters:
    arr (np.ndarray): A 2D numpy array.

    Returns:
    float: The central value or the average of the central values.
    """
    rows, cols = arr.shape
    
    if rows % 2 == 1 and cols % 2 == 1:
        return arr[rows // 2, cols // 2]
    elif rows % 2 == 1 and cols % 2 == 0:
        return np.mean(arr[rows // 2, cols // 2 - 1:cols // 2 + 1])
    elif rows % 2 == 0 and cols % 2 == 1:
        return np.mean(arr[rows // 2 - 1:rows // 2 + 1, cols // 2])
    elif rows % 2 == 0 and cols % 2 == 0:
        return np.mean(arr[rows // 2 - 1:rows // 2 + 1, cols // 2 - 1:cols // 2 + 1])
    
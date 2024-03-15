
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '15/MARCH/2024'

from typing import Optional, Tuple

import numpy as np
import xraylib
from xoppylib.scattering_functions.xoppy_calc_f1f2 import xoppy_calc_f1f2

#***********************************************************************************
# reflectivity curves
#***********************************************************************************

def reflectivity_map(material: str, density: float, thetai: float, thetaf: float,
                     ntheta: int, ei: float, ef: float, ne: int,
                     e_axis: Optional[np.ndarray] = None, mat_flag: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a reflectivity map for a given material over a range of angles and energies.

    Args:
        material (str): The material's name.
        density (float): The material's density in g/cm^3.
        thetai (float): The initial angle of incidence in milliradians (mrad).
        thetaf (float): The final angle of incidence in milliradians (mrad).
        ntheta (int): The number of angles between thetai and thetaf.
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy points between ei and ef.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis. Defaults to None.
        mat_flag (int, optional): A flag indicating special treatment for the material. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The reflectivity map with shape (ntheta, ne) if e_axis is None, else (ntheta, len(e_axis)).
            - The energy axis.
    """

    theta = np.linspace(thetai, thetaf, ntheta)

    if e_axis is None:
        reflectivityMap = np.zeros((ntheta, ne))
    else:
        reflectivityMap = np.zeros((ntheta, len(e_axis)))

    for k, th in enumerate(theta):
        reflectivityMap[k,:], ene = reflectivity_curve(material, density, th, ei, ef, ne, e_axis, mat_flag)

    return reflectivityMap, ene


def reflectivity_curve(material: str, density: float, theta: float, ei: float, ef: float, ne: int,
                       e_axis: Optional[np.ndarray] = None, mat_flag: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Calculate the reflectivity for a given material and conditions.

    Args:
        material (str): The material's name.
        density (float): The material's density in grams per cubic centimeter (g/cm^3).
        theta (float): The angle of incidence in milliradians (mrad).
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy steps.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis for point-wise calculation. Defaults to None.
        mat_flag (int, optional): A parameter to control material parsing. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The reflectivity values.
            - The corresponding energy values.
    """

    if e_axis is None:
        out_dict =  xoppy_calc_f1f2(
                descriptor   = material,
                density      = density,
                MAT_FLAG     = mat_flag,
                CALCULATE    = 9,
                GRID         = 1,
                GRIDSTART    = ei,
                GRIDEND      = ef,
                GRIDN        = ne,
                THETAGRID    = 0,
                ROUGH        = 0.0,
                THETA1       = theta,
                THETA2       = 5.0,
                THETAN       = 50,
                DUMP_TO_FILE = 0,
                FILE_NAME    = "%s.dat"%material,
                material_constants_library = xraylib,
            )
        
        energy_axis = out_dict["data"][0,:]
        reflectivity = out_dict["data"][-1,:]
    else:
        k = 0
        for E in e_axis:
            out_dict =  xoppy_calc_f1f2(
            descriptor   = material,
            density      = density,
            MAT_FLAG     = mat_flag,
            CALCULATE    = 9,
            GRID         = 2,
            GRIDSTART    = E,
            GRIDEND      = E,
            GRIDN        = 1,
            THETAGRID    = 0,
            ROUGH        = 0.0,
            THETA1       = theta,
            THETA2       = 5.0,
            THETAN       = 50,
            DUMP_TO_FILE = 0,
            FILE_NAME    = "%s.dat"%material,
            material_constants_library = xraylib,
            )
            if k == 0:
                energy_axis = np.asarray(out_dict["data"][0,:], dtype="float64")
                reflectivity = np.asarray(out_dict["data"][-1,:], dtype="float64")
                k+=1
            else:
                energy_axis = np.concatenate((energy_axis, np.asarray(out_dict["data"][0,:], dtype="float64")), axis=0)
                reflectivity  = np.concatenate((reflectivity, np.asarray(out_dict["data"][-1,:], dtype="float64")), axis=0)
            
    return reflectivity, energy_axis


#***********************************************************************************
# transmission curves
#***********************************************************************************
#!/bin/python

"""
This module provides functions for interfacing SRW when calculating wavefronts, 
synchrotron radiation, power density, and spectra. This module is based on the 
xoppy.sources module from https://github.com/oasys-kit/xoppylib
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '12/JUN/2024'
__changed__ = '24/JUL/2025'

import os
import barc4sr.aux_common as cm

magnetic_structure_type = 'arb'

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def electron_trajectory(*args, **kwargs) -> dict:
    """
    Calculate electron trajectory using SRW.

    Args:
        file_name (str): The name of the output file.

    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class

    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """
    return cm.electron_trajectory(*args, magnetic_structure_type=magnetic_structure_type, **kwargs)

#***********************************************************************************
# Wavefront
#***********************************************************************************

def wavefront(*args, **kwargs) -> dict:
    """
    Calculate emitted wavefront using SRW.

    Args:
        file_name (str): The name of the output file.
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        observation_point (float): Distance to the observation point. Default is 10 [m].
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        radiation_polarisation (int): Polarisation component to be extracted. Default is 6.
            =0 -Linear Horizontal; 
            =1 -Linear Vertical; 
            =2 -Linear 45 degrees; 
            =3 -Linear 135 degrees; 
            =4 -Circular Right; 
            =5 -Circular Left; 
            =6 -Total
        number_macro_electrons (int): Number of macro electrons. Default is -1.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. 
                                    If not specified, it defaults to the number of available CPU cores.

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """

    return cm.wavefront(*args, magnetic_structure_type=magnetic_structure_type, **kwargs)

if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
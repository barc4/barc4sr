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
__created__ = '12/MAR/2024'
__changed__ = '08/JAN/2025'

import multiprocessing as mp
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.integrate as integrate
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.constants import physical_constants


from barc4sr.aux_energy import (
    energy_wavelength,
    generate_logarithmic_energy_values,
    get_gamma,
)
from barc4sr.aux_processing import (
    write_electron_trajectory,
    write_emitted_radiation,
    write_power_density,
    write_spectrum,
    write_wavefront,
)
from barc4sr.aux_syned import barc4sr_dictionary, syned_dictionary
from barc4sr.aux_time import print_elapsed_time
from barc4sr.aux_utils import (
    set_light_source,
    srwlibCalcElecFieldSR,
    srwlibsrwl_wfr_emit_prop_multi_e,
)

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]
Z0 = physical_constants["characteristic impedance of vacuum"][0]
EPSILON_0 = physical_constants["vacuum electric permittivity"][0] 
PI = np.pi

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def electron_trajectory(file_name: str, **kwargs) -> Dict:
    """
    Calculate undulator electron trajectory using SRW.

    Args:
        file_name (str): The name of the output file.


    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.


    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()

    function_txt = "Undulator electron trajectory using SRW:"

    print(f"{function_txt} please wait...")

    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, magnetic_measurement, 10, 1e-3, 1e-3, 0, 0)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, magnetic_measurement, 10, 1e-3, 1e-3, 0, 0)

    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, True, 'bm',
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               magnetic_measurement=magnetic_measurement
                                               )

    print('completed')
    print_elapsed_time(t0)

    eTrajDict = write_electron_trajectory(file_name, eTraj)

    return eTrajDict

#***********************************************************************************
# Bending magnet radiation
#***********************************************************************************
 
def spectrum(file_name: str,
             photon_energy_min: float,
             photon_energy_max: float,
             photon_energy_points: int, 
             **kwargs) -> Dict:
    """
    Calculate 1D bending magnet spectrum using SRW.

    Args:
        file_name (str): The name of the output file.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.

    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        energy_sampling (int): Energy sampling method (0: linear, 1: logarithmic). Default is 0.
        observation_point (float): Distance to the observation point. Default is 10 [m].
        hor_slit (float): Horizontal slit size [m]. Default is 1e-3 [m].
        ver_slit (float): Vertical slit size [m]. Default is 1e-3 [m].
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
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        magfield_central_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.            
        number_macro_electrons (int): Number of macro electrons. Default is 1000.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                        it defaults to the number of available CPU cores.

    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()

    function_txt = "Bending magnet spectrum calculation using SRW:"
    calc_txt = "> Performing flux through finite aperture (___CALC___ partially-coherent simulation)"
    print(f"{function_txt} please wait...")

    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    energy_sampling = kwargs.get('energy_sampling', 0)

    observation_point = kwargs.get('observation_point', 10.)
    
    hor_slit = kwargs.get('hor_slit', 1e-3)
    ver_slit = kwargs.get('ver_slit', 1e-3)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    
    radiation_polarisation = kwargs.get('radiation_polarisation', 6)
    
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)

    number_macro_electrons = kwargs.get('number_macro_electrons', 1)

    parallel = kwargs.get('parallel', False)
    num_cores = kwargs.get('num_cores', None)

    if hor_slit < 1e-6 and ver_slit < 1e-6:
        calculation = 0
        hor_slit = 0
        ver_slit = 0
    else:
        if magnetic_measurement is None and number_macro_electrons == 1:
            calculation = 1
        if magnetic_measurement is not None or number_macro_electrons > 1:
            calculation = 2

    if json_file is not None:
        bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
                            hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, magnetic_measurement, observation_point, 
                            hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, False, 'bm',
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               magnetic_measurement=magnetic_measurement
                                               )

    # ----------------------------------------------------------------------------------
    # spectrum calculations
    # ----------------------------------------------------------------------------------

    if energy_sampling == 0: 
        energy = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        stepsize = np.log(photon_energy_max/photon_energy_min)
        energy = generate_logarithmic_energy_values(photon_energy_min,
                                                    photon_energy_max,
                                                    photon_energy_min,
                                                    stepsize)
    # ---------------------------------------------------------
    # On-Axis Spectrum from Filament Electron Beam
    if calculation == 0:
        if parallel:
            print('> Performing on-axis spectrum from filament electron beam in parallel ... ')
        else:
            print('> Performing on-axis spectrum from filament electron beam ... ', end='')

        flux, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
                                                     eBeam, 
                                                     magFldCnt,
                                                     energy,
                                                     h_slit_points=1,
                                                     v_slit_points=1,
                                                     radiation_characteristic=0, 
                                                     radiation_dependence=0,
                                                     radiation_polarisation=radiation_polarisation,
                                                     id_type='bm',
                                                     parallel=parallel,
                                                     num_cores=num_cores)
        flux = flux.reshape((photon_energy_points))
        print('completed')

    # -----------------------------------------
    # Flux through Finite Aperture

    # # simplified partially-coherent simulation
    # if calculation == 1:
    #     calc_txt = calc_txt.replace("___CALC___", "simplified")
    #     if parallel:
    #         print(f'{calc_txt} in parallel... ')
    #     else:
    #         print(f'{calc_txt} ... ', end='')

    #     # RC:2025JAN08 TODO: check the best implementation
    #     print('completed')

    # accurate partially-coherent simulation
    if calculation == 2:
        calc_txt = calc_txt.replace("___CALC___", "accurate")
        if parallel:
            print(f'{calc_txt} in parallel... ')
        else:
            print(f'{calc_txt} ... ', end='')

        flux, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                eBeam,
                                                                magFldCnt,
                                                                energy,
                                                                h_slit_points=1,
                                                                v_slit_points=1,
                                                                radiation_polarisation=radiation_polarisation,
                                                                id_type='bm',
                                                                number_macro_electrons=number_macro_electrons,
                                                                aux_file_name=file_name,
                                                                parallel=parallel,
                                                                num_cores=num_cores)       
        print('completed')

    spectrumSRdict = write_spectrum(file_name, flux, energy)

    print(f"{function_txt} finished.")

    print_elapsed_time(t0)

    return spectrumSRdict


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")
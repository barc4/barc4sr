#!/bin/python

"""
This module provides functions for interfacing SRW when calculating wavefronts, 
synchrotron radiation, power density, and spectra. This module is based on the 
xoppy.sources module from https://github.com/oasys-kit/xoppylib
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '12/MAR/2024'
__changed__ = '19/JUL/2024'

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import numpy as np
import scipy.integrate as integrate
from scipy.constants import physical_constants

from barc4sr.aux_processing import (
    write_emitted_radiation,
    write_power_density,
    write_spectrum,
    write_wavefront,
)
from barc4sr.aux_utils import (
    energy_wavelength,
    generate_logarithmic_energy_values,
    get_gamma,
    print_elapsed_time,
    set_light_source,
    srwlibCalcElecFieldSR,
    srwlibsrwl_wfr_emit_prop_multi_e,
    syned_dictionary,
    unwrap_wft_phase,
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
PI = np.pi
RMS = np.sqrt(2)/2
#***********************************************************************************
# Bending magnet radiation
#***********************************************************************************
 
def spectrum(file_name: str,
             json_file: str,
             photon_energy_min: float,
             photon_energy_max: float,
             photon_energy_points: int, 
             **kwargs) -> Dict:
    """
    Calculate 1D bending magnet spectrum using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.

    Optional Args (kwargs):
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
        magfield_central_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.
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

    energy_sampling = kwargs.get('energy_sampling', 0)
    observation_point = kwargs.get('observation_point', 10.)
    hor_slit = kwargs.get('hor_slit', 1e-3)
    ver_slit = kwargs.get('ver_slit', 1e-3)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    radiation_polarisation = kwargs.get('radiation_polarisation', 6)
    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)
    electron_trajectory = kwargs.get('electron_trajectory', False)
    filament_beam = kwargs.get('filament_beam', False)
    energy_spread = kwargs.get('energy_spread', True)
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

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
                          hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

   
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, electron_trajectory, 'bm',
                                               magnetic_measurement=magnetic_measurement,
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position)

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

    #     flux = srwlibCalcStokesUR(bl, 
    #                               eBeam, 
    #                               magFldCnt, 
    #                               energy, 
    #                               resonant_energy,
    #                               radiation_polarisation,
    #                               parallel,
    #                               num_cores)

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

    write_spectrum(file_name, flux, energy)

    print(f"{function_txt} finished.")
    print_elapsed_time(t0)

    return {'energy':energy, 'flux':flux}



#***********************************************************************************
# bending magnet auxiliary functions - power calculation
#***********************************************************************************

def total_power(ring_e: float, ring_curr: float, und_per: float, und_n_per: int,
              B: Optional[float] = None, K: Optional[float] = None,
              verbose: bool = False) -> float:
    """ 
    Calculate the total power emitted by a planar undulator in kilowatts (kW) based on Eq. 56 
    from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
    [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

    :param ring_e: Ring energy in gigaelectronvolts (GeV).
    :param ring_curr: Ring current in amperes (A).
    :param und_per: Undulator period in meters (m).
    :param und_n_per: Number of periods.
    :param B: Magnetic field in tesla (T). If not provided, it will be calculated based on K.
    :param K: Deflection parameter. Required if B is not provided.
    :param verbose: Whether to print intermediate calculation results. Defaults to False.
    
    :return: Total power emitted by the undulator in kilowatts (kW).
    """

    pass




if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")
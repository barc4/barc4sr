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
__changed__ = '12/JUN/2024'

import array
import copy
import json
import multiprocessing as mp
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import numpy as np
import scipy.integrate as integrate
from scipy.constants import physical_constants
from scipy.signal import find_peaks

from barc4sr.aux_utils import (
    energy_wavelength,
    generate_logarithmic_energy_values,
    get_gamma,
    print_elapsed_time,
    set_light_source,
    srwlibCalcElecFieldSR,
    srwlibsrwl_wfr_emit_prop_multi_e,
    syned_dictionary,
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

    function_txt = "Undulator spectrum calculation using SRW:"
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
                          hor_slit, ver_slit, hor_slit_cen, ver_slit_cen, 
                          Kh=Kh, Kh_phase=Kh_phase, Kh_symmetry=Kh_symmetry, 
                          Kv=Kv, Kv_phase=Kv_phase, Kv_symmetry=Kv_symmetry)

   
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, electron_trajectory, 'u',
                                               magnetic_measurement=magnetic_measurement,
                                               tabulated_undulator_mthd=tabulated_undulator_mthd)

    # ----------------------------------------------------------------------------------
    # spectrum calculations
    # ----------------------------------------------------------------------------------
    resonant_energy = get_emission_energy(bl['PeriodID'], 
                                        np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                        bl['ElectronEnergy'])
    if energy_sampling == 0: 
        energy = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        stepsize = np.log(photon_energy_max/resonant_energy)
        energy = generate_logarithmic_energy_values(photon_energy_min,
                                                    photon_energy_max,
                                                    resonant_energy,
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
                                                     id_type='u',
                                                     parallel=parallel,
                                                     num_cores=num_cores)
        flux = flux.reshape((photon_energy_points))
        print('completed')

    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)

    # simplified partially-coherent simulation
    if calculation == 1:
        calc_txt = calc_txt.replace("___CALC___", "simplified")
        if parallel:
            print(f'{calc_txt} in parallel... ')
        else:
            print(f'{calc_txt} ... ', end='')

        flux = srwlibCalcStokesUR(bl, 
                                  eBeam, 
                                  magFldCnt, 
                                  energy, 
                                  resonant_energy,
                                  radiation_polarisation,
                                  parallel,
                                  num_cores)

        print('completed')

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
                                                                id_type='u',
                                                                number_macro_electrons=number_macro_electrons,
                                                                aux_file_name=file_name,
                                                                parallel=parallel,
                                                                num_cores=num_cores)       
        print('completed')

    # file = open('%s_spectrum.pickle'%file_name, 'wb')
    # pickle.dump([energy, flux], file)
    # file.close()

    with h5.File('%s_spectrum.h5'%file_name, 'w') as f:
        group = f.create_group('XOPPY_SPECTRUM')
        intensity_group = group.create_group('Spectrum')
        intensity_group.create_dataset('energy', data=energy)
        intensity_group.create_dataset('flux', data=flux) 

    print(f"{function_txt} finished.")
    print_elapsed_time(t0)

    return {'energy':energy, 'flux':flux}


def power_density(file_name: str, 
                  json_file: str, 
                  hor_slit: float, 
                  hor_slit_n: int,
                  ver_slit: float,
                  ver_slit_n: int,
                  **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate undulator power density spatial distribution using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        observation_point (float): Distance to the observation point. Default is 10 [m].
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        Kh (float): Horizontal undulator parameter K. If -1, taken from the SYNED file. Default is -1.
        Kh_phase (float): Initial phase of the horizontal magnetic field [rad]. Default is 0.
        Kh_symmetry (int): Symmetry of the horizontal magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)). Default is 1.
        Kv (float): Vertical undulator parameter K. If -1, taken from the SYNED file. Default is -1.
        Kv_phase (float): Initial phase of the vertical magnetic field [rad]. Default is 0.
        Kv_symmetry (int): Symmetry of the vertical magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)). Default is 1.
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field
            0: uses the provided magnetic field, 
            1: fits the magnetic field using srwl.UtiUndFromMagFldTab). Default is 0.
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing power density, horizontal axis, and vertical axis.
    """
    
    t0 = time.time()

    print("Undulator power density spatial distribution using SRW. Please wait...")

    observation_point = kwargs.get('observation_point', 10.)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    Kh = kwargs.get('Kh', -1)
    Kh_phase = kwargs.get('Kh_phase', 0)
    Kh_symmetry = kwargs.get('Kh_symmetry', 1)
    Kv = kwargs.get('Kv', -1)
    Kv_phase = kwargs.get('Kv_phase', 0)
    Kv_symmetry = kwargs.get('Kv_symmetry', 1)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)
    electron_trajectory = kwargs.get('electron_trajectory', False)
    filament_beam = kwargs.get('filament_beam', False)
    energy_spread = kwargs.get('energy_spread', True)

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, hor_slit, 
                    ver_slit, hor_slit_cen, ver_slit_cen, Kh, Kh_phase, Kh_symmetry, 
                    Kv, Kv_phase, Kv_symmetry)
    
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, magnetic_measurement,
                                               tabulated_undulator_mthd, 
                                               electron_trajectory)
    
    # ----------------------------------------------------------------------------------
    # power density calculations
    # ----------------------------------------------------------------------------------

    #***********Precision Parameters
    arPrecPar = [0]*5     # for power density
    arPrecPar[0] = 1.5    # precision factor
    arPrecPar[1] = 1      # power density computation method (1- "near field", 2- "far field")
    arPrecPar[2] = 0.0    # initial longitudinal position (effective if arPrecPar[2] < arPrecPar[3])
    arPrecPar[3] = 0.0    # final longitudinal position (effective if arPrecPar[2] < arPrecPar[3])
    arPrecPar[4] = 20000  # number of points for (intermediate) trajectory calculation

    stk = srwlib.SRWLStokes() 
    stk.allocate(1, hor_slit_n, ver_slit_n)     
    stk.mesh.zStart = bl['distance']
    stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
    stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
    stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
    stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2

    print('> Undulator power density spatial distribution using SRW ... ', end='')
    srwlib.srwl.CalcPowDenSR(stk, eBeam, 0, magFldCnt, arPrecPar)
    print('completed')

    power_density = np.reshape(stk.arS[0:stk.mesh.ny*stk.mesh.nx], (stk.mesh.ny, stk.mesh.nx))
    h_axis = np.linspace(-bl['slitH'] / 2, bl['slitH'] / 2, hor_slit_n)
    v_axis = np.linspace(-bl['slitV'] / 2, bl['slitV'] / 2, ver_slit_n)

    with h5.File('%s_power_density.h5'%file_name, 'w') as f:
        group = f.create_group('XOPPY_POWERDENSITY')
        sub_group = group.create_group('PowerDensity')
        sub_group.create_dataset('image_data', data=power_density)
        sub_group.create_dataset('axis_x', data=h_axis*1e3)    # axis in [mm]
        sub_group.create_dataset('axis_y', data=v_axis*1e3)

    print("Undulator power density spatial distribution using SRW: finished")
    print_elapsed_time(t0)

    return power_density, h_axis, v_axis


def emitted_radiation(file_name: str, 
                      json_file: str, 
                      photon_energy_min: float,
                      photon_energy_max: float,
                      photon_energy_points: int, 
                      hor_slit: float, 
                      hor_slit_n: int,
                      ver_slit: float,
                      ver_slit_n: int,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate undulator radiation spatial and spectral distribution using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        energy_sampling = kwargs.get('energy_sampling', 0)
        observation_point (float): Distance to the observation point. Default is 10 [m].
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        Kh (float): Horizontal undulator parameter K. If -1, taken from the SYNED file. Default is -1.
        Kh_phase (float): Initial phase of the horizontal magnetic field [rad]. Default is 0.
        Kh_symmetry (int): Symmetry of the horizontal magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)). Default is 1.
        Kv (float): Vertical undulator parameter K. If -1, taken from the SYNED file. Default is -1.
        Kv_phase (float): Initial phase of the vertical magnetic field [rad]. Default is 0.
        Kv_symmetry (int): Symmetry of the vertical magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)). Default is 1.
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field
            0: uses the provided magnetic field, 
            1: fits the magnetic field using srwl.UtiUndFromMagFldTab). Default is 0.
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.
        number_macro_electrons (int): Number of macro electrons. Default is 1000.
        parallel (bool): Whether to use parallel computation. Default is False.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing power density, horizontal axis, and vertical axis.
    """

    t0 = time.time()

    print("Undulator radiation spatial and spectral distribution using SRW. Please wait...")

    energy_sampling = kwargs.get('energy_sampling', 0)
    observation_point = kwargs.get('observation_point', 10.)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    Kh = kwargs.get('Kh', -1)
    Kh_phase = kwargs.get('Kh_phase', 0)
    Kh_symmetry = kwargs.get('Kh_symmetry', 1)
    Kv = kwargs.get('Kv', -1)
    Kv_phase = kwargs.get('Kv_phase', 0)
    Kv_symmetry = kwargs.get('Kv_symmetry', 1)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)
    electron_trajectory = kwargs.get('electron_trajectory', False)
    filament_beam = kwargs.get('filament_beam', False)
    energy_spread = kwargs.get('energy_spread', True)
    number_macro_electrons = kwargs.get('number_macro_electrons', -1)
    parallel = kwargs.get('parallel', False)
    num_cores = kwargs.get('num_cores', None)

    if number_macro_electrons <= 0 :
        calculation = 0
    else:
        calculation = 1

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, hor_slit, 
                    ver_slit, hor_slit_cen, ver_slit_cen, Kh, Kh_phase, Kh_symmetry, 
                    Kv, Kv_phase, Kv_symmetry)
    
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, magnetic_measurement,
                                               tabulated_undulator_mthd, 
                                               electron_trajectory)
    
    # ----------------------------------------------------------------------------------
    # spectrum calculations
    # ----------------------------------------------------------------------------------
    resonant_energy = get_emission_energy(bl['PeriodID'], 
                                        np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                        bl['ElectronEnergy'])
    if energy_sampling == 0: 
        energy = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        stepsize = np.log(photon_energy_max/resonant_energy)
        energy = generate_logarithmic_energy_values(photon_energy_min,
                                                          photon_energy_max,
                                                          resonant_energy,
                                                          stepsize)
    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)
        
    # simplified partially-coherent simulation    
    if calculation == 0:
        if parallel:
            print('> Performing flux through finite aperture (simplified partially-coherent simulation) in parallel... ')
        else:
            print('> Performing flux through finite aperture (simplified partially-coherent simulation)... ', end='')

        intensity, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
                                                          eBeam, 
                                                          magFldCnt,
                                                          energy,
                                                          h_slit_points=hor_slit_n,
                                                          v_slit_points=ver_slit_n,
                                                          radiation_characteristic=1, 
                                                          radiation_dependence=3,
                                                          parallel=parallel)        
        print('completed')

    # accurate partially-coherent simulation
    if calculation == 1:
        if parallel:
            print('> Performing flux through finite aperture (accurate partially-coherent simulation) in parallel... ')
        else:
            print('> Performing flux through finite aperture (accurate partially-coherent simulation)...')

        intensity, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                     eBeam,
                                                                     magFldCnt,
                                                                     energy,
                                                                     hor_slit_n,
                                                                     ver_slit_n,
                                                                     number_macro_electrons,
                                                                     file_name,
                                                                     parallel,
                                                                     num_cores) 
        print('completed')
    
    with h5.File('%s_undulator_radiation.h5'%file_name, 'w') as f:
        group = f.create_group('XOPPY_RADIATION')
        radiation_group = group.create_group('Radiation')
        radiation_group.create_dataset('stack_data', data=intensity)
        radiation_group.create_dataset('axis0', data=energy)
        radiation_group.create_dataset('axis1', data=h_axis*1e3)
        radiation_group.create_dataset('axis2', data=v_axis*1e3)

    print("Undulator radiation spatial and spectral distribution using SRW: finished")
    print_elapsed_time(t0)

    return energy, intensity, h_axis, v_axis


def emitted_wavefront(file_name: str, 
                      json_file: str, 
                      photon_energy: float,
                      hor_slit: float, 
                      hor_slit_n: int,
                      ver_slit: float,
                      ver_slit_n: int,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate undulator radiation spatial and spectral distribution using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        photon_energy (float): Photon energy [eV].
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
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
        magfield_central_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (str): Path to the file containing magnetic measurement data.
            Overrides SYNED bending magnet data. Default is None.
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.
        number_macro_electrons (int): Number of macro electrons. Default is -1.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int): Number of CPU cores to use for parallel computation. If not specified, 
            it defaults to the number of available CPU cores.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing power density, horizontal axis, and vertical axis.
    """

    t0 = time.time()

    function_txt = "Bending magnet spatial distribution for a given energy using SRW:"
    calc_txt = "> Performing monochromatic wavefront calculation (___CALC___ partially-coherent simulation)"
    print(f"{function_txt} please wait...")

    observation_point = kwargs.get('observation_point', 10.)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    radiation_polarisation = kwargs.get('radiation_polarisation', 6)
    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)
    electron_trajectory = kwargs.get('electron_trajectory', False)
    filament_beam = kwargs.get('filament_beam', False)
    energy_spread = kwargs.get('energy_spread', True)
    number_macro_electrons = kwargs.get('number_macro_electrons', -1)
    parallel = kwargs.get('parallel', False)
    num_cores = kwargs.get('num_cores', None)

    if number_macro_electrons <= 0 :
        calculation = 0
    else:
        calculation = 1

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, hor_slit, 
                          ver_slit, hor_slit_cen, ver_slit_cen)
    
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, electron_trajectory, 'bm',
                                               magnetic_measurement=magnetic_measurement,
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position)
    
    # -----------------------------------------
    # Spatial limited monochromatic wavefront (total pol.)
        
    # simplified partially-coherent simulation    
    if calculation == 0:
        calc_txt = calc_txt.replace("___CALC___", "simplified")
        print(f'{calc_txt} ... ', end='')

        intensity, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
                                                          eBeam, 
                                                          magFldCnt,
                                                          photon_energy,
                                                          h_slit_points=hor_slit_n,
                                                          v_slit_points=ver_slit_n,
                                                          radiation_characteristic=1, 
                                                          radiation_dependence=3,
                                                          radiation_polarisation=radiation_polarisation,
                                                          id_type = 'bm',
                                                          parallel=False)     
        phase, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
                                                      eBeam, 
                                                      magFldCnt,
                                                      photon_energy,
                                                      h_slit_points=hor_slit_n,
                                                      v_slit_points=ver_slit_n,
                                                      radiation_characteristic=4, 
                                                      radiation_dependence=3,
                                                      radiation_polarisation=radiation_polarisation,
                                                      id_type = 'bm',
                                                      parallel=False)     
        
        print('completed')

    # accurate partially-coherent simulation
    if calculation == 1:
        calc_txt = calc_txt.replace("___CALC___", "accurate")
        if parallel:
            print(f'{calc_txt} in parallel... ')
        else:
            print(f'{calc_txt} ... ', end='')

        intensity, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                     eBeam,
                                                                     magFldCnt,
                                                                     photon_energy,
                                                                     hor_slit_n,
                                                                     ver_slit_n,
                                                                     radiation_polarisation=radiation_polarisation,
                                                                     id_type = 'bm',
                                                                     number_macro_electrons=number_macro_electrons,
                                                                     aux_file_name=file_name,
                                                                     parallel=parallel,
                                                                     num_cores=num_cores) 
        
        phase = np.zeros(intensity.shape)
        print('completed')
    
    with h5.File('%s_bending_magnet_wft.h5'%file_name, 'w') as f:
        group = f.create_group('XOPPY_WAVEFRONT')
        intensity_group = group.create_group('Intensity')
        intensity_group.create_dataset('image_data', data=intensity)
        intensity_group.create_dataset('axis_x', data=h_axis*1e3) 
        intensity_group.create_dataset('axis_y', data=v_axis*1e3)
        intensity_group = group.create_group('Phase')
        intensity_group.create_dataset('image_data', data=phase)
        intensity_group.create_dataset('axis_x', data=h_axis*1e3) 
        intensity_group.create_dataset('axis_y', data=v_axis*1e3)

    print(f"{function_txt} finished.")
    print_elapsed_time(t0)

    return {"photon_energy":photon_energy, "wavefront": {"phase":phase, "intensity":intensity}, "axis":{"x":h_axis, "y":v_axis} }

#***********************************************************************************
# read/write functions for magnetic fields
#***********************************************************************************

def read_magnetic_measurement(file_path: str) -> np.ndarray:
    """
    Read magnetic measurement data from a file.

    Parameters:
        file_path (str): The path to the file containing magnetic measurement data.

    Returns:
        np.ndarray: A NumPy array containing the magnetic measurement data.
    """

    data = []

    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                values = line.split( )
                data.append([float(value) for value in values])
                
    return np.asarray(data)


def generate_srw_magnetic_field(mag_field_array: np.ndarray, file_path: Optional[str] = None) -> srwlib.SRWLMagFld3D:
    """
    Generate a 3D magnetic field object based on the input magnetic field array.

    Parameters:
        mag_field_array (np.ndarray): Array containing magnetic field data. Each row corresponds to a point in the 3D space,
                                      where the first column represents the position along the longitudinal axis, and subsequent 
                                      columns represent magnetic field components (e.g., Bx, By, Bz).
        file_path (str, optional): File path to save the generated magnetic field object. If None, the object won't be saved.

    Returns:
        SRWLMagFld3D: Generated 3D magnetic field object.

    """
    nfield, ncomponents = mag_field_array.shape

    field_axis = (mag_field_array[:, 0] - np.mean(mag_field_array[:, 0])) * 1e-3

    Bx = mag_field_array[:, 1]
    if ncomponents > 2:
        By = mag_field_array[:, 2]
    else:
        By = np.zeros(nfield)
    if ncomponents > 3:
        Bz = mag_field_array[:, 3]
    else:
        Bz = np.zeros(nfield)

    magFldCnt = srwlib.SRWLMagFld3D(Bx, By, Bz, 1, 1, nfield - 1, 0, 0, field_axis[-1]-field_axis[0], 1)

    if file_path is not None:
        print(f">>> saving {file_path}")
        magFldCnt.save_ascii(file_path)

    return magFldCnt

#***********************************************************************************
# magnetic field properties
#***********************************************************************************

def get_magnetic_field_properties(mag_field_component: np.ndarray, 
                                  field_axis: np.ndarray, 
                                  threshold: float = 0.7,
                                  **kwargs: Any) -> Dict[str, Any]:
    """
    Perform analysis on magnetic field data.

    Parameters:
        mag_field_component (np.ndarray): Array containing magnetic field component data.
        field_axis (np.ndarray): Array containing axis data corresponding to the magnetic field component.
        threshold (float, optional): Peak detection threshold as a fraction of the maximum value.
        **kwargs: Additional keyword arguments.
            - pad (bool): Whether to pad the magnetic field component array for FFT analysis. Default is False.
            - positive_side (bool): Whether to consider only the positive side of the FFT. Default is True.

    Returns:
        Dict[str, Any]: Dictionary containing magnetic field properties including mean, standard deviation,
                        undulator period mean, undulator period standard deviation, frequency, FFT data, 
                        and first and second field integrals.
                        
        Dictionary structure:
            {
                "field": np.ndarray,                  # Array containing the magnetic field component data.
                "axis": np.ndarray,                   # Array containing the axis data corresponding to the magnetic field component.
                "mag_field_mean": float,              # Mean value of the magnetic field component.
                "mag_field_std": float,               # Standard deviation of the magnetic field component.
                "und_period_mean": float,             # Mean value of the undulator period.
                "und_period_std": float,              # Standard deviation of the undulator period.
                "first_field_integral": np.ndarray,   # Array containing the first field integral.
                "second_field_integral": np.ndarray,  # Array containing the first field integral.
                "freq": np.ndarray,                   # Array containing frequency data for FFT analysis.
                "fft": np.ndarray                     # Array containing FFT data.
            }
    """

    field_axis -= np.mean(field_axis)
    peaks, _ = find_peaks(mag_field_component, height=np.amax(mag_field_component) *threshold)

    # Calculate distances between consecutive peaks
    periods = np.diff(field_axis[peaks])
    average_period = np.mean(periods)
    period_dispersion = np.std(periods)
    average_peak = np.mean(mag_field_component[peaks])
    peak_dispersion = np.std(mag_field_component[peaks])

    print(f"Number of peaks over 0.7*Bmax: {len(peaks)}")
    print(f"Average period: {average_period * 1e3:.3f}+-{period_dispersion * 1e3:.3f} [mm]")
    print(f"Average peak: {average_peak:.3f}+-{peak_dispersion:.3f} [T]")

    mag_field_properties = {
        "field": mag_field_component,
        "axis": field_axis,
        "mag_field_mean": average_peak,
        "mag_field_std": peak_dispersion,
        "und_period_mean": average_period,
        "und_period_std": period_dispersion,
    }
    # # RC20240315: debug
    # import matplotlib.pyplot as plt
    # plt.plot(mag_field_component)
    # plt.plot(peaks, mag_field_component[peaks], "x")
    # plt.plot(np.zeros_like(mag_field_component), "--", color="gray")
    # plt.show()

    # First and second field integrals

    first_field_integral = integrate.cumulative_trapezoid(mag_field_component, field_axis, initial=0)
    second_field_integral = integrate.cumulative_trapezoid(first_field_integral, field_axis, initial=0)

    mag_field_properties["first_field_integral"] = first_field_integral
    mag_field_properties["second_field_integral"] = second_field_integral

    # Frequency analysis of the magnetic field
    pad = kwargs.get("pad", False)
    positive_side = kwargs.get("positive_side", True)

    if pad:
        mag_field_component = np.pad(mag_field_component, 
                                     (int(len(mag_field_component) / 2), 
                                      int(len(mag_field_component) / 2)))
    
    naxis = len(mag_field_component)
    daxis = field_axis[1] - field_axis[0]

    fftfield = np.abs(np.fft.fftshift(np.fft.fft(mag_field_component)))
    freq = np.fft.fftshift(np.fft.fftfreq(naxis, daxis))

    if positive_side:
        fftfield = 2 * fftfield[freq > 0]
        freq = freq[freq > 0]

    mag_field_properties["freq"] = freq
    mag_field_properties["fft"] = fftfield

    return mag_field_properties

#***********************************************************************************
# undulator auxiliary functions - power calculation
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

    if B is None:
        if K is None:
            raise TypeError("Please, provide either B or K for the undulator")
        else:
            B = get_B_from_K(K, und_per)
            if verbose:
                print(">>> B = %.5f [T]"%B)

    return 0.63*(ring_e**2)*(B**2)*ring_curr*und_per*und_n_per



#***********************************************************************************
# Potpourri
#***********************************************************************************


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")
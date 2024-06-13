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
from joblib import Parallel, delayed
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from barc4sr.utils import (
    energy_wavelength,
    generate_logarithmic_energy_values,
    get_gamma,
    print_elapsed_time,
    set_light_source,
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
# xoppy_undulators.py inspired modules for undulator radiation
#***********************************************************************************
 
def spectrum(file_name: str,
             json_file: str,
             photon_energy_min: float,
             photon_energy_max: float,
             photon_energy_points: int, 
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 1D undulator spectrum using SRW.

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
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                        it defaults to the number of available CPU cores.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of photon energy and flux.
    """

    t0 = time.time()

    print("Undulator spectrum calculation using SRW. Please wait...")

    energy_sampling = kwargs.get('energy_sampling', 0)
    observation_point = kwargs.get('observation_point', 10.)
    hor_slit = kwargs.get('hor_slit', 1e-3)
    ver_slit = kwargs.get('ver_slit', 1e-3)
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

    1/0
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
    # On-Axis Spectrum from Filament Electron Beam (total pol.)
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
                                                     parallel=parallel,
                                                     num_cores=num_cores)
        flux = flux.reshape((photon_energy_points))
        print('completed')
    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)

    # simplified partially-coherent simulation
    if calculation == 1:
        if parallel:
            print('> Performing flux through finite aperture (simplified partially-coherent simulation) in parallel... ')
        else:
            print('> Performing flux through finite aperture (simplified partially-coherent simulation)... ', end='')

        flux = srwlibCalcStokesUR(bl, 
                                  eBeam, 
                                  magFldCnt, 
                                  energy, 
                                  resonant_energy,
                                  parallel,
                                  num_cores)

        print('completed')

    # accurate partially-coherent simulation
    if calculation == 2:
        if parallel:
            print('> Performing flux through finite aperture (accurate partially-coherent simulation) in parallel... ')
        else:
            print('> Performing flux through finite aperture (accurate partially-coherent simulation)...')

        flux, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                eBeam,
                                                                magFldCnt,
                                                                energy,
                                                                1,
                                                                1,
                                                                number_macro_electrons,
                                                                file_name,
                                                                parallel,
                                                                num_cores)       
        print('completed')

    file = open('%s_spectrum.pickle'%file_name, 'wb')
    pickle.dump([energy, flux], file)
    file.close()

    print("Undulator spectrum calculation using SRW: finished")
    print_elapsed_time(t0)

    return energy, flux


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
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                        it defaults to the number of available CPU cores.

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


def emitted_wavefront():
    pass

#***********************************************************************************
# SRW interfaced functions
#***********************************************************************************

def srwlibCalcElecFieldSR(bl: dict, 
                          eBeam: srwlib.SRWLPartBeam, 
                          magFldCnt: srwlib.SRWLMagFldC, 
                          energy_array: np.ndarray,
                          h_slit_points: int, 
                          v_slit_points: int, 
                          radiation_characteristic: int, 
                          radiation_dependence: int, 
                          parallel: bool,
                          num_cores: int=None) -> np.ndarray:
    """
    Calculates the electric field for synchrotron radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_characteristic (int): Radiation characteristic:
               =0 -"Single-Electron" Intensity; 
               =1 -"Multi-Electron" Intensity; 
               =4 -"Single-Electron" Radiation Phase; 
               =5 -Re(E): Real part of Single-Electron Electric Field;
               =6 -Im(E): Imaginary part of Single-Electron Electric Field
        radiation_dependence (int): Radiation dependence (e.g., 1 for angular distribution).
               =0 -vs e (photon energy or time);
               =1 -vs x (horizontal position or angle);
               =2 -vs y (vertical position or angle);
               =3 -vs x&y (horizontal and vertical positions or angles);
               =4 -vs e&x (photon energy or time and horizontal position or angle);
               =5 -vs e&y (photon energy or time and vertical position or angle);
               =6 -vs e&x&y (photon energy or time, horizontal and vertical positions or angles);
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                            it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data, horizontal and vertical axes
    """
    
    arPrecPar = [0]*7
    arPrecPar[0] = 1     # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    arPrecPar[1] = 0.01  # relative precision
    arPrecPar[2] = 0     # longitudinal position to start integration (effective if < zEndInteg)
    arPrecPar[3] = 0     # longitudinal position to finish integration (effective if > zStartInteg)
    arPrecPar[4] = 50000 # Number of points for trajectory calculation
    arPrecPar[5] = 1     # Use "terminating terms"  or not (1 or 0 respectively)
    arPrecPar[6] = 0     # sampling factor for adjusting nx, ny (effective if > 0)

    if num_cores is None:
        num_cores = mp.cpu_count()

    dE = np.diff(energy_array)    
    dE1 = np.min(dE)
    dE2 = np.max(dE)

    wiggler_regime = bool(energy_array[-1]>200*energy_array[0])

    if parallel:
        if np.allclose(dE1, dE2) and wiggler_regime:
            chunk_size = 20
            n_slices = len(energy_array)

            chunks = [(energy_array[i:i + chunk_size],
                    bl, 
                    eBeam,
                    magFldCnt, 
                    arPrecPar, 
                    h_slit_points, 
                    v_slit_points, 
                    radiation_characteristic, 
                    radiation_dependence,
                    parallel) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(core_srwlibCalcElecFieldSR, chunks)
        else:
            dE = (energy_array[-1] - energy_array[0]) / num_cores
            energy_chunks = []

            for i in range(num_cores):
                bffr = copy.copy(energy_array)                
                bffr = np.delete(bffr, bffr < dE * (i) + energy_array[0])
                if i + 1 != num_cores:
                    bffr = np.delete(bffr, bffr >= dE * (i + 1) + energy_array[0])
                energy_chunks.append(bffr)

            results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcElecFieldSR)((
                                                                        list_pairs,
                                                                        bl,
                                                                        eBeam,
                                                                        magFldCnt,
                                                                        arPrecPar,
                                                                        h_slit_points,
                                                                        v_slit_points,
                                                                        radiation_characteristic,
                                                                        radiation_dependence,
                                                                        parallel))
                                                for list_pairs in energy_chunks)
            
        for i, (intensity_chunck, h_chunck, v_chunck, e_chunck, t_chunck) in enumerate(results):
            if i == 0:
                intensity = intensity_chunck
                energy_array = np.asarray([e_chunck[0]])
                energy_chunks = np.asarray([len(e_chunck)])
                time_array = np.asarray([t_chunck])
            else:
                intensity = np.concatenate((intensity, intensity_chunck), axis=0)
                energy_array = np.concatenate((energy_array, np.asarray([e_chunck[0]])))
                energy_chunks = np.concatenate((energy_chunks, np.asarray([len(e_chunck)])))
                time_array = np.concatenate((time_array, np.asarray([t_chunck])))

        if not wiggler_regime:
            print(">>> ellapse time:")
            for ptime in range(len(time_array)):
                print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")

    else:
        results = core_srwlibCalcElecFieldSR((energy_array,
                                             bl, 
                                             eBeam,
                                             magFldCnt, 
                                             arPrecPar, 
                                             h_slit_points, 
                                             v_slit_points, 
                                             radiation_characteristic, 
                                             radiation_dependence,
                                             parallel))
        intensity = results[0]

    if h_slit_points == 1 or v_slit_points == 1:
        x_axis = np.asarray([0])
        y_axis = np.asarray([0])
    else:
        x_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        y_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    return intensity, x_axis, y_axis


def core_srwlibCalcElecFieldSR(args: Tuple[np.ndarray, dict, srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, List[float], int, int, int, int, bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Core function to calculate electric field for synchrotron radiation.

    Args:
        args (Tuple): Tuple containing the following elements:
            energy_array (np.ndarray): Array of photon energies [eV].
            bl (dict): Dictionary containing beamline parameters.
            eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            arPrecPar (List[float]): Array of parameters for SR calculation.
            h_slit_points (int): Number of horizontal slit points.
            v_slit_points (int): Number of vertical slit points.
            rad_characteristic (int): Radiation characteristic (e.g., 0 for intensity).
            rad_dependence (int): Radiation dependence (e.g., 1 for angular distribution).
            parallel (bool): Whether to use parallel computation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Tuple containing intensity data, 
                                                          horizontal axis, vertical axis, 
                                                          and computation time.
    """

    energy_array, bl, eBeam, magFldCnt, arPrecPar,  h_slit_points, v_slit_points, \
        rad_characteristic, rad_dependence, parallel = args
    
    tzero = time.time()

    _inPol = 6
    _inIntType = rad_characteristic
    _inDepType = rad_dependence

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
        _inDepType = 0
        intensity = np.zeros((energy_array.size))
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        _inDepType = 3
        intensity = np.zeros((energy_array.size, vAxis.size, hAxis.size))

    if parallel:    
        # this is rather convinient for step by step calculations and less memory intensive
        for ie in range(energy_array.size):
            try:
                mesh = srwlib.SRWLRadMesh(energy_array[ie], energy_array[ie], 1,
                                         hAxis[0], hAxis[-1], h_slit_points,
                                         vAxis[0], vAxis[-1], v_slit_points, 
                                         bl['distance'])

                wfr = srwlib.SRWLWfr()
                wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
                wfr.mesh = mesh
                wfr.partBeam = eBeam

                srwlib.srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)
                arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) #"flat" array to take 2D intensity data
                srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)

                if _inDepType == 0:    # 0 -vs e (photon energy or time);
                    intensity[ie] = np.asarray(arI1, dtype="float64")

                else:
                    data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) #np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                    intensity[ie, :, :] = data
            except:
                 raise ValueError("Error running SRW.")
    else:
        try:
            mesh = srwlib.SRWLRadMesh(energy_array[0], energy_array[-1], len(energy_array),
                                    hAxis[0], hAxis[-1], h_slit_points,
                                    vAxis[0], vAxis[-1], v_slit_points, 
                                    bl['distance'])
            
            wfr = srwlib.SRWLWfr()
            wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
            wfr.mesh = mesh
            wfr.partBeam = eBeam

            # srwl_bl.calc_sr_se sets eTraj=0 despite having measured magnetic field
            srwlib.srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)

            if _inDepType == 0:    # 0 -vs e (photon energy or time);
                arI1 = array.array('f', [0]*wfr.mesh.ne)
                srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
                intensity = np.asarray(arI1, dtype="float64")
            else:
                for ie in range(len(energy_array)):
                    arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
                    srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, energy_array[ie], 0, 0)
                    data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) #np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                    intensity[ie, :, :] = data

        except:
             raise ValueError("Error running SRW.")

    return intensity, hAxis, vAxis, energy_array, time.time()-tzero


def srwlibCalcStokesUR(bl: dict, 
                       eBeam: srwlib.SRWLPartBeam, 
                       magFldCnt: srwlib.SRWLMagFldC, 
                       energy_array: np.ndarray, 
                       resonant_energy: float, 
                       parallel: bool,
                       num_cores: int=None) -> np.ndarray:
    """
    Calculates the Stokes parameters for undulator radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        resonant_energy (float): Resonant energy [eV].
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data.
    """
    
    arPrecPar = [0]*5   # for spectral flux vs photon energy
    arPrecPar[0] = 1    # initial UR harmonic to take into account
    arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy_array[-1]) #final UR harmonic to take into account
    arPrecPar[2] = 1.5  # longitudinal integration precision parameter
    arPrecPar[3] = 1.5  # azimuthal integration precision parameter
    arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

    if parallel:
        if num_cores is None:
            num_cores = mp.cpu_count()

        energy_chunks = np.array_split(list(energy_array), num_cores)

        results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcStokesUR)(
                                                                    list_pairs,
                                                                    bl,
                                                                    eBeam,
                                                                    magFldCnt,
                                                                    resonant_energy,
                                                                    )
                                             for list_pairs in energy_chunks)
        energy_array = []
        time_array = []
        energy_chunks = []

        k = 0
        for calcs in results:
            energy_array.append(calcs[1][0])
            time_array.append(calcs[2])
            energy_chunks.append(len(calcs[0]))
            if k == 0:
                intensity = np.asarray(calcs[0], dtype="float64")
            else:
                intensity = np.concatenate((intensity, np.asarray(calcs[0], dtype="float64")), axis=0)
            k+=1
        print(">>> ellapse time:")

        for ptime in range(len(time_array)):
            print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")

    else:
        results = core_srwlibCalcStokesUR(energy_array,
                                          bl, 
                                          eBeam,
                                          magFldCnt, 
                                          resonant_energy)
        
        intensity = np.asarray(results[0], dtype="float64")

    return intensity


def core_srwlibCalcStokesUR(energy_array: np.ndarray, 
                            bl: dict, 
                            eBeam: srwlib.SRWLPartBeam, 
                            magFldCnt: srwlib.SRWLMagFldC, 
                            resonant_energy: float) -> Tuple[np.ndarray, float]:
    """
    Core function to calculate Stokes parameters for undulator radiation.

    Args:
        energy_array (np.ndarray): Array of photon energies [eV].
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        resonant_energy (float): Resonant energy [eV].
    Returns:
        Tuple[np.ndarray, float]: Tuple containing intensity data and computation time.
    """

    tzero = time.time()

    try:

        arPrecPar = [0]*5   # for spectral flux vs photon energy
        arPrecPar[0] = 1    # initial UR harmonic to take into account
        arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy_array[-1]) #final UR harmonic to take into account
        arPrecPar[2] = 1.5  # longitudinal integration precision parameter
        arPrecPar[3] = 1.5  # azimuthal integration precision parameter
        arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

        npts = len(energy_array)
        stk = srwlib.SRWLStokes() 
        stk.allocate(npts, 1, 1)     
        stk.mesh.zStart = bl['distance']
        stk.mesh.eStart = energy_array[0]
        stk.mesh.eFin =   energy_array[-1]
        stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
        stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
        stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
        stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2
        und = magFldCnt.arMagFld[0]
        srwlib.srwl.CalcStokesUR(stk, eBeam, und, arPrecPar)
        intensity = stk.arS[0:npts]
    except:
         raise ValueError("Error running SRW.")

    return intensity, energy_array, time.time()-tzero


def srwlibsrwl_wfr_emit_prop_multi_e(bl: dict,
                                     eBeam: srwlib.SRWLPartBeam, 
                                     magFldCnt: srwlib.SRWLMagFldC, 
                                     energy_array: np.ndarray,
                                     h_slit_points: int, 
                                     v_slit_points: int, 
                                     number_macro_electrons: int, 
                                     aux_file_name: str,
                                     parallel: bool,
                                     num_cores: int=None):
    """
    Interface function to compute multi-electron emission and propagation through a beamline using SRW.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        number_macro_electrons (int): Total number of macro-electrons.
        aux_file_name (str): Auxiliary file name for saving intermediate data.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.
    Returns:
        np.ndarray: Array containing intensity data.
    """
    nMacroElecAvgPerProc = 10   # number of macro-electrons / wavefront to average on worker processes
    nMacroElecSavePer = 100     # intermediate data saving periodicity (in macro-electrons)
    srCalcMeth = 1              # SR calculation method 
    srCalcPrec = 0.01           # SR calculation rel. accuracy

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    if num_cores is None:
        num_cores = mp.cpu_count()

    dE = np.diff(energy_array)    
    dE1 = np.min(dE)
    dE2 = np.max(dE)

    wiggler_regime = bool(energy_array[-1]>200*energy_array[0])

    if parallel:
        if np.allclose(dE1, dE2) and wiggler_regime:
            chunk_size = 20
            n_slices = len(energy_array)

            chunks = [(energy_array[i:i + chunk_size],
                        bl,
                        eBeam, 
                        magFldCnt, 
                        h_slit_points, 
                        v_slit_points, 
                        number_macro_electrons, 
                        aux_file_name+'_'+str(i),
                        srCalcMeth,
                        srCalcPrec,
                        nMacroElecAvgPerProc,
                        nMacroElecSavePer) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(core_srwlibsrwl_wfr_emit_prop_multi_e, chunks)
        else:
            dE = (energy_array[-1] - energy_array[0]) / num_cores
            energy_chunks = []

            for i in range(num_cores):
                bffr = copy.copy(energy_array)                
                bffr = np.delete(bffr, bffr < dE * (i) + energy_array[0])
                if i + 1 != num_cores:
                    bffr = np.delete(bffr, bffr >= dE * (i + 1) + energy_array[0])
                energy_chunks.append(bffr)

            results = Parallel(n_jobs=num_cores)(delayed(core_srwlibsrwl_wfr_emit_prop_multi_e)((
                                                                        list_pairs,
                                                                        bl,
                                                                        eBeam, 
                                                                        magFldCnt, 
                                                                        h_slit_points, 
                                                                        v_slit_points, 
                                                                        number_macro_electrons, 
                                                                        aux_file_name+'_'+str(list_pairs[0]),
                                                                        srCalcMeth,
                                                                        srCalcPrec,
                                                                        nMacroElecAvgPerProc,
                                                                        nMacroElecSavePer))
                                                for list_pairs in energy_chunks)

        for i, (intensity_chunck, e_chunck, t_chunck) in enumerate(results):
            if i == 0:
                intensity = intensity_chunck
                energy_chunck = np.asarray([e_chunck[0]])
                energy_chunks = np.asarray([len(e_chunck)])
                time_array = np.asarray([t_chunck])
            else:
                intensity = np.concatenate((intensity, intensity_chunck), axis=0)
                energy_chunck = np.concatenate((energy_chunck, np.asarray([e_chunck[0]])))
                energy_chunks = np.concatenate((energy_chunks, np.asarray([len(e_chunck)])))
                time_array = np.concatenate((time_array, np.asarray([t_chunck])))

        if not wiggler_regime:
            print(">>> ellapse time:")
            for ptime in range(len(time_array)):
                print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_chunck[ptime]:.1f} eV).")
    else:
        results = core_srwlibsrwl_wfr_emit_prop_multi_e((energy_array,
                                                        bl,
                                                        eBeam, 
                                                        magFldCnt, 
                                                        h_slit_points, 
                                                        v_slit_points, 
                                                        number_macro_electrons, 
                                                        aux_file_name,
                                                        srCalcMeth,
                                                        srCalcPrec,
                                                        nMacroElecAvgPerProc,
                                                        nMacroElecSavePer))
        intensity = np.asarray(results[0], dtype="float64")

    return intensity, hAxis, vAxis


def core_srwlibsrwl_wfr_emit_prop_multi_e(args: Tuple[np.ndarray, dict, srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, int, int, int, str, int, float, int, int]) -> Tuple[np.ndarray, float]:
    """
    Core function for computing multi-electron emission and propagation through a beamline using SRW.

    Args:
        args (tuple): Tuple containing arguments:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - number_macro_electrons (int): Total number of macro-electrons.
            - aux_file_name (str): Auxiliary file name for saving intermediate data.
            - srCalcMeth (int): SR calculation method.
            - srCalcPrec (float): SR calculation relative accuracy.
            - nMacroElecAvgPerProc (int): Number of macro-electrons / wavefront to average on worker processes.
            - nMacroElecSavePer (int): Intermediate data saving periodicity (in macro-electrons).

    Returns:
        tuple: A tuple containing intensity data array and the elapsed time.
    """

    energy_array, bl, eBeam, magFldCnt, h_slit_points, v_slit_points, \
        number_macro_electrons, aux_file_name, srCalcMeth, srCalcPrec, \
        nMacroElecAvgPerProc, nMacroElecSavePer = args
    
    tzero = time.time()

    try:    
        mesh = srwlib.SRWLRadMesh(energy_array[0], 
                            energy_array[-1], 
                            len(energy_array),
                            bl['slitHcenter'] - bl['slitH']/2,
                            bl['slitHcenter'] + bl['slitH']/2, 
                            h_slit_points,
                            bl['slitVcenter'] - bl['slitV']/2, 
                            bl['slitVcenter'] - bl['slitV']/2, 
                            v_slit_points,
                            bl['distance'])

        MacroElecFileName = aux_file_name + '_'+ str(int(number_macro_electrons / 1000)).replace('.', 'p') +'k_ME_intensity.dat'

        stk = srwlib.srwl_wfr_emit_prop_multi_e(_e_beam = eBeam, 
                                                _mag = magFldCnt,
                                                _mesh = mesh,
                                                _sr_meth = srCalcMeth,
                                                _sr_rel_prec = srCalcPrec,
                                                _n_part_tot = number_macro_electrons,
                                                _n_part_avg_proc=nMacroElecAvgPerProc, 
                                                _n_save_per=nMacroElecSavePer,
                                                _file_path=MacroElecFileName, 
                                                _sr_samp_fact=-1, 
                                                _opt_bl=None,
                                                _char=0)
    
        os.system('rm %s'% MacroElecFileName)
        me_intensity = np.asarray(stk.to_int(_pol=6), dtype='float64')

        if h_slit_points != 1 or v_slit_points != 1:
            data = np.zeros((len(energy_array), v_slit_points, h_slit_points))
            k = 0
            for iy in range(v_slit_points):
                for ix in range(h_slit_points):
                    for ie in range(len(energy_array)):
                        data[ie, iy, ix] = me_intensity[k]
                        k+=1
            me_intensity = data

    except:
         raise ValueError("Error running SRW.")

    return (me_intensity, energy_array, time.time()-tzero)

#***********************************************************************************
# read calculations
#***********************************************************************************

def read_spectrum(file_list: List[str], computer_code: str = 'xoppy') -> Dict[str, Any]:
    """
    Reads spectrum data from files and processes it using proc_spectrum function.

    This function reads spectrum data from files specified in the 'file_list' and processes
    it using the 'proc_spectrum' function to compute spectral power, cumulated power, and integrated power.

    Parameters:
        - file_list (List[str]): A list of file paths containing spectrum data.
        - computer_code (str): The code used to generate the spectrum data ('xoppy', 'srw', or 'spectra').

    Returns:
        Dict[str, Any]: A dictionary containing processed spectrum data with the following keys:
            - 'spectrum': A dictionary containing various properties of the spectrum including:
                - 'energy': Array containing energy values.
                - 'flux': Array containing spectral flux data.
                - 'spectral_power': Array containing computed spectral power.
                - 'cumulated_power': Cumulated power computed using cumulative trapezoid integration.
                - 'integrated_power': Integrated power computed using trapezoid integration.
    """
    energy = []
    flux = []

    if computer_code == 'xoppy' or computer_code == 'srw':
        for sim in file_list:
            f = open(sim, "rb")
            data = np.asarray(pickle.load(f))
            f.close()

            energy = np.concatenate((energy, data[0, :]))
            flux = np.concatenate((flux, data[1, :]))
    elif computer_code == 'spectra':
        for jsonfile in file_list:
            f = open(jsonfile)
            data = json.load(f)
            f.close()

            energy = np.concatenate((energy, data['Output']['data'][0]))
            flux = np.concatenate((flux, data['Output']['data'][1]))
    else:
        raise ValueError("Invalid computer code. Please specify either 'xoppy', 'srw', or 'spectra'.")

    return proc_spectrum(flux, energy)


def read_power_density(file_name: str, computer_code: str = 'xoppy') -> Dict[str, Any]:
    """
    Reads power density data from an XOPPY HDF5 file or SPECTRA JSON file and processes it.

    This function reads power density data from either an XOPPY HDF5 file or a SPECTRA JSON 
    file specified by 'file_name'. It extracts the power density map along with 
    corresponding x and y axes from the file, and then processes this data using the 
    'proc_power_density' function

    Parameters:
        - file_name (str): File path containing power density data.
        - computer_code (str): The code used to generate the power density data ('xoppy', 'srw', or 'spectra').

    Returns:
        Dict[str, Any]: A dictionary containing processed power density data with the following keys:
            - 'axis': A dictionary containing 'x' and 'y' axes arrays.
            - 'power_density': A dictionary containing power density-related data, including the power density map,
              total received power, and peak power density.
    """
    if computer_code == 'xoppy' or computer_code == 'srw':
        f = h5.File(file_name, "r")
        PowDenSR = f["XOPPY_POWERDENSITY"]["PowerDensity"]["image_data"][()]

        x = f["XOPPY_POWERDENSITY"]["PowerDensity"]["axis_x"][()]
        y = f["XOPPY_POWERDENSITY"]["PowerDensity"]["axis_y"][()]

    elif computer_code == 'spectra':
        f = open(file_name)
        data = json.load(f)
        f.close()

        PowDenSR = np.reshape(data['Output']['data'][2],
                            (len(data['Output']['data'][1]), 
                            len(data['Output']['data'][0])))

        if "mrad" in data['Output']['units'][2]:
            dist = data["Input"]["Configurations"]["Distance from the Source (m)"]
            dx = (data["Input"]["Configurations"]["x Range (mm)"][1]-data["Input"]["Configurations"]["x Range (mm)"][0])*1e-3
            dy = (data["Input"]["Configurations"]["y Range (mm)"][1]-data["Input"]["Configurations"]["y Range (mm)"][0])*1e-3

            dtx = 2*np.arctan(dx/dist/2)*1e3    # mrad
            dty = 2*np.arctan(dy/dist/2)*1e3

            PowDenSR *= 1e3 * (dtx*dty)/(dx*dy*1e3*1e3)
            x = data['Output']['data'][0]
            y = data['Output']['data'][1]
        else:
            PowDenSR *= 1e3
    else:
        raise ValueError("Invalid computer code. Please specify either 'xoppy', 'srw', or 'spectra'.")

    return proc_power_density(PowDenSR, x, y)


def read_emitted_radiation(file_list: List[str], computer_code: str = 'srw', parallel_processing: bool = False) -> dict:
    """
    Reads XOPPY undulator radiation data from a list of files and processes it.

    This function reads the XOPPY undulator radiation data from a list of HDF5 files,
    concatenates the spectral flux data, and processes it using either the proc_undulator_radiation function
    or the proc_undulator_radiation_parallel function based on the value of parallel_processing.

    Parameters:
        - file_list (List[str]): A list of file paths containing the XOPPY undulator radiation data.
        - computer_code (str): The code used to generate the power density data ('xoppy' or 'srw').
        - parallel_processing (bool, optional): Whether to use parallel processing. Defaults to False.

    Returns:
        dict: A dictionary containing processed undulator radiation data.

    Notes:
        - The input HDF5 files should contain the following datasets:
            - 'XOPPY_RADIATION/Radiation/stack_data': 3D array representing the spectral flux data.
            - 'XOPPY_RADIATION/Radiation/axis0': 1D array representing the energy axis.
            - 'XOPPY_RADIATION/Radiation/axis1': 1D array representing the x-axis.
            - 'XOPPY_RADIATION/Radiation/axis2': 1D array representing the y-axis.
        - The spectral flux data from different files will be concatenated along the 0-axis.
        - The x and y axes are assumed to be the same for all files in the file_list.
    """
    energy = []
    spectral_flux_3D = []

    k = 0

    for sim in file_list:
        print(sim)
        f = h5.File(sim, "r")

        if k == 0:
            spectral_flux_3D = f["XOPPY_RADIATION"]["Radiation"]["stack_data"][()]
            k+=1
        else:
            spectral_flux_3D = np.concatenate((spectral_flux_3D, f["XOPPY_RADIATION"]["Radiation"]["stack_data"][()]), 0)
        energy = np.concatenate((energy, f["XOPPY_RADIATION"]["Radiation"]["axis0"][()]))

    print("UR files loaded")
    if computer_code == 'xoppy':
        spectral_flux_3D = spectral_flux_3D.swapaxes(1, 2)

    x = f["XOPPY_RADIATION"]["Radiation"]["axis1"][()]
    y = f["XOPPY_RADIATION"]["Radiation"]["axis2"][()]

    if parallel_processing:
        return proc_spatial_spectral_dist_parallel(spectral_flux_3D, energy, x, y)
    else:
        return proc_spatial_spectral_dist(spectral_flux_3D, energy, x, y)

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


def generate_magnetic_measurement(und_per: float, B: float, num_und_per: int, 
                                  step_size: float, add_terminations: bool = True, 
                                  field_direction: str="v",
                                  und_per_disp: float = 0, B_disp: float = 0, 
                                  initial_phase_disp: float = 0, seed: int = 69, 
                                  num_samples: int = 1000, file_path: str=None, save_srw: bool=True) -> tuple:
    """
    Generate a magnetic measurement.

    This function generates a magnetic measurement with optional variations and noise.

    Args:
        und_per (float): Period of the undulator.
        B (float): Amplitude of the magnetic field.
        num_und_per (int): Number of undulator periods.
        step_size (float): Step size between samples.
        add_terminations (bool, optional): Whether to add magnetic terminations. Defaults to True.
        und_per_disp (float, optional): Standard deviation for random variation in undulator period. Defaults to 0.
        B_disp (float, optional): Standard deviation for random variation in magnetic field amplitude. Defaults to 0.
        initial_phase_disp (float, optional): Standard deviation for random variation in initial phase (in degrees). Defaults to 0.
        seed (int, optional): Seed for the random number generator. Defaults to 69.
        num_samples (int, optional): Number of samples for averaging. Defaults to 1000.
        file_path (str, optional): File path to save the generated magnetic field object. If None, the object won't be saved.
        save_srw (bool, optional): Whether to save the data in SRW format. Defaults to True.

    Returns:
        tuple: A tuple containing the magnetic field array and the axis array.
    """

    pad = 10
    nsteps = int((num_und_per+pad) * und_per / step_size)
    axis = np.linspace(-(num_und_per + pad) * und_per / 2, (num_und_per + pad) * und_per / 2, nsteps)
    
    if und_per_disp != 0 or B_disp != 0 or initial_phase_disp != 0:
        print("Adding phase errors")
        Bd = B + B_disp * np.sin(2 * np.pi * axis / (und_per * np.random.uniform(7.5, 12.5)) + np.random.normal(0, 1))
        magnetic_field = np.zeros((num_samples, len(axis)))
        dund_per = np.random.normal(loc=und_per, scale=und_per_disp, size=num_samples)
        dphase = np.random.normal(loc=0, scale=initial_phase_disp * np.pi / 180, size=num_samples)
        for i in range(num_samples):
            magnetic_field[i,:] = np.sin(2 * np.pi * axis / dund_per[i] + dphase[i])
        magnetic_field = Bd*np.mean(magnetic_field, axis=0)
    else:
        magnetic_field = B * np.sin(2 * np.pi * axis / und_per)

    if add_terminations:
        magnetic_field[axis < -(num_und_per) * und_per / 2] *= 3/4
        magnetic_field[axis > (num_und_per) * und_per / 2] *= 3/4
        magnetic_field[axis < -(num_und_per + 1) * und_per / 2] *= 1/3
        magnetic_field[axis > (num_und_per + 1) * und_per / 2] *= 1/3         
        magnetic_field[axis < -(num_und_per + 2) * und_per / 2] = 0 
        magnetic_field[axis > (num_und_per + 2) * und_per / 2] = 0  

    else:
        magnetic_field[axis < -(num_und_per) * und_per / 2] = 0
        magnetic_field[axis > (num_und_per) * und_per / 2] = 0

    if field_direction == "v":
        magnetic_field_vertical = magnetic_field
        magnetic_field_horizontal = magnetic_field*0
    elif field_direction == "h":
        magnetic_field_vertical = magnetic_field*0
        magnetic_field_horizontal = magnetic_field
    else:
        raise ValueError("Not valid field direction given.")
    
    if file_path is not None:
        if save_srw:
            magFldCnt = np.zeros([len(axis), 3])
            for i in range(len(axis)):
                magFldCnt[i,0] = axis[i]*1e3    # field is measured in mm on the benchtest
                magFldCnt[i,1] = magnetic_field_horizontal[i]
                magFldCnt[i,2] = magnetic_field_vertical[i]
            magFldCnt = generate_srw_magnetic_field(magFldCnt, file_path.replace(".txt", ".dat"))

        else:
            print(field_direction)
            print(np.amax(magnetic_field_vertical))
            print(np.amax(magnetic_field_horizontal))

            print(f">>> saving {file_path}")
            with open(file_path, 'w') as file:
                file.write("# Magnetic field data\n")
                file.write("# Axis_position   Horizontal_field   Vertical_field\n")
                for i in range(len(axis)):
                    file.write(f"{axis[i]}   {magnetic_field_horizontal[i]}   {magnetic_field_vertical[i]}\n")

    return magnetic_field, axis


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
# undulator auxiliary functions - magnetic field and K values
#***********************************************************************************

def get_K_from_B(B: float, und_per: float) -> float:
    """
    Calculate the undulator parameter K from the magnetic field B and the undulator period.

    Parameters:
    B (float): Magnetic field in Tesla.
    und_per (float): Undulator period in meters.

    Returns:
    float: The undulator parameter K.
    """
    K = CHARGE * B * und_per / (2 * PI * MASS * LIGHT)
    return K


def get_B_from_K(K: float, und_per: float) -> float:
    """
    Calculate the undulator magnetic field in Tesla from the undulator parameter K and the undulator period.

    Parameters:
    K (float): The undulator parameter K.
    und_per (float): Undulator period in meters.

    Returns:
    float: Magnetic field in Tesla.
    """
    B = K * 2 * PI * MASS * LIGHT/(CHARGE * und_per)
    return B

#***********************************************************************************
# undulator auxiliary functions - field-gap relationship
#***********************************************************************************

def fit_gap_field_relation(gap_table: List[float], B_table: List[float], 
                           und_per: float) -> Tuple[float, float, float]:
    """
    Fit parameters coeff0, coeff1, and coeff2 for an undulator from the given tables:

    B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

    Parameters:
        gap_table (List[float]): List of gap sizes in meters.
        B_table (List[float]): List of magnetic field values in Tesla corresponding to the gap sizes.
        und_per (float): Undulator period in meters.

    Returns:
        Tuple[float, float, float]: Fitted parameters (coeff0, coeff1, coeff2).
    """
    def _model(gp, c0, c1, c2):
        return c0 * np.exp(c1*gp + c2*gp**2)

    def _fit_parameters(gap, und_per, B):
        gp = gap / und_per
        popt, pcov = curve_fit(_model, gp, B, p0=(1, 1, 1)) 
        return popt

    popt = _fit_parameters(np.asarray(gap_table), und_per, np.asarray(B_table))
    coeff0_fit, coeff1_fit, coeff2_fit = popt

    print("Fitted parameters:")
    print("coeff0:", coeff0_fit)
    print("coeff1:", coeff1_fit)
    print("coeff2:", coeff2_fit)

    return coeff0_fit, coeff1_fit, coeff2_fit


def get_B_from_gap(gap: Union[float, np.ndarray], und_per: float, coeff: Tuple[float, float, float]) -> Union[float, np.ndarray, None]:
    """
    Calculate the magnetic field B from the given parameters:
       B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

    Parameters:
        gap (Union[float, np.ndarray]): Gap size(s) in meters.
        und_per (float): Undulator period in meters.
        coeff (Tuple[float, float, float]): Fit coefficients.

    Returns:
        Union[float, np.ndarray, None]: Calculated magnetic field B if gap and period are positive, otherwise None.
    """
    if isinstance(gap, np.ndarray):
        if np.any(gap <= 0) or und_per <= 0:
            return None
        gp = gap / und_per
    else:
        if gap <= 0 or und_per <= 0:
            return None
        gp = gap / und_per

    B = coeff[0] * np.exp(coeff[1] * gp + coeff[2] * gp**2)
    return B

#***********************************************************************************
# undulator auxiliary functions - undulator emission
#***********************************************************************************

def get_emission_energy(und_per: float, K: float, ring_e: float, n: int = 1, theta: float = 0) -> float:
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


def find_emission_harmonic_and_K(energy: float, und_per: float, ring_e: float, Kmin: float = 0.1, theta: float = 0) -> Tuple[int, float]:
    """
    Find the emission harmonic number and undulator parameter K for a given energy in a storage ring.

    Parameters:
        - energy (float): Energy of the emitted radiation in electron volts.
        - und_per (float): Undulator period in meters.
        - ring_e (float): Energy of electrons in GeV.
        - Kmin (float, optional): Minimum value of the undulator parameter (default is 0.1).
        - theta (float, optional): Observation angle in radians (default is 0).

    Returns:
        Tuple[int, float]: A tuple containing the emission harmonic number and the undulator parameter K.

    Raises:
        ValueError: If no valid harmonic is found.
    """
    count = 0
    harm = 0
    gamma = get_gamma(ring_e)
    wavelength = energy_wavelength(energy, 'eV')

    while harm == 0:
        n = 2 * count + 1
        K = np.sqrt(2 * ((2 * n * wavelength * gamma ** 2) / und_per - 1 - (gamma * theta) ** 2))
        if K >= Kmin:
            harm = int(n)
        count += 1
        # Break loop if no valid harmonic is found
        if count > 21:
            raise ValueError("No valid harmonic found.")

    return harm, K
        

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


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")
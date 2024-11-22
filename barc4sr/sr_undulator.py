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
__changed__ = '31/OCT/2024'

import multiprocessing as mp
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import numpy as np
import scipy.integrate as integrate
from joblib import Parallel, delayed
from numba import njit, prange
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from barc4sr.aux_processing import (
    write_emitted_radiation,
    write_power_density,
    write_spectrum,
    write_tuning_curve,
    write_wavefront,
)
from barc4sr.aux_utils import (
    barc4sr_dictionary,
    energy_wavelength,
    generate_logarithmic_energy_values,
    get_gamma,
    print_elapsed_time,
    set_light_source,
    set_magnetic_structure,
    srwlibCalcElecFieldSR,
    srwlibCalcStokesUR,
    srwlibsrwl_wfr_emit_prop_multi_e,
    syned_dictionary,
    unwrap_wft_phase,
    write_magnetic_field,
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
# undulator radiation
#***********************************************************************************
 
def spectrum(file_name: str,
             photon_energy_min: float,
             photon_energy_max: float,
             photon_energy_points: int, 
             **kwargs) -> Dict:
    """
    Calculate 1D undulator spectrum using SRW.

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
        wiggler_approximation (bool): To enable a simplified calculation for energy ranges several times over the 1st harmonic
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field
            0: uses the provided magnetic field, 
            1: fits the magnetic field using srwl.UtiUndFromMagFldTab. Default is 0.
        number_macro_electrons (int): Number of macro electrons. Default is 1.
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

    wiggler_approximation = kwargs.get('wiggler_approximation', False)

    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)

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

   
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, False, 'u',
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
                                                     id_type='w' if wiggler_approximation else 'u',
                                                     parallel=parallel,
                                                     num_cores=num_cores)
        flux = flux.reshape((photon_energy_points))
        print('completed')

    # -----------------------------------------
    # Flux through Finite Aperture 

    # simplified partially-coherent simulation
    if calculation == 1:
        calc_txt = calc_txt.replace("___CALC___", "simplified")
        if parallel:
            print(f'{calc_txt} in parallel... ')
        else:
            print(f'{calc_txt} ... ', end='')

        if wiggler_approximation:
            warnings.warn("Wiggler approximation not applicable for this calculation", UserWarning)
            
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
                                                                id_type='w' if wiggler_approximation else 'u',
                                                                number_macro_electrons=number_macro_electrons,
                                                                aux_file_name=file_name,
                                                                parallel=parallel,
                                                                num_cores=num_cores)       
        print('completed')

    spectral_power = flux*CHARGE*1E3
    cumulated_power = integrate.cumulative_trapezoid(spectral_power, energy, initial=0)
    integrated_power = integrate.trapezoid(spectral_power, energy)

    if file_name is not None:
        write_spectrum(file_name, flux, energy)

    print(f"{function_txt} finished.")

    print_elapsed_time(t0)

    spectrumSRdict = {
        "spectrum":{
            "energy":energy,
            "flux": flux,
            "spectral_power": spectral_power,
            "cumulated_power": cumulated_power,
            "integrated_power": integrated_power
        }
    }

    return spectrumSRdict


def power_density(file_name: str, 
                  hor_slit: float,
                  ver_slit: float,
                  **kwargs) -> Dict:
    """
    Calculate undulator power density spatial distribution using SRW.

    Args:
        file_name (str): The name of the output file.
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

        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field
            0: uses the provided magnetic field, 
            1: fits the magnetic field using srwl.UtiUndFromMagFldTab). Default is 0.

    Returns:
        Dict: A dictionary containing power density, horizontal axis, and vertical axis.
    """
    
    t0 = time.time()

    print("Undulator power density spatial distribution using SRW. Please wait...")

    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    observation_point = kwargs.get('observation_point', 10.)

    hor_slit_n = kwargs.get('hor_slit_n', 501)
    ver_slit_n= kwargs.get('ver_slit_n', 501)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    radiation_polarisation = kwargs.get('radiation_polarisation', 6)

    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
                            hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, magnetic_measurement, observation_point, 
                            hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, False, 'u',
                                               magnetic_measurement=magnetic_measurement,
                                               tabulated_undulator_mthd=tabulated_undulator_mthd)
    
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

    PowDen = np.reshape(stk.to_int(radiation_polarisation), (stk.mesh.ny, stk.mesh.nx))
    h_axis = np.linspace(-bl['slitH'] / 2, bl['slitH'] / 2, hor_slit_n)
    v_axis = np.linspace(-bl['slitV'] / 2, bl['slitV'] / 2, ver_slit_n)

    dh = (h_axis[1]-h_axis[0])*1E3
    dv = (v_axis[1]-v_axis[0])*1E3

    write_power_density(file_name, PowDen, h_axis, v_axis)

    print("Undulator power density spatial distribution using SRW: finished")
    print_elapsed_time(t0)

    PowDenSRdict = {
        "axis": {
            "x": h_axis,
            "y": v_axis,
            },
        "power_density": {
            "map":PowDen,
            "CumPow": PowDen.sum()*dh*dv,
            "PowDenSRmax": PowDen.max()
            }
        }
    
    return PowDenSRdict


def emitted_radiation(file_name: str, 
                      json_file: str, 
                      photon_energy_min: float,
                      photon_energy_max: float,
                      photon_energy_points: int, 
                      hor_slit: float, 
                      hor_slit_n: int,
                      ver_slit: float,
                      ver_slit_n: int,
                      **kwargs) -> Dict:
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
        radiation_polarisation (int): Polarisation component to be extracted. Default is 6.
            =0 -Linear Horizontal; 
            =1 -Linear Vertical; 
            =2 -Linear 45 degrees; 
            =3 -Linear 135 degrees; 
            =4 -Circular Right; 
            =5 -Circular Left; 
            =6 -Total
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
        Dict: undulator radiation spatial and spectral distribution, energy axis, horizontal axis, and vertical axis.
    """

    t0 = time.time()

    function_txt = "Undulator radiation spatial and spectral distribution using SRW:"
    calc_txt = "> Performing flux through finite aperture (___CALC___ partially-coherent simulation)"
    print(f"{function_txt} please wait...")

    energy_sampling = kwargs.get('energy_sampling', 0)

    observation_point = kwargs.get('observation_point', 10.)

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    radiation_polarisation = kwargs.get('radiation_polarisation', 6)

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
    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)
        
    # simplified partially-coherent simulation    
    if calculation == 0:
        calc_txt = calc_txt.replace("___CALC___", "simplified")
        if parallel:
            print(f'{calc_txt} in parallel... ')
        else:
            print(f'{calc_txt} ... ', end='')

        intensity, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
                                                          eBeam, 
                                                          magFldCnt,
                                                          energy,
                                                          h_slit_points=hor_slit_n,
                                                          v_slit_points=ver_slit_n,
                                                          radiation_characteristic=1, 
                                                          radiation_dependence=3,
                                                          radiation_polarisation=radiation_polarisation,
                                                          id_type='u',
                                                          parallel=parallel,
                                                          num_cores=num_cores)        
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
                                                                     energy,
                                                                     hor_slit_n,
                                                                     ver_slit_n,
                                                                     radiation_polarisation=radiation_polarisation,
                                                                     id_type='u',
                                                                     number_macro_electrons=number_macro_electrons,
                                                                     aux_file_name=file_name,
                                                                     parallel=parallel,
                                                                     num_cores=num_cores) 
        print('completed')
    
    write_emitted_radiation(file_name, intensity, energy, h_axis, v_axis)

    print("Undulator radiation spatial and spectral distribution using SRW: finished")
    print_elapsed_time(t0)

    return {'energy': energy, 'intensity':intensity, 'axis': {'x': h_axis, 'y': v_axis}}


def emitted_wavefront(file_name: str, 
                      json_file: str, 
                      photon_energy: float,
                      hor_slit: float, 
                      hor_slit_n: int,
                      ver_slit: float,
                      ver_slit_n: int,
                      **kwargs) -> Dict:
    """
    Calculate undulator emitted wavefront using SRW.

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
            Overrides SYNED undulatort data. Default is None.
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.
        number_macro_electrons (int): Number of macro electrons. Default is -1.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int): Number of CPU cores to use for parallel computation. If not specified, 
            it defaults to the number of available CPU cores.
    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """

    t0 = time.time()

    function_txt = "Undulator spatial distribution for a given energy using SRW:"
    calc_txt = "> Performing monochromatic wavefront calculation (___CALC___ partially-coherent simulation)"
    print(f"{function_txt} please wait...")

    observation_point = kwargs.get('observation_point', 10.)

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    radiation_polarisation = kwargs.get('radiation_polarisation', 6)

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

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
                          hor_slit, ver_slit, hor_slit_cen, ver_slit_cen, 
                          Kh=Kh, Kh_phase=Kh_phase, Kh_symmetry=Kh_symmetry, 
                          Kv=Kv, Kv_phase=Kv_phase, Kv_symmetry=Kv_symmetry)

   
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, electron_trajectory, 'u',
                                               magnetic_measurement=magnetic_measurement,
                                               tabulated_undulator_mthd=tabulated_undulator_mthd)
    
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
                                                          id_type = 'u',
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
                                                      id_type = 'u',
                                                      parallel=False)     

        phase = unwrap_wft_phase(phase, h_axis, v_axis, observation_point, photon_energy)
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
                                                                     id_type = 'u',
                                                                     number_macro_electrons=number_macro_electrons,
                                                                     aux_file_name=file_name,
                                                                     parallel=False,
                                                                     num_cores=num_cores,
                                                                     srApprox=1) 
        
        phase = np.zeros(intensity.shape)
        print('completed')
    
    write_wavefront(file_name, intensity, phase, h_axis, v_axis)

    print(f"{function_txt} finished.")
    print_elapsed_time(t0)

    return {"photon_energy":photon_energy, "wavefront": {"phase":phase, "intensity":intensity}, "axis":{"x":h_axis, "y":v_axis} }


def coherent_modes():
    pass

def tuning_curve(file_name: str,
             json_file: str,
             photon_energy_min: float,
             photon_energy_max: float,
             photon_energy_points: int, 
             nHarmMax: int,
             **kwargs) -> Dict:
    """
    Calculate undulator tuning curve using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.
        nHarmMax (int): number of the highest harmonic to be takent into account.
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
        Kmin (float): Minimum K-value (cutt off for highest harmonic energy). Default is -1.
        Kmax (float): Maximum K-value (cutt off for lowest harmonic energy). Default is -1.
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
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()

    function_txt = "Undulator K-tuning curve calculation using SRW:"
    calc_txt = "> Performing flux through finite aperture (___CALC___ partially-coherent simulation)"
    print(f"{function_txt} please wait...")

    energy_sampling = kwargs.get('energy_sampling', 0)

    observation_point = kwargs.get('observation_point', 10.)

    even_harmonics = kwargs.get('even_harmonics', False)

    hor_slit = kwargs.get('hor_slit', 1e-23)
    ver_slit = kwargs.get('ver_slit', 1e-23)
    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    radiation_polarisation = kwargs.get('radiation_polarisation', 6)

    Kmin = kwargs.get('Kmin', 1E-3)
    Kmax = kwargs.get('Kmax', None)

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

    if parallel is True and num_cores is None:
        num_cores = mp.cpu_count()

    chunk_size = 20

    if hor_slit < 1e-6 and ver_slit < 1e-6:
        calculation = 0
        hor_slit = 0
        ver_slit = 0
    else:
        if magnetic_measurement is None and number_macro_electrons == 1:
            calculation = 1
        if magnetic_measurement is not None or number_macro_electrons > 1:
            calculation = 3

    bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
                          hor_slit, ver_slit, hor_slit_cen, ver_slit_cen, 
                          Kh=Kh, Kh_phase=Kh_phase, Kh_symmetry=Kh_symmetry, 
                          Kv=Kv, Kv_phase=Kv_phase, Kv_symmetry=Kv_symmetry)
   
    eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, filament_beam, 
                                               energy_spread, electron_trajectory, 'u',
                                               magnetic_measurement=magnetic_measurement,
                                               tabulated_undulator_mthd=tabulated_undulator_mthd)

    # ----------------------------------------------------------------------------------
    # tuning curve calculations
    # ----------------------------------------------------------------------------------

    if nHarmMax > np.floor(photon_energy_max/photon_energy_min):
        nHarmMax = int(np.floor(photon_energy_max/photon_energy_min))

    if energy_sampling == 0: 
        energy = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        if np.sqrt(bl['Kv']**2 + bl['Kh']**2) == 0:
            resonant_energy = photon_energy_min
        else:
            resonant_energy = get_emission_energy(bl['PeriodID'], 
                                                np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                                bl['ElectronEnergy'])

        stepsize = np.log(photon_energy_max/resonant_energy)
        energy = generate_logarithmic_energy_values(photon_energy_min,
                                                    photon_energy_max,
                                                    resonant_energy,
                                                    stepsize)

    tc = np.zeros((len(energy), nHarmMax+1))       
    K = np.zeros((len(energy), nHarmMax))

    for nharm in range(nHarmMax):
        if (nharm + 1) % 2 != 0:
            for i, eng in enumerate(energy):
                harm, deflection_parameter = find_emission_harmonic_and_K(eng, bl['PeriodID'], bl['ElectronEnergy'], Kmin, 0, nharm+1)         
                if harm == nharm+1:
                    K[i, nharm] = deflection_parameter

    if Kmax is None:
        Kmax = K[0, 0]
    
    K[K>Kmax] = 0
    K[K<Kmin] = 0

    # ---------------------------------------------------------
    # On-Axis Tuning curve from Filament Electron Beam
    if calculation == 0:    # TODO: implement analytical equation
        if parallel:
            print('> Performing on-axis tuning curve from filament electron beam in parallel ... ')
            n_slices = len(energy)
            chunks = [(energy[i:i + chunk_size],
                        K[i:i + chunk_size, :],
                        nHarmMax, 
                        even_harmonics,
                        bl,
                        eBeam,
                        magnetic_measurement,
                        tabulated_undulator_mthd,
                        radiation_polarisation
                    ) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(tc_on_axis_srwlibCalcElecFieldSR, chunks)
            tc = np.concatenate(results, axis=0)
            
        else:
            print('> Performing on-axis tuning curve from filament electron beam ... ', end='')
            for nharm in range(nHarmMax):
                if even_harmonics or (nharm + 1) % 2 != 0:
                    for i, dE in enumerate(energy):
                        deflec_param = K[i, nharm]
                        if deflec_param>0:
                            bl['Kv'] = deflec_param

                            magFldCnt = set_magnetic_structure(bl, id_type='u',
                                                    magnetic_measurement = magnetic_measurement, 
                                                    tabulated_undulator_mthd = tabulated_undulator_mthd)
                            tc[i, nharm+1], h_axis, v_axis = srwlibCalcElecFieldSR(
                                                                bl, 
                                                                eBeam, 
                                                                magFldCnt,
                                                                dE,
                                                                h_slit_points=1,
                                                                v_slit_points=1,
                                                                radiation_characteristic=0, 
                                                                radiation_dependence=0,
                                                                radiation_polarisation=radiation_polarisation,
                                                                id_type='u',
                                                                parallel=False,
                                                                num_cores=1
                                                                )

    # -----------------------------------------
    # Flux through Finite Aperture 

    # simplified partially-coherent simulation
    if calculation == 1:
        calc_txt = calc_txt.replace("___CALC___", "simplified / CalcElecFieldSR")
        if parallel:
            print(f'{calc_txt} in parallel... ')
            n_slices = len(energy)
            chunks = [(energy[i:i + chunk_size],
                        K[i:i + chunk_size, :],
                        nHarmMax, 
                        even_harmonics,
                        bl,
                        eBeam,
                        magnetic_measurement,
                        tabulated_undulator_mthd,
                        radiation_polarisation
                    ) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(tc_through_slit_srwlibCalcElecFieldSR, chunks)
            tc = np.concatenate(results, axis=0)
        else:
            print(f'{calc_txt} ... ', end='')
            for nharm in range(nHarmMax):
                if even_harmonics or (nharm + 1) % 2 != 0:
                    for i, dE in enumerate(energy):
                        deflec_param = K[i, nharm]
                        if deflec_param>0:
                            bl['Kv'] = deflec_param

                            magFldCnt = set_magnetic_structure(bl, id_type='u',
                                                    magnetic_measurement = magnetic_measurement, 
                                                    tabulated_undulator_mthd = tabulated_undulator_mthd)
                            flux, h_axis, v_axis = srwlibCalcElecFieldSR(
                                                        bl, 
                                                        eBeam, 
                                                        magFldCnt,
                                                        dE,
                                                        h_slit_points=101,
                                                        v_slit_points=101,
                                                        radiation_characteristic=1, 
                                                        radiation_dependence=3,
                                                        radiation_polarisation=radiation_polarisation,
                                                        id_type='u',
                                                        parallel=parallel,
                                                        num_cores=num_cores
                                                        )
                            tc[i, nharm+1] = (np.sum(flux)*(h_axis[1]-h_axis[0])*(v_axis[1]-v_axis[0]))*1E6           
        print('completed')

    if calculation == 2:
        warnings.warn("Very slow calculation - consider setting the number of macro electrons to 1.", UserWarning)

        calc_txt = calc_txt.replace("___CALC___", "accurate")
        if parallel:
            print(f'{calc_txt} in parallel... ')
            n_slices = len(energy)
            chunks = [(energy[i:i + chunk_size],
                        K[i:i + chunk_size, :],
                        nHarmMax, 
                        even_harmonics,
                        bl,
                        eBeam,
                        magnetic_measurement,
                        tabulated_undulator_mthd,
                        radiation_polarisation,
                        number_macro_electrons, 
                        file_name
                    ) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(tc_through_slit_srwlibsrwl_wfr_emit_prop_multi_e, chunks)
            tc = np.concatenate(results, axis=0)
        else:
            print(f'{calc_txt} ... ', end='')


        for nharm in range(nHarmMax):
            if even_harmonics or (nharm + 1) % 2 != 0:
                for i, dE in enumerate(energy):
                    deflec_param = K[i, nharm]
                    if deflec_param>0:
                        bl['Kv'] = deflec_param
                        magFldCnt = set_magnetic_structure(bl, id_type='u',
                                                magnetic_measurement = magnetic_measurement, 
                                                tabulated_undulator_mthd = tabulated_undulator_mthd)
                        
                        tc[i, nharm+1], h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(
                                                                bl, 
                                                                eBeam,
                                                                magFldCnt,
                                                                dE,
                                                                h_slit_points=1,
                                                                v_slit_points=1,
                                                                radiation_polarisation=radiation_polarisation,
                                                                id_type='u',
                                                                number_macro_electrons=number_macro_electrons,
                                                                aux_file_name=file_name,
                                                                parallel=parallel,
                                                                num_cores=num_cores
                                                                )  

    tc[:,0] = np.max(tc,axis=1)
    K = np.insert(K, 0, 0, axis=1)

    write_tuning_curve(file_name, tc, K, energy)

    print(f"{function_txt} finished.")
    print_elapsed_time(t0)

    return {'energy':energy, 'flux':tc, 'K': K}


def tc_on_axis_srwlibCalcElecFieldSR(args):
    energy, K, nHarmMax, even_harmonics, bl, eBeam, magnetic_measurement, \
        tabulated_undulator_mthd, radiation_polarisation = args

    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = K[i, nharm]
                if deflec_param>0:
                    bl['Kv'] = deflec_param

                    magFldCnt = set_magnetic_structure(bl, id_type='u',
                                            magnetic_measurement = magnetic_measurement, 
                                            tabulated_undulator_mthd = tabulated_undulator_mthd)
                    htc[i, nharm+1], h_axis, v_axis = srwlibCalcElecFieldSR(
                                                bl, 
                                                eBeam, 
                                                magFldCnt,
                                                dE,
                                                h_slit_points=1,
                                                v_slit_points=1,
                                                radiation_characteristic=0, 
                                                radiation_dependence=0,
                                                radiation_polarisation=radiation_polarisation,
                                                id_type='u',
                                                parallel=False,
                                                num_cores=1
                                                )
    return htc


def tc_through_slit_srwlibCalcElecFieldSR(args):

    energy, K, nHarmMax, even_harmonics, bl, eBeam, magnetic_measurement, \
        tabulated_undulator_mthd, radiation_polarisation = args

    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = K[i, nharm]
                if deflec_param>0:
                    bl['Kv'] = deflec_param

                    magFldCnt = set_magnetic_structure(bl, id_type='u',
                                            magnetic_measurement = magnetic_measurement, 
                                            tabulated_undulator_mthd = tabulated_undulator_mthd)
                    flux, h_axis, v_axis = srwlibCalcElecFieldSR(
                                                bl, 
                                                eBeam, 
                                                magFldCnt,
                                                dE,
                                                h_slit_points=101,
                                                v_slit_points=101,
                                                radiation_characteristic=1, 
                                                radiation_dependence=3,
                                                radiation_polarisation=radiation_polarisation,
                                                id_type='u',
                                                parallel=False,
                                                num_cores=1
                                                )
                    htc[i, nharm+1] = (np.sum(flux)*(h_axis[1]-h_axis[0])*(v_axis[1]-v_axis[0]))*1E6    
    return htc


def tc_through_slit_srwlibsrwl_wfr_emit_prop_multi_e(args):

    energy, K, nHarmMax, even_harmonics, bl, eBeam, magnetic_measurement, \
        tabulated_undulator_mthd, radiation_polarisation, number_macro_electrons,\
             file_name = args

    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = K[i, nharm]
                if deflec_param>0:
                    bl['Kv'] = deflec_param
                    magFldCnt = set_magnetic_structure(bl, id_type='u',
                                            magnetic_measurement = magnetic_measurement, 
                                            tabulated_undulator_mthd = tabulated_undulator_mthd)
                    
                    htc[i, nharm+1], h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(
                                                            bl, 
                                                            eBeam,
                                                            magFldCnt,
                                                            dE,
                                                            h_slit_points=1,
                                                            v_slit_points=1,
                                                            radiation_polarisation=radiation_polarisation,
                                                            id_type='u',
                                                            number_macro_electrons=number_macro_electrons,
                                                            aux_file_name=file_name,
                                                            parallel=False,
                                                            num_cores=1
                                                            )  

#***********************************************************************************
# Magnetic field functions
#***********************************************************************************

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
        Bd = B + B_disp * np.sin(2 * PI * axis / (und_per * np.random.uniform(7.5, 12.5)) + np.random.normal(0, 1))
        magnetic_field = np.zeros((num_samples, len(axis)))
        dund_per = np.random.normal(loc=und_per, scale=und_per_disp, size=num_samples)
        dphase = np.random.normal(loc=0, scale=initial_phase_disp * PI / 180, size=num_samples)
        for i in range(num_samples):
            magnetic_field[i,:] = np.sin(2 * PI * axis / dund_per[i] + dphase[i])
        magnetic_field = Bd*np.mean(magnetic_field, axis=0)
    else:
        magnetic_field = B * np.sin(2 * PI * axis / und_per)

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
            magFldCnt = write_magnetic_field(magFldCnt, file_path.replace(".txt", ".dat"))

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


def find_emission_harmonic_and_K(energy: float, und_per: float, ring_e: float, Kmin: float = 0.1, theta: float = 0,
                                 starting_harmonic: int = 1) -> Tuple[int, float]:
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
    count = np.floor((starting_harmonic-1)/2)
    harm = 0
    gamma = get_gamma(ring_e)
    wavelength = energy_wavelength(energy, 'eV')

    while harm == 0:
        n = 2 * count + 1
        try:
            arg_sqrt = 2 * ((2 * n * wavelength * gamma ** 2) / und_per - 1 - (gamma * theta) ** 2)
            if arg_sqrt>=0:
                K = np.sqrt(arg_sqrt)
            else:
                K=-1
            if K >= Kmin:
                harm = int(n)
        except ValueError:
            K = None

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
    Calculate the total power emitted by a planar undulator in watts (W) based on Eq. 56 
    from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
    [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

    :param ring_e: Ring energy in gigaelectronvolts (GeV).
    :param ring_curr: Ring current in amperes (A).
    :param und_per: Undulator period in meters (m).
    :param und_n_per: Number of periods.
    :param B: Magnetic field in tesla (T). If not provided, it will be calculated based on K.
    :param K: Deflection parameter. Required if B is not provided.
    :param verbose: Whether to print results. Defaults to False.
    
    :return: Total power emitted by the undulator in watts (W).
    """

    if B is None:
        if K is None:
            raise TypeError("Please, provide either B or K for the undulator")
        else:
            B = get_B_from_K(K, und_per)
    else:
        K  = get_K_from_B(B, und_per)

    gamma = get_gamma(ring_e)
    tot_pow = (und_n_per*Z0*ring_curr*CHARGE*2*PI*LIGHT*gamma**2*K**2)/(6*und_per)*1E-3
    if verbose:
        print(f"Total power emitted by the undulator: {tot_pow:.3f} kW")

    return tot_pow

def power_through_slit(hor_slit: float, ver_slit: float, observation_point: float, ring_e: float, ring_curr: float, und_per: float, und_n_per: int,
                       B: Optional[float] = None, K: Optional[float] = None,
                       verbose: bool = False, **kwargs) -> float:
    """ 
    Calculate the power passing through a slit in watts (W), based on Eq. 50 
    from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
    [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995).

    The integration is performed only over one quadrant (positive phi and psi) for 
    computational efficiency, and the result is scaled by 4, leveraging symmetry 
    about zero in both directions.

    :param hor_slit (float): Horizontal slit size [m].
    :param ver_slit (float): Vertical slit size [m].
    :patam observation_point (float): Distance from source to slits [m].
    :param ring_e: Ring energy in gigaelectronvolts (GeV).
    :param ring_curr: Ring current in amperes (A).
    :param und_per: Undulator period in meters (m).
    :param und_n_per: Number of periods.
    :param B: Magnetic field in tesla (T). If not provided, it will be calculated based on K.
    :param K: Deflection parameter. Required if B is not provided.
    :param verbose: Whether to print results. Defaults to False.
    :param kwargs: Additional keyword arguments:
                   - npix: Number of pixels in the grid for integration. Defaults to 501.

    :return: Power passing through the slit in watts (W).
    """

    npix = kwargs.get('npix', 501)
    gamma = get_gamma(ring_e)

    if B is None:
        if K is None:
            raise TypeError("Please, provide either B or K for the undulator")
        else:
            B = get_B_from_K(K, und_per)
    else:
        K  = get_K_from_B(B, und_per)

    hor_slit_in_rad = np.arctan(hor_slit/2/observation_point)*2
    ver_slit_in_rad = np.arctan(ver_slit/2/observation_point)*2

    # positive quadrant
    dphi = np.arctan(np.linspace(0, hor_slit/2, npix)/observation_point)
    dpsi = np.arctan(np.linspace(0, ver_slit/2, npix)/observation_point)

    gk = G_K(K)

    d2P_d2phi_0 = total_power(ring_e, ring_curr, und_per, und_n_per, K=K, 
                              verbose=False)*1E3*(gk*21*gamma**2)/(16*PI*K)

    pos_quadrant = compute_f_K_numba(dphi, dpsi, K, gamma, gk)

    full_quadrant = np.block([
        [np.flip(np.flip(pos_quadrant[1:, 1:], axis=0), axis=1),
         np.flip(pos_quadrant[1:, :], axis=0)],
        [np.flip(pos_quadrant[:, 1:], axis=1), 
          pos_quadrant]])

    dx = np.linspace(-hor_slit / 2, hor_slit / 2, full_quadrant.shape[1])
    dy = np.linspace(-ver_slit / 2, ver_slit / 2, full_quadrant.shape[0])

    dx_step = (dx[1]-dx[0])*1E3
    dy_step = (dy[1]-dy[0])*1E3

    dphi = np.arctan(dx/observation_point)
    dpsi = np.arctan(dy/observation_point)

    dphi_step = np.diff(dphi, prepend=dphi[0])  
    dpsi_step = np.diff(dpsi, prepend=dpsi[0])  

    weights = np.outer(dphi_step, dpsi_step)

    d2P_d2phi = d2P_d2phi_0*full_quadrant*weights/dx_step/dy_step

    CumPow = d2P_d2phi.sum()*dx_step*dy_step

    if verbose:
        print(f"Power emitted by the undulator throgh a {hor_slit_in_rad*1E3} x {ver_slit_in_rad*1E3} mrad² slit: {CumPow*1E-3:.3f} kW")

    PowDenSRdict = {
        "axis": {
            "x": dx,
            "y": dy,
            },
        "power_density": {
            "map":d2P_d2phi,
            "CumPow": CumPow,
            "PowDenSRmax": d2P_d2phi.max()
            }
        }
    
    return PowDenSRdict

def G_K(K):
    """ Angular function f_k, based on Eq. 52 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    numerator = K*(K**6 + 24/7 * K**4 + 4*K**2 + 16/7)
    denominator = (1+K**2)**(7/2)

    return numerator/denominator

@njit
def f_K_numba(phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    n_points = int(2*np.pi*1001)  # Number of integration points
    return integrate_trapezoidal_numba(f_K_integrand_numba, -np.pi, np.pi, n_points, phi, psi, K, gamma)

@njit
def f_K_integrand_numba(csi, phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    X = gamma * psi
    Y = K * np.cos(csi) - gamma * phi
    D = 1 + X**2 + Y**2
    return (np.sin(csi)**2) * ((1 + X**2 - Y**2)**2 + 4 * (X * Y)**2) / (D**5)

@njit(parallel=True)
def compute_f_K_numba(dphi, dpsi, K, gamma, gk):
    npix = len(dphi)
    angular_function_f_K = np.zeros((npix, npix))
    for j in prange(npix):
        for i in range(npix):
            angular_function_f_K[i, j] = f_K_numba(dphi[j], dpsi[i], K, gamma)
    return angular_function_f_K*16*K/(7*np.pi*gk)

@njit
def integrate_trapezoidal_numba(func, a, b, n, *args):
    """Custom trapezoidal integration."""
    h = (b - a) / n
    s = 0.5 * (func(a, *args) + func(b, *args))
    for i in range(1, n):
        s += func(a + i * h, *args)
    return s * h

def f_K_joblib(phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """

    result, error = integrate.quad(f_K_integrand_joblib, -np.pi, np.pi, args=(phi, psi, K, gamma))
    return result

def f_K_integrand_joblib(csi, phi, psi, K, gamma):
    """ Angular function f_k, based on Eq. 53 from K. J. Kim, "Optical and power 
    characteristics of synchrotron radiation sources" [also Erratum 34(4)1243(Apr1995)],
    Opt. Eng 34(2), 342 (1995).
    """
    X = gamma * psi
    Y = K * np.cos(csi) - gamma * phi
    D = 1 + X**2 + Y**2
    return (np.sin(csi)**2) * ((1 + X**2 - Y**2)**2 + 4 * (X * Y)**2) / (D**5)

def compute_f_K_joblib(dphi, dpsi, K, gamma, n_jobs=10):

    npix = len(dphi)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_value)(i, j, phi, psi, K, gamma)
        for j, phi in enumerate(dphi)
        for i, psi in enumerate(dpsi)
    )

    angular_function_f_K = np.zeros((npix, npix))
    for i, j, value in results:
        angular_function_f_K[i, j] = value

    return angular_function_f_K*16*K/(7*PI*G_K(K))

def compute_value(i, j, phi, psi, K, gamma):
    return i, j, f_K_joblib(phi, psi, K, gamma)



if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")



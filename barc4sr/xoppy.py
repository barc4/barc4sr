"""
This module provides functions for interfacing with SRW when calculating undulator 
radiation, power density, and spectra. This is based on the xoppy.sources.srundplug 
module from https://github.com/oasys-kit/xoppylib
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MARCH/2024'
__changed__ = '28/MARCH/2024'

import array
import copy
import multiprocessing as mp
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants

from barc4sr.undulator import get_B_from_K, get_emission_energy
from barc4sr.utils import (
    generate_logarithmic_energy_values,
    get_gamma,
    print_elapsed_time,
    read_syned_file,
)

try:
    import srwpy.srwlib as srwlib

    USE_SRWLIB = True
    print('SRW distribution of SRW')
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
    print('OASYS distribution of SRW')
if USE_SRWLIB is False:
    print("SRW is not available")

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]
PI = np.pi

#***********************************************************************************
# xoppy_undulators.py inspired modules for undulator radiation
#***********************************************************************************
 
def undulator_spectrum(file_name: str,
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
    number_macro_electrons = kwargs.get('number_macro_electrons', 1000)
    parallel = kwargs.get('parallel', False)
    num_cores = kwargs.get('num_cores', None)

    if hor_slit < 1e-6 and ver_slit < 1e-6:
        calculation = 0
        hor_slit = 0
        ver_slit = 0
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
                                                     parallel=parallel)
        print('completed')
    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)

    # simplified partially-coherent simulation
    if calculation == 1 and magnetic_measurement is None:
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
    if calculation == 1 and magnetic_measurement is not None:
        print('> Performing flux through finite aperture (accurate partially-coherent simulation)... ', end='')
        flux, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                eBeam,
                                                                magFldCnt,
                                                                energy,
                                                                1,
                                                                1,
                                                                number_macro_electrons,
                                                                file_name)       
        print('completed')

    file = open('%s_spectrum.pickle'%file_name, 'wb')
    pickle.dump([energy, flux], file)
    file.close()

    print("Undulator spectrum calculation using SRW: finished")
    print_elapsed_time(t0)

    return energy, flux


def undulator_power_density(file_name: str, 
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


def undulator_radiation(file_name: str, 
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
    Calculate undulator power density spatial distribution using SRW.

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
        print('> Performing flux through finite aperture (accurate partially-coherent simulation)... ', end='')
        flux, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
                                                                eBeam,
                                                                magFldCnt,
                                                                energy,
                                                                hor_slit_n,
                                                                ver_slit_n,
                                                                number_macro_electrons,
                                                                file_name) 
        print('completed')
    
    with h5.File('%s_undulator_radiation.h5'%file_name, 'w') as f:
        group = f.create_group('XOPPY_RADIATION')
        radiation_group = group.create_group('Radiation')
        radiation_group.create_dataset('stack_data', data=intensity)
        radiation_group.create_dataset('axis0', data=energy)
        radiation_group.create_dataset('axis1', data=h_axis)
        radiation_group.create_dataset('axis2', data=v_axis)

    print("Undulator radiation spatial and spectral distribution using SRW: finished")
    print_elapsed_time(t0)

    return energy, intensity, h_axis, v_axis

#***********************************************************************************
# SRW interfaced functions
#***********************************************************************************

def set_light_source(file_name: str,
                     bl: dict,
                     filament_beam: bool,
                     energy_spread: bool,
                     magnetic_measurement: Optional[str],
                     tabulated_undulator_mthd: int,
                     electron_trajectory: bool) -> Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]:
    """
    Set up the light source parameters including electron beam, magnetic structure, and electron trajectory.

    Args:
        file_name (str): The name of the output file.
        bl (dict): Beamline parameters dictionary containing essential information for setup.
        filament_beam (bool): Flag indicating whether to set the beam emittance to zero.
        energy_spread (bool): Flag indicating whether to set the beam energy spread to zero.
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field.
        electron_trajectory (bool): Whether to calculate and save electron trajectory.

    Returns:
        Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]: A tuple containing the electron beam,
        magnetic structure, and electron trajectory.
    """    
    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    print('> Generating the electron beam ... ', end='')
    eBeam = set_electron_beam(bl,
                              filament_beam,
                              energy_spread)
    print('completed')
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    print('> Generating the magnetic structure ... ', end='')
    magFldCnt = set_magnetic_structure(bl,
                                       magnetic_measurement, 
                                       tabulated_undulator_mthd)
    print('completed')
    # ----------------------------------------------------------------------------------
    # calculate electron trajectory
    # ----------------------------------------------------------------------------------
    print('> Electron trajectory calculation ... ', end='')
    if electron_trajectory:
        print('>> srwlCalcPartTraj ... ', end='')
        electron_trajectory_file_name = file_name+"_eTraj.dat"
        eTraj = srwlCalcPartTraj(eBeam, magFldCnt)
        eTraj.save_ascii(electron_trajectory_file_name)
        print(f">>>{electron_trajectory_file_name}<<< ", end='')
    else:
        eTraj = 0
    print('completed')

    return eBeam, magFldCnt, eTraj


def set_electron_beam(bl: dict, filament_beam: bool, energy_spread: bool) -> srwlib.SRWLPartBeam:
    """
    Set up the electron beam parameters.

    Parameters:
        bl (dict): Dictionary containing beamline parameters.
        filament_beam (bool): Flag indicating whether to set the beam emittance to zero.
        energy_spread (bool): Flag indicating whether to set the beam energy spread to zero.

    Returns:
        srwlib.SRWLPartBeam: Electron beam object initialized with specified parameters.

    """

    eBeam = srwlib.SRWLPartBeam()
    eBeam.Iavg = bl['ElectronCurrent']  # average current [A]
    eBeam.partStatMom1.x = 0.  # initial transverse positions [m]
    eBeam.partStatMom1.y = 0.
    eBeam.partStatMom1.z = - bl['PeriodID'] * (bl['NPeriods'] + 4) / 2  # initial longitudinal positions
    eBeam.partStatMom1.xp = 0  # initial relative transverse velocities
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = get_gamma(bl['ElectronEnergy'])

    if filament_beam:
        sigX = 1e-25
        sigXp = 1e-25
        sigY = 1e-25
        sigYp = 1e-25
        if energy_spread:
            sigEperE = bl['ElectronEnergySpread']
        else:
            sigEperE = 1e-25    
    else:
        sigX = bl['ElectronBeamSizeH']  # horizontal RMS size of e-beam [m]
        sigXp = bl['ElectronBeamDivergenceH']  # horizontal RMS angular divergence [rad]
        sigY = bl['ElectronBeamSizeV']  # vertical RMS size of e-beam [m]
        sigYp = bl['ElectronBeamDivergenceV']  # vertical RMS angular divergence [rad]
        if energy_spread:
            sigEperE = bl['ElectronEnergySpread']
        else:
            sigEperE = 1e-25    

    # 2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX * sigX  # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0  # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp * sigXp  # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY * sigY  # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0  # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp * sigYp  # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE * sigEperE  # <(E-<E>)^2>/<E>^2

    return eBeam


def set_magnetic_structure(bl: dict, magnetic_measurement: Union[str, None], 
                           tabulated_undulator_mthd: int=0) -> srwlib.SRWLMagFldC:
    """
    Sets up the magnetic field container.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        magnetic_measurement (Union[str, None]): Path to the tabulated magnetic field data or None for ideal sinusoidal undulator.
        tabulated_undulator_mthd (int, optional): Method to use for generating undulator field if magnetic_measurement is provided. Defaults to 0

    Returns:
        srwlib.SRWLMagFldC: Magnetic field container.

    """
    if magnetic_measurement is None:    # ideal sinusoidal undulator magnetic structure

        und = srwlib.SRWLMagFldU()
        und.set_sin(_per=bl["PeriodID"],
                    _len=bl['PeriodID']*bl['NPeriods'], 
                    _bx=get_B_from_K(bl['Kh'],bl["PeriodID"]), 
                    _by=get_B_from_K(bl['Kv'],bl["PeriodID"]), 
                    _phx=bl['KhPhase'], 
                    _phy=bl['KvPhase'], 
                    _sx=bl['MagFieldSymmetryH'], 
                    _sy=bl['MagFieldSymmetryV'])

        magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[und],
                                        _arXc=srwlib.array('d', [0.0]),
                                        _arYc=srwlib.array('d', [0.0]),
                                        _arZc=srwlib.array('d', [0.0]))
        
    else:    # tabulated magnetic field
        magFldCnt = srwlib.srwl_uti_read_mag_fld_3d(magnetic_measurement, _scom='#')
        print(" tabulated magnetic field ... ", end="")
        if tabulated_undulator_mthd  != 0:   # similar to srwl_bl.set_und_per_from_tab()
            # TODO: parametrise
            """Setup periodic Magnetic Field from Tabulated one
            :param _rel_ac_thr: relative accuracy threshold
            :param _max_nh: max. number of harmonics to create
            :param _max_per: max. period length to consider
            """
            _rel_ac_thr=0.05
            _max_nh=50
            _max_per=0.1
            arHarm = []
            for i in range(_max_nh): arHarm.append(srwlib.SRWLMagFldH())
            magFldCntHarm = srwlib.SRWLMagFldC(srwlib.SRWLMagFldU(arHarm))
            srwlib.srwl.UtiUndFromMagFldTab(magFldCntHarm, magFldCnt, [_rel_ac_thr, _max_nh, _max_per])
            return magFldCntHarm
        
    return magFldCnt


def srwlCalcPartTraj(eBeam:srwlib.SRWLPartBeam, magFldCnt: srwlib.SRWLMagFldC,
                     number_points: int = 50000, ctst: float = 0, ctfi: float = 0) -> srwlib.SRWLPrtTrj:
    """
    Calculate the trajectory of an electron through a magnetic field.

    Args:
        eBeam (srwlib.SRWLPartBeam): Particle beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container representing the magnetic field.
        number_points (int, optional): Number of points for trajectory calculation. Defaults to 50000.
        ctst (float, optional): Initial time (ct) for trajectory calculation. Defaults to 0.
        ctfi (float, optional): Final time (ct) for trajectory calculation. Defaults to 0.

    Returns:
        srwlib.SRWLPrtTrj: Object containing the calculated trajectory.
    """
    partTraj = srwlib.SRWLPrtTrj()
    partTraj.partInitCond = eBeam.partStatMom1
    partTraj.allocate(number_points, True)
    partTraj.ctStart = ctst
    partTraj.ctEnd = ctfi

    arPrecPar = [1] 
    srwlib.srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)

    return partTraj


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

    if parallel is True and np.allclose(dE1, dE2):

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
            processed_chunks = pool.map(core_srwlibCalcElecFieldSR, chunks)

        for i, (intensity_chunck) in enumerate(processed_chunks):
            if i == 0:
                intensity = intensity_chunck
            else:
                intensity = np.concatenate((intensity, intensity_chunck), axis=0)

    elif parallel:

        dE = (energy_array[-1] - energy_array[0]) / num_cores
        energy_chunks = []
        for i in range(num_cores):
            bffr = copy.copy(energy_array)
            bffr = np.delete(bffr, bffr <= dE * (i))
            if i + 1 != num_cores:
                bffr = np.delete(bffr, bffr > dE * (i + 1))
            energy_chunks.append(bffr)

        results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcElecFieldSR)(
                                                                    list_pairs,
                                                                    bl,
                                                                    eBeam,
                                                                    magFldCnt,
                                                                    arPrecPar,
                                                                    h_slit_points,
                                                                    v_slit_points,
                                                                    radiation_characteristic,
                                                                    radiation_dependence,
                                                                    parallel)
                                             for list_pairs in energy_chunks)

        energy_array = []
        time_array = []
        energy_chunks = []
        k = 0
        for calcs in results:
            energy_array.append(calcs[3][0])
            time_array.append(calcs[4])
            energy_chunks.append(len(calcs[3]))
            if k == 0:
                intensity = calcs[0]
            else:
                intensity = np.concatenate((intensity, calcs[0]), axis=0)
            k+=1
        print(">>> ellapse time:")
        for ptime in range(len(time_array)):
            print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")
            
        if h_slit_points == 1 or v_slit_points == 1:
            x_axis = np.asarray([0])
            y_axis = np.asarray([0])
        else:
            x_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
            y_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    else:
        results = core_srwlibCalcElecFieldSR(energy_array,
                                             bl, 
                                             eBeam,
                                             magFldCnt, 
                                             arPrecPar, 
                                             h_slit_points, 
                                             v_slit_points, 
                                             radiation_characteristic, 
                                             radiation_dependence,
                                             parallel)
        intensity = results[0]
        x_axis = results[1]
        y_axis = results[2] 

    return intensity, x_axis, y_axis


def core_srwlibCalcElecFieldSR(energy_array: np.ndarray,
                               bl: dict, 
                               eBeam: srwlib.SRWLPartBeam, 
                               magFldCnt: srwlib.SRWLMagFldC, 
                               arPrecPar: List[float], 
                               h_slit_points: int, 
                               v_slit_points: int, 
                               rad_characteristic: int, 
                               rad_dependence: int, 
                               parallel: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Core function to calculate electric field for synchrotron radiation.

    Args:
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

    tzero = time.time()

    _inPol = 6
    _inIntType = rad_characteristic
    _inDepType = rad_dependence

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
        _inDepType = 0
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        _inDepType = 3
        intensity = np.zeros((energy_array.size, hAxis.size, vAxis.size))

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

                arI0 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) #"flat" array to take 2D intensity data
                srwlib.srwl.CalcIntFromElecField(arI0, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)

                data = np.ndarray(buffer=arI0, shape=(wfr.mesh.ny, wfr.mesh.nx), dtype=arI0.typecode)

                for ix in range(h_slit_points):
                    for iy in range(v_slit_points):
                        intensity[ie, ix, iy,] = data[iy, ix]
            except:
                print("Error running SRW")
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
                pass
        except:
            print("Error running SRW")

    return intensity, hAxis, vAxis, time.time()-tzero


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
                                                                    arPrecPar,
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
                                          arPrecPar)
        
        intensity = np.asarray(results[0], dtype="float64")

    return intensity


def core_srwlibCalcStokesUR(energy_array: np.ndarray, 
                            bl: dict, 
                            eBeam: srwlib.SRWLPartBeam, 
                            magFldCnt: srwlib.SRWLMagFldC, 
                            arPrecPar: List[float]) -> Tuple[np.ndarray, float]:
    """
    Core function to calculate Stokes parameters for undulator radiation.

    Args:
        energy_array (np.ndarray): Array of photon energies [eV].
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        arPrecPar (List[float]): Array of parameters for UR calculation.
    Returns:
        Tuple[np.ndarray, float]: Tuple containing intensity data and computation time.
    """

    tzero = time.time()

    try:
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
        print("Error running SRW")

    return intensity, energy_array, time.time()-tzero


def srwlibsrwl_wfr_emit_prop_multi_e(bl,
                                     eBeam, 
                                     magFldCnt, 
                                     energy_array,
                                     h_slit_points, 
                                     v_slit_points, 
                                     number_macro_electrons, 
                                     aux_file_name):
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

    Returns:
        np.ndarray: Array containing intensity data.
    """
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
    nMacroElecAvgPerProc = 10   # number of macro-electrons / wavefront to average on worker processes
    nMacroElecSavePer = 100     # intermediate data saving periodicity (in macro-electrons)
    srCalcMeth = 1              # SR calculation method 
    srCalcPrec = 0.01           # SR calculation rel. accuracy

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
    
    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    return stk.to_int(_pol=6), hAxis, vAxis

#***********************************************************************************
# Potpourri
#***********************************************************************************

def syned_dictionary(json_file: str, magnetic_measurement: Union[str, None], observation_point: float, 
                     hor_slit: float, ver_slit: float, hor_slit_cen: float, ver_slit_cen: float, 
                     Kh: float, Kh_phase: float, Kh_symmetry: int, Kv: float, Kv_phase: float, 
                     Kv_symmetry: int) -> dict:
    """
    Generate beamline parameters based on SYNED JSON configuration file and input parameters.

    Args:
        json_file (str): The path to the SYNED JSON configuration file.
        magnetic_measurement (Union[str, None]): The path to the file containing magnetic measurement data.
            Overrides SYNED undulator data.
        observation_point (float): The distance to the observation point [m].
        hor_slit (float): Horizontal slit size [m].
        ver_slit (float): Vertical slit size [m].
        hor_slit_cen (float): Horizontal slit center position [m].
        ver_slit_cen (float): Vertical slit center position [m].
        Kh (float): Horizontal undulator parameter K. If -1, it's taken from the SYNED file.
        Kh_phase (float): Initial phase of the horizontal magnetic field [rad].
        Kh_symmetry (int): Symmetry of the horizontal magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)).
        Kv (float): Vertical undulator parameter K. If -1, it's taken from the SYNED file.
        Kv_phase (float): Initial phase of the vertical magnetic field [rad].
        Kv_symmetry (int): Symmetry of the vertical magnetic field vs longitudinal position.
            1 for symmetric (B ~ cos(2*Pi*n*z/per + ph)),
           -1 for anti-symmetric (B ~ sin(2*Pi*n*z/per + ph)).

    Returns:
        dict: A dictionary containing beamline parameters.
    """

    data = read_syned_file(json_file)

    if magnetic_measurement is None:
        if Kh == -1:
            Kh =  data["magnetic_structure"]["K_horizontal"]
        if Kv == -1:
            Kv =  data["magnetic_structure"]["K_vertical"]
            
    beamline = {}
    # accelerator
    beamline['ElectronEnergy'] = data["electron_beam"]["energy_in_GeV"]
    beamline['ElectronCurrent'] = data["electron_beam"]["current"]
    beamline['ElectronEnergySpread'] = data["electron_beam"]["energy_spread"]
    # electron beam
    beamline['ElectronBeamSizeH'] = np.sqrt(data["electron_beam"]["moment_xx"])
    beamline['ElectronBeamSizeV'] = np.sqrt(data["electron_beam"]["moment_yy"])
    beamline['ElectronBeamDivergenceH'] = np.sqrt(data["electron_beam"]["moment_xpxp"])
    beamline['ElectronBeamDivergenceV'] = np.sqrt(data["electron_beam"]["moment_ypyp"])
    # undulator
    beamline['magnetic_measurement'] = magnetic_measurement
    beamline['NPeriods'] = data["magnetic_structure"]["number_of_periods"]
    beamline['PeriodID'] = data["magnetic_structure"]["period_length"]
    beamline['Kh'] = Kh
    beamline['KhPhase'] = Kh_phase
    beamline['MagFieldSymmetryH'] = Kh_symmetry
    beamline['Kv'] = Kv
    beamline['KvPhase'] = Kv_phase
    beamline['MagFieldSymmetryV'] = Kv_symmetry
    # radiation observation
    beamline['distance'] = observation_point
    beamline['slitH'] = hor_slit
    beamline['slitV'] = ver_slit
    beamline['slitHcenter'] = hor_slit_cen
    beamline['slitVcenter'] = ver_slit_cen
  
    return beamline


def get_undulator_max_harmonic_number(resonant_energy: float, photonEnergyMax: float) -> int:
    """
    Calculate the maximum harmonic number for an undulator to be considered by srwlib.CalcStokesUR.

    Args:
        resonant_energy (float): The resonance energy of the undulator [eV].
        photonEnergyMax (float): The maximum photon energy of interest [eV].

    Returns:
        int: The maximum harmonic number.
    """
    srw_max_harmonic_number = int(photonEnergyMax / resonant_energy * 2.5)

    return srw_max_harmonic_number


if __name__ == '__main__':
    print("This is the barc4sr.xoppy lib!")
    print("This module provides functions for interfacing SRW when calculating undulator radiation, power density, and spectra.")
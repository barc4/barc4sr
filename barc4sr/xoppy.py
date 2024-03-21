
"""
This module provides functions for interfacing with SRW when calculating undulator 
radiation, power density, and spectra.
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MARCH/2024'
__changed__ = '21/MARCH/2024'

import array
import copy
import json
import multiprocessing as mp
import pickle
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants

from barc4sr.undulator import get_B_from_K, get_emission_energy
from barc4sr.utils import generate_logarithmic_energy_values, get_gamma

try:
    import srwpy.srwl_bl as srwl_bl
    import srwpy.srwlib as srwlib

    USE_SRWLIB = True
    print('SRW distribution of SRW')
except:
    import oasys_srw.srwl_bl as srwl_bl
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

scanCounter = 0

#***********************************************************************************
# read/write functions for input values
#***********************************************************************************
    
def read_syned_file(json_file: str) -> Dict[str, Any]:
    """
    Reads a SYNED JSON configuration file and returns its contents as a dictionary.

    Parameters:
        json_file (str): The path to the SYNED JSON configuration file.

    Returns:
        dict: A dictionary containing the contents of the JSON file.
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def generate_beamline(json_file: str, magnetic_measurement: Union[str, None], observation_point: float, 
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

#***********************************************************************************
# xoppy_undulators.py modernised modules
#***********************************************************************************

def xoppy_calc_undulator_spectrum(file_name: str, json_file: str, photon_energy_min: float, 
                                  photon_energy_max: float, photon_energy_points: int, 
                                  energy_sampling: int, observation_point: float, 
                                  hor_slit: float, ver_slit: float,  hor_slit_cen: float = 0, 
                                  ver_slit_cen: float = 0, Kh: float = -1, Kh_phase: float = 0, 
                                  Kh_symmetry: int = 1, Kv: float = -1, Kv_phase: float = 0, 
                                  Kv_symmetry: int = 1, magnetic_measurement: Optional[str] = None, 
                                  tabulated_undulator_mthd: int = 0, electron_trajectory: bool = False, 
                                  filament_beam: bool = False, energy_spread: bool = True, 
                                  calculation: int = 0, number_macro_electrons: int = 1000, 
                                  parallel: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 1D undulator spectrum using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.
        energy_sampling (int): Energy sampling method (0: linear, 1: logarithmic).
        observation_point (float): Distance to the observation point [m].
        hor_slit (float): Horizontal slit size [m].
        ver_slit (float): Vertical slit size [m].
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
        tabulated_undulator_mthd (int): Method to tabulate the undulator field (0: uses the provided magnetic field, 
            1: fits the magnetic field using srwl.UtiUndFromMagFldTab). Default is 0.
        electron_trajectory (bool): Whether to calculate and save electron trajectory. Default is False.
        filament_beam (bool): Whether to use a filament electron beam. Default is False.
        energy_spread (bool): Whether to include energy spread. Default is True.
        calculation (int): Spectrum calculation method (0: on-axis, 1: through finite aperture). Default is 0.
        number_macro_electrons (int): Number of macro electrons. Default is 1000.
        parallel (bool): Whether to use parallel computation. Default is False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of photon energy and flux.
    """
    
    bl = generate_beamline(json_file, magnetic_measurement, observation_point, hor_slit, 
                           ver_slit, hor_slit_cen, ver_slit_cen, Kh, Kh_phase, Kh_symmetry, 
                           Kv, Kv_phase, Kv_symmetry)

    print("Undulator flux calculation using SRW. Please wait...")
    energy, flux = undulator_spectrum_1d_srw(
                    bl,
                    photon_energy_min=photon_energy_min,
                    photon_energy_max=photon_energy_max,
                    photon_energy_points=photon_energy_points,
                    energy_sampling = energy_sampling,
                    calculation = calculation,
                    filament_beam = filament_beam,
                    energy_spread = energy_spread,
                    magnetic_measurement=magnetic_measurement,
                    tabulated_undulator_mthd=tabulated_undulator_mthd,
                    electron_trajectory=electron_trajectory,
                    number_macro_electrons=number_macro_electrons,
                    aux_file_name=file_name,
                    parallel=parallel,
                    )
    print("Undulator flux calculation using SRW: finished")
        
    file = open('%s_spectrum.pickle'%file_name, 'wb')
    pickle.dump([energy, flux], file)
    file.close()
    return energy, flux

def undulator_spectrum_1d_srw(bl: dict,
                              photon_energy_min: float, 
                              photon_energy_max: float, 
                              photon_energy_points: int, 
                              energy_sampling: int, 
                              calculation: int, 
                              filament_beam: bool,
                              energy_spread: bool,
                              magnetic_measurement: Optional[str], 
                              tabulated_undulator_mthd: int, 
                              electron_trajectory: bool, 
                              number_macro_electrons: int, 
                              aux_file_name: str, 
                              parallel: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate undulator spectrum vs photon energy.

    Args:
        bl (dict): Beamline parameters dictionary.
        photon_energy_min (float): Minimum photon energy [eV].
        photon_energy_max (float): Maximum photon energy [eV].
        photon_energy_points (int): Number of photon energy points.
        energy_sampling (int): Energy sampling method (0: linear, 1: logarithmic).
        calculation (int): Spectrum calculation method (0: on-axis, 1: through finite aperture).
        filament_beam (bool): Whether to use a filament electron beam.
        energy_spread (bool): Whether to include energy spread.
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field (0: constant, 1: quadratic).
        electron_trajectory (bool): Whether to calculate electron trajectory.
        number_macro_electrons (int): Number of macro electrons.
        aux_file_name (str): Auxiliary file name.
        parallel (bool): Whether to use parallel computation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of photon energy and intensity.
    """
    t0 = time.time()
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
        electron_trajectory_file_name = aux_file_name+"_eTraj.dat"
        eTraj = calculate_electron_trajectory(eBeam, magFldCnt)
        eTraj.save_ascii(electron_trajectory_file_name)
        print(f">>>{electron_trajectory_file_name}<<< ", end='')
    else:
        eTraj = 0
    print('completed')

    # ----------------------------------------------------------------------------------
    # spectrum calculations
    # ----------------------------------------------------------------------------------
    resonant_energy = get_emission_energy(bl['PeriodID'], 
                                        np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                        bl['ElectronEnergy'])
    if energy_sampling == 0: 
        energy_array = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        stepsize = np.log(photon_energy_max/resonant_energy)
        energy_array = generate_logarithmic_energy_values(photon_energy_min,
                                                          photon_energy_max,
                                                          resonant_energy,
                                                          stepsize)

    # ---------------------------------------------------------
    # On-Axis Spectrum from Filament Electron Beam (total pol.)
    if calculation == 0:
        print('> Performing on-axis spectrum from filament electron beam ... ', end='')
        intensity_array = interface_srwlibCalcElecFieldSR(bl, 
                                                          eBeam, 
                                                          magFldCnt,
                                                          energy_array,
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
        print('> Performing flux through finite aperture (simplified partially-coherent simulation)... ', end='')
        intensity_array = interface_srwlibCalcStokesUR(bl, 
                                                       eBeam, 
                                                       magFldCnt, 
                                                       energy_array, 
                                                       resonant_energy,
                                                       parallel)

        print('completed')
    # accurate partially-coherent simulation
    if calculation == 1 and magnetic_measurement is not None:
        print('> Performing flux through finite aperture (accurate partially-coherent simulation)... ', end='')



        intensity_array = interface_srwl_wfr_emit_prop_multi_e(bl, 
                                                               eBeam,
                                                               magFldCnt,
                                                               energy_array,
                                                               1,
                                                               1,
                                                               number_macro_electrons,
                                                               aux_file_name)       
        print('completed')

    return energy_array, intensity_array


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


def calculate_electron_trajectory(eBeam:srwlib.SRWLPartBeam, magFldCnt: srwlib.SRWLMagFldC,
                                  number_points: int = 50000, ctst: float = 0, ctfi: float = 0             
                                  ) -> srwlib.SRWLPrtTrj:
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


def interface_srwlibCalcElecFieldSR(bl, eBeam, magFldCnt, energy_array,
                                    h_slit_points, v_slit_points, radiation_characteristic, 
                                    radiation_dependence, parallel):
    
    arPrecPar = [0]*7
    arPrecPar[0] = 1     # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    arPrecPar[1] = 0.01  # relative precision
    arPrecPar[2] = 0     # longitudinal position to start integration (effective if < zEndInteg)
    arPrecPar[3] = 0     # longitudinal position to finish integration (effective if > zStartInteg)
    arPrecPar[4] = 50000 # Number of points for trajectory calculation
    arPrecPar[5] = 1     # Use "terminating terms"  or not (1 or 0 respectively)
    arPrecPar[6] = 0     # sampling factor for adjusting nx, ny (effective if > 0)

    if parallel:
        pass

    else:
        results = core_srwlibCalcElecFieldSR(bl, 
                                             eBeam,
                                             magFldCnt, 
                                             arPrecPar, 
                                             energy_array,
                                             h_slit_points, 
                                             v_slit_points, 
                                             radiation_characteristic, 
                                             radiation_dependence,
                                             parallel)
        intensity = results[0]

    return intensity


def core_srwlibCalcElecFieldSR(bl, eBeam, magFldCnt, arPrecPar, energy_array,
                                    h_slit_points, v_slit_points, rad_characteristic, 
                                    rad_dependence, parallel):

    tzero = time.time()

    _inPol = 6
    _inIntType = rad_characteristic
    _inDepType = rad_dependence
    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
        _inDepType = 0
    else:
        hAxis = np.linspace(-bl['slitH'] / 2, bl['slitH'] / 2, h_slit_points)
        vAxis = np.linspace(-bl['slitV'] / 2, bl['slitV'] / 2, v_slit_points)
        _inDepType = 3
        intensity = np.zeros((energy_array.size, hAxis.size, vAxis.size))

    if parallel:
        pass

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

            arI1 = array.array('f', [0]*wfr.mesh.ne)
            srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
            if _inDepType == 0:    # 0 -vs e (photon energy or time);
                intensity = np.asarray(arI1, dtype="float64")
        except:
            print("Error running SRW")

    return intensity, hAxis, vAxis, time.time()-tzero


def interface_srwlibCalcStokesUR(bl, eBeam, magFldCnt, energy_array, resonant_energy, parallel):
    
    arPrecPar = [0]*5   # for spectral flux vs photon energy
    arPrecPar[0] = 1    # initial UR harmonic to take into account
    arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy_array[-1]) #final UR harmonic to take into account
    arPrecPar[2] = 1.5  # longitudinal integration precision parameter
    arPrecPar[3] = 1.5  # azimuthal integration precision parameter
    arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

    t0 = time.time()

    if parallel:
        pass

    else:
        results = core_srwlibCalcStokesUR(bl, 
                                          eBeam,
                                          magFldCnt, 
                                          arPrecPar,
                                          energy_array, 
                                          parallel)
        intensity = results[0]

    return intensity


def core_srwlibCalcStokesUR(bl, eBeam, magFldCnt, arPrecPar, energy_array, parallel):

    tzero = time.time()

    if parallel:
        pass

    else:
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
            intensArray = stk.arS[0:npts]
        except:
            print("Error running SRW")

    return intensArray, time.time()-tzero


def interface_srwl_wfr_emit_prop_multi_e(bl, eBeam, magFldCnt, energy_array,
                                         h_slit_points, v_slit_points, number_macro_electrons, 
                                         aux_file_name):

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
    intensArray = stk.to_int(_pol=6)

    return intensArray


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
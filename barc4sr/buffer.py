
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MARCH/2024'
__changed__ = '15/MARCH/2024'

import array
import copy
import json
import multiprocessing as mp
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
    
def read_syned_file(jsonfile: str) -> Dict[str, Any]:
    """
    Reads a SYNED JSON configuration file and returns its contents as a dictionary.

    Parameters:
        jsonfile (str): The path to the SYNED JSON configuration file.

    Returns:
        dict: A dictionary containing the contents of the JSON file.
    """
    with open(jsonfile) as f:
        data = json.load(f)
    return data

#***********************************************************************************
# xoppy_undulators.py modernised modules
#***********************************************************************************

def xoppy_calc_undulator_spectrum(ELECTRONENERGY=6.04,
                                  ELECTRONENERGYSPREAD=0.001,
                                  ELECTRONCURRENT=0.2,
                                  ELECTRONBEAMSIZEH=0.000395,
                                  ELECTRONBEAMSIZEV=9.9e-06,
                                  ELECTRONBEAMDIVERGENCEH=1.05e-05,
                                  ELECTRONBEAMDIVERGENCEV=3.9e-06,
                                  PERIODID=0.018,
                                  NPERIODS=222,
                                  KV=1.68,
                                  KVPHASE=0.0,
                                  VSIMMETRY=1,
                                  KH=0.0,
                                  KHPHASE=0.0,
                                  HSIMMETRY=1,
                                  DISTANCE=30.0,
                                  GAPH=0.001,
                                  GAPV=0.001,
                                  GAPH_CENTER=0.0,
                                  GAPV_CENTER=0.0,
                                  PHOTONENERGYMIN=3000.0,
                                  PHOTONENERGYMAX=55000.0,
                                  PHOTONENERGYPOINTS=500,
                                  USEEMITTANCES=1,
                                  MULTIELECTRONS=0):
    
    print("> Inside xoppy_calc_undulator_spectrum. \n")

    bl = OrderedDict()
    bl['ElectronBeamDivergenceH'] = ELECTRONBEAMDIVERGENCEH
    bl['ElectronBeamDivergenceV'] = ELECTRONBEAMDIVERGENCEV
    bl['ElectronBeamSizeH'] = ELECTRONBEAMSIZEH
    bl['ElectronBeamSizeV'] = ELECTRONBEAMSIZEV
    bl['ElectronCurrent'] = ELECTRONCURRENT
    bl['ElectronEnergy'] = ELECTRONENERGY
    bl['ElectronEnergySpread'] = ELECTRONENERGYSPREAD
    bl['Kv'] = KV
    bl['Kh'] = KH
    bl['KvPhase'] = KVPHASE
    bl['KvPhase'] = KHPHASE
    bl['NPeriods'] = NPERIODS
    bl['PeriodID'] = PERIODID
    bl['MagFieldSymmetryH'] = HSIMMETRY
    bl['MagFieldSymmetryV'] = VSIMMETRY
    bl['distance'] = DISTANCE
    bl['gapH'] = GAPH
    bl['gapV'] = GAPV
    bl['gapHcenter'] = GAPH_CENTER
    bl['gapVcenter'] = GAPV_CENTER

    if USEEMITTANCES:
        zero_emittance = False
    else:
        zero_emittance = True

    if MULTIELECTRONS:
        zero_emittance = False
    else:
        zero_emittance = True
  
    print("Undulator flux calculation using SRW. Please wait...")
    e, f = calc_undulator_spectrum_1d_srw(
                        bl,
                        photon_energy_min=PHOTONENERGYMIN,
                        photon_energy_max=PHOTONENERGYMAX,
                        photon_energy_points=PHOTONENERGYPOINTS,
                        zero_emittance=zero_emittance,
                        )
    print("Done")
        
    return e, f

def calc_undulator_spectrum_1d_srw(bl,
                                   photon_energy_min=3000.0,
                                   photon_energy_max=55000.0,
                                   photon_energy_points=500,
                                   energy_sampling = 0,
                                   calculation = False,
                                   zero_emittance = False,
                                   magnetic_measurement=None,
                                   tabulated_undulator_mthd=0,
                                   electron_trajectory=False,
                                   parallel=False):
    
    t0 = time.time()
    print("inside calc_undulator_spectrum_1d_srw")

    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    eBeam = set_electron_beam(bl,
                              zero_emittance)
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    magFldCnt = set_magnetic_structure(bl,
                                       magnetic_measurement, 
                                       tabulated_undulator_mthd)
    # ----------------------------------------------------------------------------------
    # calculate electron trajectory
    # ----------------------------------------------------------------------------------
    if electron_trajectory:
        eTraj = calcualte_electron_trajectory(eBeam, magFldCnt)
    else:
        eTraj = 0
    # ----------------------------------------------------------------------------------
    # spectrum calculations
    # ----------------------------------------------------------------------------------
    if energy_sampling == 0: #lins_paced
        energy_array = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
    else:
        resonant_energy = get_emission_energy(bl['PeriodID'], 
                                              np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                              bl['ElectronEnergy'])
        stepsize = np.log(photon_energy_max/resonant_energy)
        energy_array = generate_logarithmic_energy_values(photon_energy_min,
                                                          photon_energy_max,
                                                          resonant_energy,
                                                          stepsize)
    # ---------------------------------------------------------
    # On-Axis Spectrum from Filament Electron Beam (total pol.)
    if calculation == 0:
        intensity_array = interface_srwlibCalcElecFieldSR(bl, 
                                                          eBeam, 
                                                          magFldCnt,
                                                          eTraj,
                                                          photon_energy_min,
                                                          photon_energy_max,
                                                          photon_energy_points,
                                                          energy_array,
                                                          h_slit_points=1,
                                                          v_slit_points=1,
                                                          parallel=parallel)
    # -----------------------------------------
    # Flux through Finite Aperture (total pol.)

    # simplified partially-coherent simulation
    if calculation == 1:
        pass
    # real partially-coherent simulation
    if calculation == 2:
        pass
    

def set_electron_beam(bl: dict, zero_emittance: bool) -> srwlib.SRWLPartBeam:
    """
    Set up the electron beam parameters.

    Parameters:
        bl (dict): Dictionary containing beamline parameters.
        zero_emittance (bool): Flag indicating whether to set the beam emittance to zero.

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

    if zero_emittance:
        sigX = 1e-25
        sigXp = 1e-25
        sigY = 1e-25
        sigYp = 1e-25
        sigEperE = 1e-25
    else:
        sigX = bl['ElectronBeamSizeH']  # horizontal RMS size of e-beam [m]
        sigXp = bl['ElectronBeamDivergenceH']  # horizontal RMS angular divergence [rad]
        sigY = bl['ElectronBeamSizeV']  # vertical RMS size of e-beam [m]
        sigYp = bl['ElectronBeamDivergenceV']  # vertical RMS angular divergence [rad]
        sigEperE = bl['ElectronEnergySpread']

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
                    _bx=get_B_from_K(bl['Kv'],bl["PeriodID"]), 
                    _by=get_B_from_K(bl['Kh'],bl["PeriodID"]), 
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
            _max_nh=5
            _max_per=0.1
            arHarm = []
            for i in range(_max_nh): arHarm.append(srwlib.SRWLMagFldH())
            magFldCntHarm = srwlib.SRWLMagFldC(srwlib.SRWLMagFldU(arHarm))
            srwlib.UtiUndFromMagFldTab(magFldCntHarm, magFldCnt, [_rel_ac_thr, _max_nh, _max_per])
            return magFldCntHarm
    return magFldCnt


def calcualte_electron_trajectory(eBeam:srwlib.SRWLPartBeam, magFldCnt: srwlib.SRWLMagFldC,
                                  number_points: int = 50000, ctst: float = -1, ctfi: float = 1             
                                  ) -> srwlib.SRWLPrtTrj:
    """
    Calculate the trajectory of an electron through a magnetic field.

    Args:
        eBeam (srwlib.SRWLPartBeam): Particle beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container representing the magnetic field.
        number_points (int, optional): Number of points for trajectory calculation. Defaults to 50000.
        ctst (float, optional): Initial time (ct) for trajectory calculation. Defaults to -1.
        ctfi (float, optional): Final time (ct) for trajectory calculation. Defaults to 1.

    Returns:
        srwlib.SRWLPrtTrj: Object containing the calculated trajectory.
    """
    partTraj = srwlib.SRWLPrtTrj()
    partTraj.partInitCond = eBeam.partStatMom1
    partTraj.allocate(number_points, True)
    partTraj.ctStart = ctst
    partTraj.ctEnd = ctfi

    arPrecPar = [1] 
    print('Electron trajectory calculation ... ', end='')
    srwlib.CalcPartTraj(partTraj, magFldCnt, arPrecPar)
    print('completed')

    return srwlib.SRWLPrtTrj


def interface_srwlibCalcElecFieldSR(bl, eBeam, magFldCnt, eTraj, energy_array,
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

    t0 = time.time()

    if parallel:
        pass
        # num_cores = mp.cpu_count()
        # dE = (energy_array[-1] - energy_array[0]) / num_cores
        # energy_chunks = []
        # for i in range(num_cores):
        #     bffr = copy.copy(energy_array)
        #     bffr = np.delete(bffr, bffr <= dE * (i))
        #     if i + 1 != num_cores:
        #         bffr = np.delete(bffr, bffr > dE * (i + 1))
        #     energy_chunks.append(bffr)

        # results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcElecFieldSR)()
        #                                      for list_pairs in energy_chunks)
        # energy_array = []
        # time_array = []
        # energy_chunks = []
        # k = 0
        # for stuff in results:
        #     energy_array.append(stuff[3][0])
        #     time_array.append(stuff[4])
        #     energy_chunks.append(len(stuff[3]))
        #     if k == 0:
        #         intensity = stuff[0]
        #     else:
        #         intensity = np.concatenate((intensity, stuff[0]), axis=0)
        #     k+=1
        # print(">>> ellapse time:")
        # for ptime in range(len(time_array)):
        #     print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")
            
        # hAxis = np.linspace(-bl['gapH'] / 2, bl['gapH'] / 2, h_slit_points)
        # vAxis = np.linspace(-bl['gapV'] / 2, bl['gapV'] / 2, v_slit_points)

    else:
        # intensity, hAxis, vAxis, energy_array, t
        results = core_srwlibCalcElecFieldSR(bl, 
                                             eBeam,
                                             magFldCnt, 
                                             eTraj, 
                                             arPrecPar, 
                                             energy_array,
                                             h_slit_points, 
                                             v_slit_points, 
                                             radiation_characteristic, 
                                             radiation_dependence,
                                             parallel)
        intensity = results[0]

    return intensity


def core_srwlibCalcElecFieldSR(bl, eBeam, magFldCnt, eTraj, arPrecPar, energy_array,
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
        hAxis = np.linspace(-bl['gapH'] / 2, bl['gapH'] / 2, h_slit_points)
        vAxis = np.linspace(-bl['gapV'] / 2, bl['gapV'] / 2, v_slit_points)
        _inDepType = 3
        intensity = np.zeros((energy_array.size, hAxis.size, vAxis.size))

    if parallel:
        pass
        # for ie in range(energy_array.size):
        #     try:
        #         mesh = srwlib.SRWLRadMesh(energy_array[ie], energy_array[ie], 1,
        #                                 hAxis[0], hAxis[-1], h_slit_points,
        #                                 vAxis[0], vAxis[-1], v_slit_points, 
        #                                 bl['distance'])

        #         wfr = srwlib.SRWLWfr()
        #         wfr.allocate(1, mesh.nx, mesh.ny)
        #         wfr.mesh = mesh
        #         wfr.partBeam = eBeam

        #         srwlib.srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecPar)
        #         mesh0 = wfr.mesh
        #         arI0 = array.array('f', [0]*mesh0.nx*mesh0.ny) 
        #         srwlib.srwl.CalcIntFromElecField(arI0, wfr, _inPol, _inIntType, _inDepType, energy_array[ie], 0, 0)
        #         if _inDepType == 3:
        #             data = np.ndarray(buffer=arI0, shape=(mesh0.ny, mesh0.nx), dtype=arI0.typecode)
        #             for ix in range(hAxis.size):
        #                 for iy in range(vAxis.size):
        #                     intensity[ie, ix, iy,] = data[iy, ix]
        #         else:
        #             intensity = np.asarray
        #     except:
        #         print("Error running SRW")
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
            srwlib.srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecPar)
            arI1 = array.array('f', [0]*wfr.mesh.ne)
            srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
            if _inDepType == 0:    # 0 -vs e (photon energy or time);
                intensity = np.asarray(arI1, dtype="float64")
        except:
            print("Error running SRW")

    return intensity, hAxis, vAxis, time.time()-tzero


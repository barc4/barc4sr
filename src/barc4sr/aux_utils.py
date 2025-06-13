#!/bin/python

""" 
This module provides SR classes, SRW interfaced functions, r/w SYNED compatible functions,
r/w functions for the electron trajectory and magnetic field as well as other auxiliary 
functions.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MAR/2024'
__changed__ = '04/DEC/2024'

import array
import copy
import multiprocessing as mp
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants

from barc4sr.aux_energy import get_gamma

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

#***********************************************************************************
# SRW interface functions (high level)
#***********************************************************************************

def set_light_source(file_name: str,
                     bl: dict,
                     electron_trajectory: bool,
                     id_type: str,
                     **kwargs) -> Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]:
    """
    Set up the light source parameters including electron beam, magnetic structure, and electron trajectory.

    Args:
        file_name (str): The name of the output file.
        bl (dict): Beamline parameters dictionary containing essential information for setup.
        electron_trajectory (bool): Whether to calculate and save electron trajectory.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        magfield_initial_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (str): Path to the file containing magnetic measurement data.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field.

    Returns:
        Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]: A tuple containing the electron beam,
        magnetic structure, and electron trajectory.
    """    

    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)
    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)

    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    print('> Generating the electron beam ... ', end='')
    eBeam = set_electron_beam(bl,
                              id_type,
                              ebeam_initial_position=ebeam_initial_position)
    print('completed')
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    print('> Generating the magnetic structure ... ', end='')
    magFldCnt = set_magnetic_structure(bl, 
                                       id_type,
                                       magnetic_measurement = magnetic_measurement, 
                                       magfield_central_position = magfield_central_position,
                                       tabulated_undulator_mthd = tabulated_undulator_mthd)
    print('completed')
    # ----------------------------------------------------------------------------------
    # calculate electron trajectory
    # ----------------------------------------------------------------------------------
    print('> Electron trajectory calculation ... ', end='')
    if electron_trajectory:
        # electron_trajectory_file_name = file_name+"_eTraj.dat"
        eTraj = srwlCalcPartTraj(eBeam, magFldCnt)
        # eTraj.save_ascii(electron_trajectory_file_name)
        # print(f">>>{electron_trajectory_file_name}<<< ", end='')
    else:
        eTraj = 0
    print('completed')
    return eBeam, magFldCnt, eTraj


def set_electron_beam(bl: dict, 
                      id_type: str, 
                      **kwargs) -> srwlib.SRWLPartBeam:
    """
    Set up the electron beam parameters.

    Parameters:
        bl (dict): Dictionary containing beamline parameters.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]

    Returns:
        srwlib.SRWLPartBeam: Electron beam object initialized with specified parameters.

    """
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    eBeam = srwlib.SRWLPartBeam()
    eBeam.Iavg = bl['ElectronCurrent']  # average current [A]
    eBeam.partStatMom1.x = 0.  # initial transverse positions [m]
    eBeam.partStatMom1.y = 0.
    if id_type.startswith('u'):
        eBeam.partStatMom1.z = - bl['PeriodID'] * (bl['NPeriods'] + 4) / 2  # initial longitudinal positions
    else:
        eBeam.partStatMom1.z = ebeam_initial_position
    eBeam.partStatMom1.xp = 0  # initial relative transverse divergence [rad]
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = get_gamma(bl['ElectronEnergy'])

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


def set_magnetic_structure(bl: dict, 
                           id_type: str, 
                           **kwargs) -> srwlib.SRWLMagFldC:
    """
    Sets up the magnetic field container.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        magfield_central_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (str): Path to the tabulated magnetic field data.
        tabulated_undulator_mthd (int): Method to use for generating undulator field if magnetic_measurement is provided. Defaults to 0

    Returns:
        srwlib.SRWLMagFldC: Magnetic field container.

    """
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    magfield_central_position = kwargs.get('magfield_central_position', 0)

    if id_type.startswith('u'):
        tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)
        if magnetic_measurement is None:    # ideal sinusoidal undulator magnetic structure
            und = srwlib.SRWLMagFldU()
            und.set_sin(_per=bl["PeriodID"],
                        _len=bl['PeriodID']*bl['NPeriods'], 
                        _bx=bl['Kh']*2*PI*MASS*LIGHT/(CHARGE*bl["PeriodID"]), 
                        _by=bl['Kv']*2*PI*MASS*LIGHT/(CHARGE*bl["PeriodID"]), 
                        _phx=bl['MagFieldPhaseH'], 
                        _phy=bl['MagFieldPhaseV'], 
                        _sx=bl['MagFieldSymmetryH'], 
                        _sy=bl['MagFieldSymmetryV'])

            magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[und],
                                            _arXc=srwlib.array('d', [0.0]),
                                            _arYc=srwlib.array('d', [0.0]),
                                            _arZc=srwlib.array('d', [magfield_central_position]))
            
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
                for i in range(_max_nh): 
                    arHarm.append(srwlib.SRWLMagFldH())
                magFldCntHarm = srwlib.SRWLMagFldC(srwlib.SRWLMagFldU(arHarm))
                srwlib.srwl.UtiUndFromMagFldTab(magFldCntHarm, magFldCnt, [_rel_ac_thr, _max_nh, _max_per])
                return magFldCntHarm
            
    if id_type.startswith('bm'):
        # RC:2025JAN08 TODO: recheck magfield central position/extraction angle for edge radiation
        bm = srwlib.SRWLMagFldM()
        bm.G = bl["B"]
        bm.m = 1         # multipole order: 1 for dipole, 2 for quadrupole, 3 for sextupole, 4 for octupole
        bm.n_or_s = 'n'  # normal ('n') or skew ('s')
        bm.Leff = bl["Leff"]
        bm.Ledge = bl["Ledge"]
        bm.R = bl["R"]

        magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[bm],
                                       _arXc=srwlib.array('d', [0.0]),
                                       _arYc=srwlib.array('d', [0.0]),
                                       _arZc=srwlib.array('d', [magfield_central_position]))

    return magFldCnt

#***********************************************************************************
# SRW interface functions (low level)
#***********************************************************************************

def srwlCalcPartTraj(eBeam:srwlib.SRWLPartBeam,
                     magFldCnt: srwlib.SRWLMagFldC,
                     number_points: int = 50000, 
                     ctst: float = 0, 
                     ctfi: float = 0) -> srwlib.SRWLPrtTrj:
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
                          radiation_polarisation: int,
                          id_type: str,
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
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                            it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data, horizontal and vertical axes
    """
    
    arPrecPar = [0]*7
    if id_type.startswith('bm') or id_type.startswith('w'):
        arPrecPar[0] = 2      # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    else:
        arPrecPar[0] = 1
    arPrecPar[1] = 0.001  
    arPrecPar[2] = 0     # longitudinal position to start integration (effective if < zEndInteg)
    arPrecPar[3] = 0     # longitudinal position to finish integration (effective if > zStartInteg)
    arPrecPar[4] = 50000 # Number of points for trajectory calculation
    arPrecPar[5] = 1     # Use "terminating terms"  or not (1 or 0 respectively)
    arPrecPar[6] = 0     # sampling factor for adjusting nx, ny (effective if > 0)

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        dE = np.diff(energy_array)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy_array[-1]>51*energy_array[0])

        # if np.allclose(dE1, dE2) and wiggler_regime:
        if wiggler_regime:
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
                    radiation_polarisation,
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

            results = Parallel(n_jobs=num_cores, backend="loky")(delayed(core_srwlibCalcElecFieldSR)((
                                                                        list_pairs,
                                                                        bl,
                                                                        eBeam,
                                                                        magFldCnt,
                                                                        arPrecPar,
                                                                        h_slit_points,
                                                                        v_slit_points,
                                                                        radiation_characteristic,
                                                                        radiation_dependence,
                                                                        radiation_polarisation,
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
                                             radiation_polarisation,
                                             parallel))
        intensity = results[0]

    if h_slit_points == 1 or v_slit_points == 1:
        x_axis = np.asarray([0])
        y_axis = np.asarray([0])
    else:
        x_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        y_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    return intensity, x_axis, y_axis


def core_srwlibCalcElecFieldSR(args: Tuple[np.ndarray, 
                                           dict, 
                                           srwlib.SRWLPartBeam, 
                                           srwlib.SRWLMagFldC, 
                                           List[float], 
                                           int, int, int, int, int, bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
            radiation_polarisation (int): Polarisation component to be extracted.
            parallel (bool): Whether to use parallel computation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Tuple containing intensity data, 
                                                          horizontal axis, vertical axis, 
                                                          and computation time.
    """

    energy_array, bl, eBeam, magFldCnt, arPrecPar,  h_slit_points, v_slit_points, \
        rad_characteristic, rad_dependence, rad_polarisation, parallel = args
    
    tzero = time()

    _inPol = rad_polarisation
    _inIntType = rad_characteristic
    _inDepType = rad_dependence

    monochromatic = False
    if isinstance(energy_array, int) or isinstance(energy_array, float):
        monochromatic = True 

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
        _inDepType = 0
        intensity = np.zeros((energy_array.size))
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        _inDepType = 3
        if monochromatic:
            intensity =  np.zeros((vAxis.size, hAxis.size))
        else:
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
                if _inIntType == 4:
                    arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                else:
                    arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)

                srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
                if _inDepType == 0:    # 0 -vs e (photon energy or time);
                    intensity[ie] = np.asarray(arI1, dtype="float64")
                else:
                    # data = np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                    data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) 
                    intensity[ie, :, :] = data
            except:
                 raise ValueError("Error running SRW.")
    else:
        try:
            if monochromatic:
                ei = ef = energy_array
                nf = 1
            else:
                ei = energy_array[0]
                ef = energy_array[-1]
                nf = len(energy_array)

            mesh = srwlib.SRWLRadMesh(ei, ef, nf,
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
                if monochromatic:
                    if _inIntType == 4:
                        arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                    else:
                        arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
                    srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, ei, 0, 0)
                    intensity = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx))
                else:
                    for ie in range(len(energy_array)):
                        if _inIntType == 4:
                            arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                        else:
                            arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
                        srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, energy_array[ie], 0, 0)
                        data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) #np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                        intensity[ie, :, :] = data

        except:
             raise ValueError("Error running SRW.")

    return intensity, hAxis, vAxis, energy_array, time()-tzero


def srwlibsrwl_wfr_emit_prop_multi_e(bl: dict,
                                     eBeam: srwlib.SRWLPartBeam, 
                                     magFldCnt: srwlib.SRWLMagFldC, 
                                     energy_array: np.ndarray,
                                     h_slit_points: int, 
                                     v_slit_points: int, 
                                     radiation_polarisation: int,
                                     id_type: str,
                                     number_macro_electrons: int, 
                                     aux_file_name: str,
                                     parallel: bool,
                                     num_cores: int=None,
                                     srApprox: int=0):
    """
    Interface function to compute multi-electron emission and propagation through a beamline using SRW.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).
        number_macro_electrons (int): Total number of macro-electrons.
        aux_file_name (str): Auxiliary file name for saving intermediate data.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.
        srApprox (int): Approximation to be used at multi-electron integration: 
                0- none (i.e. do standard M-C integration over 5D phase space volume of e-beam), 
                1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance
    Returns:
        np.ndarray: Array containing intensity data.
    """
    nMacroElecAvgPerProc = 10   # number of macro-electrons / wavefront to average on worker processes
    nMacroElecSavePer = 100     # intermediate data saving periodicity (in macro-electrons)
    if id_type.startswith('bm') or id_type.startswith('w'):
        srCalcMeth = 2          # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    else:
        srCalcMeth = 1

    srApprox = 0
    srCalcPrec = 0.01           # SR calculation rel. accuracy

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        dE = np.diff(energy_array)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy_array[-1]>51*energy_array[0])

        # if np.allclose(dE1, dE2) and wiggler_regime:
        if wiggler_regime:
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
                        srApprox,
                        radiation_polarisation,
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

            results = Parallel(n_jobs=num_cores, backend="loky")(delayed(core_srwlibsrwl_wfr_emit_prop_multi_e)((
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
                                                                        srApprox,
                                                                        radiation_polarisation,
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
                                                        srApprox,
                                                        radiation_polarisation,
                                                        nMacroElecAvgPerProc,
                                                        nMacroElecSavePer))
        intensity = np.asarray(results[0], dtype="float64")

    return intensity, hAxis, vAxis


def core_srwlibsrwl_wfr_emit_prop_multi_e(args: Tuple[np.ndarray,
                                                      dict, 
                                                      srwlib.SRWLPartBeam, 
                                                      srwlib.SRWLMagFldC, 
                                                      int, int, int, str, int, float,
                                                      int, int, int, int]) -> Tuple[np.ndarray, float]:
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
            - srApprox (int): Approximation to be used at multi-electron integration: 
                    0- none (i.e. do standard M-C integration over 5D phase space volume of e-beam), 
                    1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance
            - radiation_polarisation (int): Polarisation component to be extracted.
            - nMacroElecAvgPerProc (int): Number of macro-electrons / wavefront to average on worker processes.
            - nMacroElecSavePer (int): Intermediate data saving periodicity (in macro-electrons).

    Returns:
        tuple: A tuple containing intensity data array and the elapsed time.
    """

    energy_array, bl, eBeam, magFldCnt, h_slit_points, v_slit_points, \
        number_macro_electrons, aux_file_name, srCalcMeth, srCalcPrec, srApprox, radiation_polarisation,\
        nMacroElecAvgPerProc, nMacroElecSavePer = args
    
    tzero = time()

    try:    
        
        if isinstance(energy_array, int) or isinstance(energy_array, float):
            monochromatic = True 
            ei = ef = energy_array
            nf = 1
        else:
            monochromatic = False
            ei = energy_array[0]
            ef = energy_array[-1]
            nf = len(energy_array)

        mesh = srwlib.SRWLRadMesh(ei, 
                                  ef, 
                                  nf,
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
                                                _char=0,
                                                _me_approx=srApprox)
    
        os.system('rm %s'% MacroElecFileName)
        me_intensity = np.asarray(stk.to_int(_pol=radiation_polarisation), dtype='float64')

        if h_slit_points != 1 or v_slit_points != 1:
            k = 0
            if monochromatic:
                data = np.zeros((v_slit_points, h_slit_points))
                for iy in range(v_slit_points):
                    for ix in range(h_slit_points):
                        data[iy, ix] = me_intensity[k]
                        k+=1
            else:
                data = np.zeros((len(energy_array), v_slit_points, h_slit_points))
                for iy in range(v_slit_points):
                    for ix in range(h_slit_points):
                        for ie in range(len(energy_array)):
                            data[ie, iy, ix] = me_intensity[k]
                            k+=1
            me_intensity = data

    except:
         raise ValueError("Error running SRW.")

    return (me_intensity, energy_array, time()-tzero)


def srwlibCalcStokesUR(bl: dict, 
                       eBeam: srwlib.SRWLPartBeam, 
                       magFldCnt: srwlib.SRWLMagFldC, 
                       energy_array: np.ndarray, 
                       resonant_energy: float, 
                       h_slit_points: int, 
                       v_slit_points: int, 
                       radiation_polarisation: int,
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
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data.
    """

    def smart_split_energy(energy_array, num_cores):
        """
        Splits an array of energy values into chunks based on a weighted distribution.

        Parameters:
        energy_array (numpy.ndarray): The array of energy values to be split.
        num_cores (int): The number of chunks to split the energy array into.

        Returns:
        List[numpy.ndarray]: A list of numpy arrays, where each array is a chunk of the original energy array.
        """
        energy_array = np.sort(energy_array)
        weights = np.exp(-np.linspace(0, 2, num_cores))
        weights /= weights.sum()
        
        split_indices = np.cumsum(weights * len(energy_array)).astype(int)
        split_indices = np.insert(split_indices, 0, 0)
        split_indices[-1] = len(energy_array)

        energy_chunks = [energy_array[split_indices[i]:split_indices[i+1]] for i in range(num_cores)]
        return energy_chunks
    
    if parallel:
        if num_cores is None:
            num_cores = mp.cpu_count()

        dE = np.diff(energy_array)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy_array[-1]>21*energy_array[0])
        if np.allclose(dE1, dE2) and wiggler_regime:
            energy_chunks = smart_split_energy(energy_array, num_cores)
        else:
            energy_chunks = np.array_split(list(energy_array), num_cores)

        results = Parallel(n_jobs=num_cores, backend="loky")(delayed(core_srwlibCalcStokesUR)((
                                                                    list_pairs,
                                                                    bl,
                                                                    eBeam,
                                                                    magFldCnt,
                                                                    h_slit_points,
                                                                    v_slit_points,
                                                                    resonant_energy,
                                                                    radiation_polarisation))
                                             for list_pairs in energy_chunks)
        
        energy_array = [calcs[1][0] for calcs in results]
        time_array = [calcs[2] for calcs in results]
        energy_chunks = [len(calcs[0]) for calcs in results]

        # Collect intensity arrays first, then concatenate all at once
        intensity_arrays = [np.asarray(calcs[0], dtype="float64") for calcs in results]
        intensity = np.concatenate(intensity_arrays, axis=0)

        # Print results
        print(">>> elapsed time:")
        for i, (t, pts, e0) in enumerate(zip(time_array, energy_chunks, energy_array), start=1):
            print(f" Core {i}: {t:.2f} s for {pts} pts (E0 = {e0:.1f} eV).")
        # energy_array = []
        # time_array = []
        # energy_chunks = []

        # k = 0
        # for calcs in results:
        #     energy_array.append(calcs[1][0])
        #     time_array.append(calcs[2])
        #     energy_chunks.append(len(calcs[0]))
        #     if k == 0:
        #         intensity = np.asarray(calcs[0], dtype="float64")
        #     else:
        #         intensity = np.concatenate((intensity, np.asarray(calcs[0], dtype="float64")), axis=0)
        #     k+=1
        # print(">>> ellapse time:")

        # for ptime in range(len(time_array)):
        #     print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")

    else:
        results = core_srwlibCalcStokesUR((energy_array,
                                          bl, 
                                          eBeam,
                                          magFldCnt, 
                                          h_slit_points,
                                          v_slit_points,
                                          resonant_energy,
                                          radiation_polarisation))
        
        intensity = np.asarray(results[0], dtype="float64")

    return intensity


def core_srwlibCalcStokesUR(args: Tuple[np.ndarray, 
                                        dict, 
                                        srwlib.SRWLPartBeam, 
                                        srwlib.SRWLMagFldC, 
                                        np.ndarray,
                                        np.ndarray,
                                        float,
                                        int]) -> Tuple[np.ndarray, float]:
    """
    Core function to calculate Stokes parameters for undulator radiation.

    Args:
        args (tuple): Tuple containing arguments:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - resonant_energy (float): Resonant energy [eV].
            - radiation_polarisation (int): Polarisation component to be extracted.

    Returns:
        Tuple[np.ndarray, float]: Tuple containing intensity data and computation time.
    """

    energy_array, bl, eBeam, magFldCnt, h_slit_points, v_slit_points, resonant_energy, radiation_polarisation = args

    tzero = time()

    try:
        arPrecPar = [0]*5   # for spectral flux vs photon energy
        arPrecPar[0] = 1    # initial UR harmonic to take into account
        arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy_array[-1]) #final UR harmonic to take into account
        arPrecPar[2] = 1.5  # longitudinal integration precision parameter
        arPrecPar[3] = 1.5  # azimuthal integration precision parameter
        arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

        npts = len(energy_array)
        stk = srwlib.SRWLStokes() 
        stk.allocate(npts, h_slit_points, v_slit_points)     
        stk.mesh.zStart = bl['distance']
        stk.mesh.eStart = energy_array[0]
        stk.mesh.eFin =   energy_array[-1]
        stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
        stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
        stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
        stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2
        und = magFldCnt.arMagFld[0]
        srwlib.srwl.CalcStokesUR(stk, eBeam, und, arPrecPar)
        # intensity = stk.arS[0:npts]
        intensity = stk.to_int(radiation_polarisation)
    except:
         raise ValueError("Error running SRW.")

    return intensity, energy_array, time()-tzero


def tc_with_srwlibCalcElecFieldSR(bl: dict, 
                          eBeam: srwlib.SRWLPartBeam, 
                          magFldCnt: srwlib.SRWLMagFldC, 
                          energy_array: np.ndarray,
                          Kh: np.ndarray,
                          Kv: np.ndarray,
                          even_harmonics: bool,
                          h_slit_points: int, 
                          v_slit_points: int, 
                          radiation_characteristic: int, 
                          radiation_dependence: int, 
                          radiation_polarisation: int,
                          parallel: bool,
                          num_cores: int=None)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the intensity tuning curve using srwlib.CalcElecFieldSR.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        Kh (np.ndarray): Array of horizontal deflection parameters for each harmonic.
        Kv (np.ndarray): Array of vertical deflection parameters for each harmonic.
        even_harmonics (bool): Whether to include even harmonics in the calculation.
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_characteristic (int): Radiation characteristic:
               0 - "Single-Electron" Intensity;
               1 - "Multi-Electron" Intensity;
               4 - "Single-Electron" Radiation Phase;
               5 - Real part of Single-Electron Electric Field;
               6 - Imaginary part of Single-Electron Electric Field.
        radiation_dependence (int): Radiation dependence, e.g., 1 for angular distribution.
               0 - vs e (photon energy or time);
               1 - vs x (horizontal position or angle);
               2 - vs y (vertical position or angle);
               3 - vs x&y (horizontal and vertical positions or angles);
               4 - vs e&x (photon energy or time and horizontal position or angle);
               5 - vs e&y (photon energy or time and vertical position or angle);
               6 - vs e&x&y (photon energy or time, horizontal and vertical positions or angles).
        radiation_polarisation (int): Polarisation component to be extracted.
               0 - Linear Horizontal;
               1 - Linear Vertical;
               2 - Linear 45 degrees;
               3 - Linear 135 degrees;
               4 - Circular Right;
               5 - Circular Left;
               6 - Total.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation.
                                   Defaults to the number of available CPU cores.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Total intensity (tuning curve), horizontal axis, and vertical axis.
    """

    nHarmMax = Kh.shape[1]

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        chunk_size = 20
        n_slices = len(energy_array)
        chunks = [(energy_array[i:i + chunk_size],
                   Kh[i:i + chunk_size, :],
                   Kv[i:i + chunk_size, :],
                   nHarmMax,
                   even_harmonics,
                   bl,
                   eBeam,
                   magFldCnt,
                   h_slit_points,
                   v_slit_points,
                   radiation_characteristic,
                   radiation_dependence,
                   radiation_polarisation,
                    ) for i in range(0, n_slices, chunk_size)]
        
        with mp.Pool() as pool:
            results = pool.map(core_tc_with_srwlibCalcElecFieldSR, chunks)
        tc = np.concatenate(results, axis=0)
    else:
        tc = core_tc_with_srwlibCalcElecFieldSR((energy_array,
                                                 Kh, 
                                                 Kv, 
                                                 nHarmMax,
                                                 even_harmonics,
                                                 bl,
                                                 eBeam,
                                                 magFldCnt,
                                                 h_slit_points,
                                                 v_slit_points,
                                                 radiation_characteristic,
                                                 radiation_dependence,
                                                 radiation_polarisation))
        
    if h_slit_points == 1 or v_slit_points == 1:
        h_axis = np.asarray([0])
        v_axis = np.asarray([0])
    else:
        h_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        v_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        
    return tc, h_axis, v_axis


def core_tc_with_srwlibCalcElecFieldSR(args):
    """
    Core function to calculate the intensity tuning curve using srwlib.CalcElecFieldSR.

    Args:
        args (tuple): Tuple containing:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - Kh (np.ndarray): Horizontal deflection parameters for each harmonic.
            - Kv (np.ndarray): Vertical deflection parameters for each harmonic.
            - nHarmMax (int): Maximum number of harmonics.
            - even_harmonics (bool): Whether to include even harmonics.
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - radiation_characteristic (int): Radiation characteristic.
            - radiation_dependence (int): Radiation dependence.
            - radiation_polarisation (int): Polarisation component.

    Returns:
        np.ndarray: Array containing the calculated intensity or flux.
    """

    energy, Kh, Kv, nHarmMax, even_harmonics, bl, eBeam, magFldCnt, \
        h_slit_points, v_slit_points, radiation_characteristic, radiation_dependence, \
            radiation_polarisation = args

    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if (nharm + 1) % 2 == 0 and even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = np.sqrt(Kh[i, nharm]**2 + Kv[i, nharm]**2)
                if deflec_param>0:
                    bl['Kv'] = Kv[i, nharm]
                    bl['Kh'] = Kh[i, nharm]                   
                    magFldCnt = set_magnetic_structure(bl, id_type='u')
                    htc[i, nharm+1], h_axis, v_axis = srwlibCalcElecFieldSR(
                                                bl, 
                                                eBeam, 
                                                magFldCnt,
                                                dE,
                                                h_slit_points=h_slit_points,
                                                v_slit_points=v_slit_points,
                                                radiation_characteristic=radiation_characteristic, 
                                                radiation_dependence=radiation_dependence,
                                                radiation_polarisation=radiation_polarisation,
                                                id_type='u',
                                                parallel=False,
                                                num_cores=1
                                                )
                    # htc[i, nharm+1] = (np.sum(flux)*(h_axis[1]-h_axis[0])*(v_axis[1]-v_axis[0]))*1E6   
    return htc


def tc_with_srwlibCalcStokesUR(bl: dict, 
                          eBeam: srwlib.SRWLPartBeam, 
                          magFldCnt: srwlib.SRWLMagFldC, 
                          energy_array: np.ndarray,
                          Kh: np.ndarray,
                          Kv: np.ndarray,
                          even_harmonics: bool,
                          h_slit_points: int, 
                          v_slit_points: int, 
                          radiation_characteristic: int, 
                          radiation_dependence: int, 
                          radiation_polarisation: int,
                          parallel: bool,
                          num_cores: int=None)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the intensity tuning curve using srwlib.CalcStokesUR.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        Kh (np.ndarray): Array of horizontal deflection parameters for each harmonic.
        Kv (np.ndarray): Array of vertical deflection parameters for each harmonic.
        even_harmonics (bool): Whether to include even harmonics in the calculation.
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_polarisation (int): Polarisation component to be extracted.
               0 - Linear Horizontal;
               1 - Linear Vertical;
               2 - Linear 45 degrees;
               3 - Linear 135 degrees;
               4 - Circular Right;
               5 - Circular Left;
               6 - Total.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation.
                                   Defaults to the number of available CPU cores.

    Returns:
        Tuple[np.ndarray]: Total intensity (tuning curve).
    """

    nHarmMax = Kh.shape[1]

    K = np.sqrt(Kh[:, 0]**2 + Kv[:, 0]**2)

    CalcStrokesEnergy = energy_array[K != 0]
    CalcStrokesEnergy /= CalcStrokesEnergy[0]*np.sqrt(2)
    CalcStrokesEnergy /= CalcStrokesEnergy[np.argmin(np.abs(CalcStrokesEnergy - 1))]

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        chunk_size = 20
        n_slices = len(energy_array)
        chunks = [(energy_array[i:i + chunk_size],
                   Kh[i:i + chunk_size, :],
                   Kv[i:i + chunk_size, :],
                   CalcStrokesEnergy,
                   nHarmMax,
                   even_harmonics,
                   bl,
                   eBeam,
                   magFldCnt,
                   h_slit_points,
                   v_slit_points,
                   radiation_characteristic,
                   radiation_dependence,
                   radiation_polarisation,
                    ) for i in range(0, n_slices, chunk_size)]
        
        with mp.Pool() as pool:
            results = pool.map(core_tc_with_srwlibCalcStokesUR, chunks)
        tc = np.concatenate(results, axis=0)
    else:
        tc = core_tc_with_srwlibCalcStokesUR((energy_array,
                                              Kh, 
                                              Kv, 
                                              CalcStrokesEnergy,
                                              nHarmMax,
                                              even_harmonics,
                                              bl,
                                              eBeam,
                                              magFldCnt,
                                              h_slit_points,
                                              v_slit_points,
                                              radiation_characteristic,
                                              radiation_dependence,
                                              radiation_polarisation))
               
    return tc

def core_tc_with_srwlibCalcStokesUR(args):
    """
    Core function to calculate the intensity tuning curve using srwlib.CalcStokesUR.

    Args:
        args (tuple): Tuple containing:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - Kh (np.ndarray): Horizontal deflection parameters for each harmonic.
            - Kv (np.ndarray): Vertical deflection parameters for each harmonic.
            - nHarmMax (int): Maximum number of harmonics.
            - even_harmonics (bool): Whether to include even harmonics.
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - radiation_characteristic (int): Radiation characteristic.
            - radiation_dependence (int): Radiation dependence.
            - radiation_polarisation (int): Polarisation component.

    Returns:
        np.ndarray: Array containing the calculated intensity or flux.
    """

    energy, Kh, Kv, CalcStrokesEnergy, nHarmMax, even_harmonics, bl, eBeam, magFldCnt, \
        h_slit_points, v_slit_points, radiation_characteristic, radiation_dependence, \
            radiation_polarisation = args

    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if (nharm + 1) % 2 == 0 and even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = np.sqrt(Kh[i, nharm]**2 + Kv[i, nharm]**2)
                if deflec_param>0:
                    bl['Kv'] = Kv[i, nharm]
                    bl['Kh'] = Kh[i, nharm]                   
                    magFldCnt = set_magnetic_structure(bl, id_type='u')
                    e_array = CalcStrokesEnergy*dE
                    flux = srwlibCalcStokesUR(
                                              bl, 
                                              eBeam, 
                                              magFldCnt,
                                              e_array,
                                              dE,
                                              h_slit_points=h_slit_points,
                                              v_slit_points=v_slit_points,
                                              radiation_polarisation=radiation_polarisation,
                                              parallel=False,
                                              num_cores=1
                                              )
                    # # RC 2024/12/04 - quick debug
                    # import matplotlib.pyplot as plt
                    # plt.plot(e_array, flux)
                    # plt.plot(dE, flux[np.where(e_array == dE)][0], "o")
                    # plt.show()
                    htc[i, nharm+1] = flux[np.where(e_array == dE)][0]  

    return htc


def tc_with_srwlibsrwl_wfr_emit_prop_multi_e(bl: dict, 
                          eBeam: srwlib.SRWLPartBeam, 
                          magFldCnt: srwlib.SRWLMagFldC, 
                          energy_array: np.ndarray,
                          Kh: np.ndarray,
                          Kv: np.ndarray,
                          even_harmonics: bool,
                          h_slit_points: int, 
                          v_slit_points: int, 
                          radiation_polarisation: int,
                          number_macro_electrons: int, 
                          aux_file_name: str,
                          parallel: bool,
                          num_cores: int=None)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the intensity tuning curve using srwlib.srwl_wfr_emit_prop_multi_e.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        Kh (np.ndarray): Array of horizontal deflection parameters for each harmonic.
        Kv (np.ndarray): Array of vertical deflection parameters for each harmonic.
        even_harmonics (bool): Whether to include even harmonics in the calculation.
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_polarisation (int): Polarisation component to be extracted.
               0 - Linear Horizontal;
               1 - Linear Vertical;
               2 - Linear 45 degrees;
               3 - Linear 135 degrees;
               4 - Circular Right;
               5 - Circular Left;
               6 - Total.
        number_macro_electrons (int): Total number of macro-electrons.
        aux_file_name (str): Auxiliary file name for saving intermediate data.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation.
                                   Defaults to the number of available CPU cores.

    Returns:
        Tuple[np.ndarray]: Total intensity (tuning curve), horizontal axis, and vertical axis.
    """

    nHarmMax = Kh.shape[1]

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        chunk_size = 20
        n_slices = len(energy_array)
        chunks = [(energy_array[i:i + chunk_size],
                   Kh[i:i + chunk_size, :],
                   Kv[i:i + chunk_size, :],
                   nHarmMax,
                   even_harmonics,
                   bl,
                   eBeam,
                   magFldCnt,
                   h_slit_points,
                   v_slit_points,
                   radiation_polarisation,
                   number_macro_electrons,
                   aux_file_name
                    ) for i in range(0, n_slices, chunk_size)]
        
        with mp.Pool() as pool:
            results = pool.map(core_tc_with_srwlibsrwl_wfr_emit_prop_multi_e, chunks)
        tc = np.concatenate(results, axis=0)
    else:
        tc = core_tc_with_srwlibsrwl_wfr_emit_prop_multi_e((energy_array,
                                                 Kh, 
                                                 Kv, 
                                                 nHarmMax,
                                                 even_harmonics,
                                                 bl,
                                                 eBeam,
                                                 magFldCnt,
                                                 h_slit_points,
                                                 v_slit_points,
                                                 radiation_polarisation,
                                                 number_macro_electrons,
                                                 aux_file_name))
        
    if h_slit_points == 1 or v_slit_points == 1:
        h_axis = np.asarray([0])
        v_axis = np.asarray([0])
    else:
        h_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        v_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        
    return tc, h_axis, v_axis


def core_tc_with_srwlibsrwl_wfr_emit_prop_multi_e(args):
    """
    Core function to calculate the intensity tuning curve using srwlib.srwl_wfr_emit_prop_multi_e.

    Args:
        args (tuple): Tuple containing:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - Kh (np.ndarray): Horizontal deflection parameters for each harmonic.
            - Kv (np.ndarray): Vertical deflection parameters for each harmonic.
            - nHarmMax (int): Maximum number of harmonics.
            - even_harmonics (bool): Whether to include even harmonics.
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - radiation_polarisation (int): Polarisation component.
            - number_macro_electrons (int): Total number of macro-electrons.
            - aux_file_name (str): Auxiliary file name for saving intermediate data.

    Returns:
        np.ndarray: Array containing the calculated intensity or flux.
    """
    energy, Kh, Kv, nHarmMax, even_harmonics, bl, eBeam, magFldCnt, \
        h_slit_points, v_slit_points, radiation_polarisation, number_macro_electrons, file_name = args
    
    htc = np.zeros((len(energy), nHarmMax+1))

    for nharm in range(nHarmMax):
        if (nharm + 1) % 2 == 0 and even_harmonics or (nharm + 1) % 2 != 0:
            for i, dE in enumerate(energy):
                deflec_param = np.sqrt(Kh[i, nharm]**2 + Kv[i, nharm]**2)
                if deflec_param>0:
                    bl['Kv'] = Kv[i, nharm]
                    bl['Kh'] = Kh[i, nharm]                   
                    magFldCnt = set_magnetic_structure(bl, id_type='u')
                     
                    htc[i, nharm+1], h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(
                                                            bl, 
                                                            eBeam,
                                                            magFldCnt,
                                                            dE,
                                                            h_slit_points=h_slit_points,
                                                            v_slit_points=v_slit_points,
                                                            radiation_polarisation=radiation_polarisation,
                                                            id_type='u',
                                                            number_macro_electrons=number_macro_electrons,
                                                            aux_file_name=file_name,
                                                            parallel=False,
                                                            num_cores=1
                                                            )  

    return htc

#***********************************************************************************
# auxiliary functions accelerator functions
#***********************************************************************************

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
    if srw_max_harmonic_number < 15:
        srw_max_harmonic_number = 15
    return srw_max_harmonic_number



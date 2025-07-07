#!/bin/python

""" 
This module provides SRW interfaced functions.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MAR/2024'
__changed__ = '07/JUL/2025'

import array
import copy
import multiprocessing as mp
import os
from time import time
from typing import  List, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants

from barc4sr.aux_energy import get_gamma, smart_split_energy

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

def set_light_source(bl: dict,
                     electron_trajectory: bool,
                     id_type: str,
                     **kwargs) -> Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]:
    """
    Set up the light source parameters including electron beam, magnetic structure, and electron trajectory.

    Args:
        bl (dict): Beamline parameters dictionary containing essential information for setup.
        electron_trajectory (bool): Whether to calculate electron trajectory.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        verbose (bool): Whether to print dialogue.
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        magfield_initial_position (float): Longitudinal position of the magnet center [m]

    Returns:
        Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]: A tuple containing the electron beam,
        magnetic structure, and electron trajectory.
    """    

    verbose = kwargs.get('verbose', True)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)
    magfield_central_position = kwargs.get('magfield_central_position', 0)

    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    if verbose: print('> Generating the electron beam ... ', end='')
    eBeam = set_electron_beam(bl,
                              id_type,
                              ebeam_initial_position=ebeam_initial_position)
    if verbose: print('completed')
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    if verbose: print('> Generating the magnetic structure ... ', end='')
    magFldCnt = set_magnetic_structure(bl, 
                                       id_type,
                                       magfield_central_position = magfield_central_position)
    if verbose: print('completed')
    # ----------------------------------------------------------------------------------
    # calculate electron trajectory
    # ----------------------------------------------------------------------------------
    if verbose: print('> Electron trajectory calculation ... ', end='')
    if electron_trajectory:
        eTraj = srwlCalcPartTraj(eBeam, magFldCnt)
    else:
        eTraj = 0
    if verbose: print('completed')

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
        tabulated_undulator_mthd (int): Method to use for generating undulator field if magnetic_measurement is provided. Defaults to 0

    Returns:
        srwlib.SRWLMagFldC: Magnetic field container.

    """
    magfield_central_position = kwargs.get('magfield_central_position', 0)

    if id_type.startswith('u'):
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
        
    if id_type.startswith('arb'):

        field_axis = bl["MagFieldDict"]['s']

        Bx = bl["MagFieldDict"]['B'][:, 0]
        By = bl["MagFieldDict"]['B'][:, 1]
        Bz = bl["MagFieldDict"]['B'][:, 2]

        arBx = srwlib.array('d', [0.0]*len(field_axis))
        arBy = srwlib.array('d', [0.0]*len(field_axis))
        arBz = srwlib.array('d', [0.0]*len(field_axis))

        for i in range(len(field_axis)):
            arBx[i] = float(Bx[i])
            arBy[i] = float(By[i])
            arBz[i] = float(Bz[i])

        range_z = field_axis[-1]-field_axis[0]
        magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[srwlib.SRWLMagFld3D(
                                        arBx, arBy, arBz,
                                        1, 1, len(field_axis), _rz=range_z)],
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
                          energy: np.array,
                          h_slit_points: int, 
                          v_slit_points: int,
                          id_type: str) -> srwlib.SRWLWfr:
    """
    Calculates the electric field for synchrotron radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy (np.array): Photon energy array (np.array) or enerfy point (float) [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), 
        bending magnet (bm) or arbitrary (arb).

    Returns:
        srwlib.SRWLWfr: Object containing the calculated wavefront
    """
    tzero = time()
    arPrecPar = [0]*7
    if id_type in ['bm', 'w', 'arb']:
        arPrecPar[0] = 2      # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    else:
        arPrecPar[0] = 1
    arPrecPar[1] = 0.001  
    arPrecPar[2] = 0     # longitudinal position to start integration (effective if < zEndInteg)
    arPrecPar[3] = 0     # longitudinal position to finish integration (effective if > zStartInteg)
    arPrecPar[4] = 50000 # Number of points for trajectory calculation
    arPrecPar[5] = 1     # Use "terminating terms" or not (1 or 0 respectively)
    arPrecPar[6] = 0     # sampling factor for adjusting nx, ny (effective if > 0)

    if isinstance(energy, int) or isinstance(energy, float):
        eStart = energy
        eFin = energy
        ne = 1
    else:
        eStart = energy[0]
        eFin = energy[-1]
        ne = len(energy)

    mesh = srwlib.SRWLRadMesh(_eStart= eStart,
                              _eFin  = eFin,
                              _ne    = ne,
                              _xStart= -bl['slitH']/2-bl['slitHcenter'],
                              _xFin  =  bl['slitH']/2-bl['slitHcenter'],
                              _nx    =  h_slit_points,
                              _yStart= -bl['slitV']/2-bl['slitVcenter'],
                              _yFin  =  bl['slitV']/2-bl['slitVcenter'],
                              _ny    =  v_slit_points,
                              _zStart=  bl['distance'])
    
    wfr = srwlib.SRWLWfr()
    wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
    wfr.mesh = mesh
    wfr.partBeam = eBeam

    return srwlib.srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar), time()-tzero

def spectral_srwlibCalcElecFieldSR(bl: dict, 
                                   eBeam: srwlib.SRWLPartBeam, 
                                   magFldCnt: srwlib.SRWLMagFldC, 
                                   energy: np.ndarray,
                                   h_slit_points: int, 
                                   v_slit_points: int,
                                   id_type: str,
                                   parallel: bool,
                                   selected_polarisations: list,
                                   number_macro_electrons: int,
                                   verbose: bool = True
                                   ):
    
    num_cores = mp.cpu_count() - 1

    if parallel:
        dE = np.diff(energy)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy[-1]>21*energy[0])
        if np.allclose(dE1, dE2) and wiggler_regime:
            energy_chunks = smart_split_energy(energy, num_cores)
        else:
            energy_chunks = np.array_split(list(energy), num_cores)
        
        results = Parallel(n_jobs=num_cores, backend="loky")(delayed(srwlibCalcElecFieldSR)(
            bl, eBeam, magFldCnt, list_pairs, h_slit_points, v_slit_points, id_type
        ) for list_pairs in energy_chunks)
        
        wfrDicts = []
        time_array = []
        energy_array = []
        energy_chunks_lens = []

        for i, (wfr, dt) in enumerate(results):
            wfrDict = unpack_srwlib_wfr(wfr, selected_polarisations, number_macro_electrons)
            wfrDicts.append(wfrDict)
            time_array.append(dt)
            if isinstance(wfrDict['energy'], np.ndarray):
                energy_array.append(wfrDict['energy'][0])
                energy_chunks_lens.append(len(wfrDict['energy']))
            else:
                energy_array.append(wfrDict['energy'])
                energy_chunks_lens.append(1)

        spectrum = concatenate_wavefronts_energy(wfrDicts)

        if verbose and not wiggler_regime:
            print(">>> elapsed time:")
            for i, (t, npts, e0) in enumerate(zip(time_array, energy_chunks_lens, energy_array)):
                print(f" Core {i+1}: {t:.2f} s for {npts} pts (E0 = {e0:.1f} eV).")

    else:
        wfr, dt = srwlibCalcElecFieldSR(bl, eBeam, magFldCnt, energy,
                                        h_slit_points, v_slit_points, id_type) 
        spectrum = unpack_srwlib_wfr(wfr, selected_polarisations, number_macro_electrons)
        if verbose:
            print(f">>> elapsed time: {dt:.2f} s for {len(energy)} pts.")
    
    return spectrum

def srwlibCalcStokesUR(bl: dict, 
                       eBeam: srwlib.SRWLPartBeam, 
                       magFldCnt: srwlib.SRWLMagFldC, 
                       energy: np.array,
                       resonant_energy: float,
                       h_slit_points: int, 
                       v_slit_points: int):
    tzero = time()

    arPrecPar = [0]*5   # for spectral flux vs photon energy
    arPrecPar[0] = 1    # initial UR harmonic to take into account
    arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy[-1]) #final UR harmonic to take into account
    arPrecPar[2] = 1.5  # longitudinal integration precision parameter
    arPrecPar[3] = 1.5  # azimuthal integration precision parameter
    if bl['slitH'] <= 1e-6 or bl['slitV'] <= 1e-6:
        arPrecPar[4] = 2
    else:
        arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

    npts = len(energy)
    stk = srwlib.SRWLStokes() 
    stk.allocate(npts, h_slit_points, v_slit_points)     
    stk.mesh.zStart = bl['distance']
    stk.mesh.eStart = energy[0]
    stk.mesh.eFin =   energy[-1]
    stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
    stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
    stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
    stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2
    und = magFldCnt.arMagFld[0]

    return srwlib.srwl.CalcStokesUR(stk, eBeam, und, arPrecPar), time()-tzero

def spectral_srwlibCalcStokesUR(bl: dict, 
                                eBeam: srwlib.SRWLPartBeam, 
                                magFldCnt: srwlib.SRWLMagFldC, 
                                energy: np.ndarray,
                                resonant_energy:float,
                                h_slit_points: int, 
                                v_slit_points: int,
                                parallel: bool,
                                selected_polarisations: list,
                                verbose: bool = True
                                ):

    num_cores = mp.cpu_count() - 1

    if parallel:
        dE = np.diff(energy)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy[-1]>21*energy[0])
        if np.allclose(dE1, dE2) and wiggler_regime:
            energy_chunks = smart_split_energy(energy, num_cores)
        else:
            energy_chunks = np.array_split(list(energy), num_cores)
        
        results = Parallel(n_jobs=num_cores, backend="loky")(delayed(srwlibCalcStokesUR)(
            bl, eBeam, magFldCnt, list_pairs, resonant_energy, h_slit_points, v_slit_points
        ) for list_pairs in energy_chunks)
        
        stkDicts = []
        time_array = []
        energy_array = []
        energy_chunks_lens = []

        for i, (stk, dt) in enumerate(results):
            stkDict = unpack_srwlib_stk(stk, selected_polarisations)
            stkDicts.append(stkDict)
            time_array.append(dt)
            if isinstance(stkDict['energy'], np.ndarray):
                energy_array.append(stkDict['energy'][0])
                energy_chunks_lens.append(len(stkDict['energy']))
            else:
                energy_array.append(stkDict['energy'])
                energy_chunks_lens.append(1)

        spectrum = concatenate_stokes_energy(stkDicts)

        if verbose and not wiggler_regime:
            print(">>> elapsed time:")
            for i, (t, npts, e0) in enumerate(zip(time_array, energy_chunks_lens, energy_array)):
                print(f" Core {i+1}: {t:.2f} s for {npts} pts (E0 = {e0:.1f} eV).")

    else:
        stk, dt = srwlibCalcStokesUR(bl, eBeam, magFldCnt, energy, resonant_energy,
                                     h_slit_points, v_slit_points) 
        spectrum = unpack_srwlib_stk(stk, selected_polarisations)

        if verbose:
            print(f">>> elapsed time: {dt:.2f} s for {len(energy)} pts.")

    return spectrum

def unpack_srwlib_wfr(wfr: srwlib.SRWLWfr, selected_polarisations: list, number_macro_electrons: int) -> dict:
    """
    Unpacks SRW wavefront data.

    Parameters:
        wfr (srwlib.SRWLWfr): The SRW wavefront object containing the simulated electric field.
        selected_polarisations (list or str): List of polarisations to export. Can be a single
                         string or a list of strings. Accepted values include: 'LH', 'LV', 
                         'L45', 'L135', 'CR', 'CL', 'T'.
        number_macro_electrons (int): Number of macro electrons. 

    Returns:
        dict: A dictionary containing the computed axes and intensities for selected
              polarisations.
    """
    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings.")
    
    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    if wfr.mesh.ne > 1:
        wfrDict = {'energy': np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne)}
        energy_content = True
    else:
        wfrDict = {'energy': wfr.mesh.eStart}
        energy_content = False

    if wfr.mesh.nx > 1 or wfr.mesh.ny > 1:
        wfrDict['axis'] = {'x': np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx),
                           'y': np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)}
        spatial_distribution = True
    else:
        wfrDict['axis'] = {'x': 0, 'y': 0}
        spatial_distribution = False
    
    if energy_content and spatial_distribution:
        _inDepType = 6
    elif energy_content and not spatial_distribution:
        _inDepType = 0
    elif not energy_content and spatial_distribution:
        _inDepType = 3

    all_polarisations = ['LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T']
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return unpack_srwlib_wfr(wfr, ['T'], number_macro_electrons)
    
    _inIntType = int(number_macro_electrons)

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index
        arInt = srwlib.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny*wfr.mesh.ne)
        srwlib.srwl.CalcIntFromElecField(arInt, wfr, _inPol, _inIntType, _inDepType,
                                         wfr.mesh.eStart,
                                        (wfr.mesh.xStart+wfr.mesh.xFin)/2,
                                        (wfr.mesh.yStart+wfr.mesh.yFin)/2)
        wfrDict.update({polarisation:np.asarray(arInt, dtype="float64").reshape((wfr.mesh.ne, wfr.mesh.ny, wfr.mesh.nx))})

    return wfrDict

def concatenate_wavefronts_energy(wfrDicts: list) -> dict:
    """
    Concatenate multiple wfrDict dictionaries along the energy axis.
    
    Parameters:
        wfrDicts (list): List of wavefront dictionaries as returned by unpack_srwlib_wfr,
                         computed over different energy ranges.

    Returns:
        dict: Concatenated dictionary with combined energy axis.
    """
    if not wfrDicts:
        raise ValueError("Input list is empty.")

    energy_all = np.concatenate([wfr['energy'] if np.ndim(wfr['energy']) else [wfr['energy']] for wfr in wfrDicts])
    
    concatenated = {
        'energy': energy_all,
        'axis': wfrDicts[0]['axis'],
    }

    polarisations = [k for k in wfrDicts[0].keys() if k not in ['energy', 'axis']]

    for pol in polarisations:
        concatenated[pol] = np.concatenate([wfr[pol] for wfr in wfrDicts], axis=0)

    return concatenated

def unpack_srwlib_stk(stk: srwlib.SRWLStokes, selected_polarisations: list) -> dict:
    """
    Unpacks SRW stokes data.

    Parameters:
        stk (srwlib.SRWLStokes): The SRW stokes object containing the simulated electric field.
        selected_polarisations (list or str): List of polarisations to export. Can be a single
                         string or a list of strings. Accepted values include: 'LH', 'LV', 
                         'L45', 'L135', 'CR', 'CL', 'T'.
        number_macro_electrons (int): Number of macro electrons. 

    Returns:
        dict: A dictionary containing the computed axes and intensities for selected
              polarisations.
    """

    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings.")
    
    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    stkDict = {'energy': np.linspace(stk.mesh.eStart, stk.mesh.eFin, stk.mesh.ne)}

    stkDict['axis'] = {'x': stk.mesh.xFin-stk.mesh.xStart,
                       'y': stk.mesh.yFin-stk.mesh.yStart}

    all_polarisations = ['LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T']
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return unpack_srwlib_stk(stk, ['T'])
    
    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index
        stkDict.update({polarisation:np.asarray(stk.to_int(_inPol), dtype="float64")})

    return stkDict

def concatenate_stokes_energy(stkDicts: list) -> dict:
    """
    Concatenate multiple stkDict dictionaries along the energy axis.

    Parameters:
        stkDicts (list): List of Stokes dictionaries as returned by unpack_srwlib_stk,
                         computed over different energy ranges.

    Returns:
        dict: Concatenated dictionary with combined energy axis.
    """
    if not stkDicts:
        raise ValueError("Input list is empty.")

    energy_all = np.concatenate([stk['energy'] if np.ndim(stk['energy']) else [stk['energy']] for stk in stkDicts])
    
    concatenated = {
        'energy': energy_all,
        'axis': stkDicts[0]['axis'],
    }

    polarisations = [k for k in stkDicts[0].keys() if k not in ['energy', 'axis']]
    for pol in polarisations:
        concatenated[pol] = np.concatenate([stk[pol] for stk in stkDicts], axis=0)

    return concatenated
# def srwlibsrwl_wfr_emit_prop_multi_e:
# def srwlibsrwl_wfr_emit_prop_multi_e_spectral:

def srwlibCalcPowDenSR(bl: dict, 
                       eBeam: srwlib.SRWLPartBeam, 
                       magFldCnt: srwlib.SRWLMagFldC, 
                       h_slit_points: int, 
                       v_slit_points: int) -> srwlib.SRWLStokes:
    """
    Calculates the power density.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.

    Returns:
        srwlib.SRWLStokes: Object containing the calculated wavefront
    """
    arPrecPar = [0]*5     # for power density
    arPrecPar[0] = 1.5    # precision factor
    arPrecPar[1] = 1      # power density computation method (1- "near field", 2- "far field")
    arPrecPar[2] = 0.0    # initial longitudinal position (effective if arPrecPar[2] < arPrecPar[3])
    arPrecPar[3] = 0.0    # final longitudinal position (effective if arPrecPar[2] < arPrecPar[3])
    arPrecPar[4] = 50000  # number of points for (intermediate) trajectory calculation

    stk = srwlib.SRWLStokes() 
    stk.allocate(1, h_slit_points, v_slit_points)     
    stk.mesh.zStart = bl['distance']
    stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
    stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
    stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
    stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2

    return srwlib.srwl.CalcPowDenSR(stk, eBeam, 0, magFldCnt, arPrecPar)



# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------


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

if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
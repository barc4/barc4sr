#!/bin/python

""" 
This module provides SRW interfaced functions.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MAR/2024'
__changed__ = '12/JUL/2025'

import copy
import multiprocessing as mp
import os
from time import time
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle
from wofry.propagator.propagator import (
    PropagationElements,
    PropagationManager,
    PropagationParameters,
)
from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofryimpl.propagator.util.tally import TallyCoherentModes
from wofryimpl.propagator.util.undulator_coherent_mode_decomposition_1d import (
    UndulatorCoherentModeDecomposition1D,
)

from barc4sr.aux_energy import energy_wavelength, get_gamma, smart_split_energy

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
                     ebeam_initial_condition: list = 6*[0],
                     verbose: bool = True,
                     ) -> Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]:
    """
    Set up the light source parameters including electron beam, magnetic structure, and electron trajectory.

    Args:
        bl (dict): Beamline parameters dictionary containing essential information for setup.
        electron_trajectory (bool): Whether to calculate electron trajectory.

    Optional Args (kwargs):
        verbose (bool): Whether to print dialogue.
        ebeam_initial_condition (float): Electron beam initial condition from the 6D phase space (posX, posY, posZ, angX, angY, Energy)

    Returns:
        Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]: A tuple containing the electron beam,
        magnetic structure, and electron trajectory.
    """    
    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    if verbose: print('> Generating the electron beam ... ', end='')
    eBeam = set_electron_beam(bl,
                              ebeam_initial_condition=ebeam_initial_condition)
    if verbose: print('completed')
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    if verbose: print('> Generating the magnetic structure ... ', end='')
    magFldCnt = set_magnetic_structure(bl)
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
                      ebeam_initial_condition: list = 6*[0]) -> srwlib.SRWLPartBeam:
    """
    Set up the electron beam parameters.

    Parameters:
        bl (dict): Dictionary containing beamline parameters.

    Optional Args (kwargs):
        ebeam_initial_condition (float): Electron beam initial condition from the 6D phase space (posX, posY, posZ, angX, angY, Energy)

    Returns:
        srwlib.SRWLPartBeam: Electron beam object initialized with specified parameters.

    """

    eBeam = srwlib.SRWLPartBeam()
    eBeam.Iavg = bl['ElectronCurrent']                 # average current [A]
    eBeam.partStatMom1.x = ebeam_initial_condition[0]  # initial transverse positions [m]
    eBeam.partStatMom1.y = ebeam_initial_condition[1]
    if bl['Class'].startswith('u'):
        eBeam.partStatMom1.z = - bl['PeriodID'] * (bl['NPeriods'] + 4) / 2
    else:
        eBeam.partStatMom1.z = ebeam_initial_condition[2]
    eBeam.partStatMom1.xp = ebeam_initial_condition[3]
    eBeam.partStatMom1.yp = ebeam_initial_condition[4]
    if ebeam_initial_condition[5] == 0:
        eBeam.partStatMom1.gamma = get_gamma(bl['ElectronEnergy'])
    else:
        eBeam.partStatMom1.gamma = get_gamma(ebeam_initial_condition[5])

    sigX = bl['ElectronBeamSizeH']         # horizontal RMS size of e-beam [m]
    sigXp = bl['ElectronBeamDivergenceH']  # horizontal RMS angular divergence [rad]
    sigY = bl['ElectronBeamSizeV']         # vertical RMS size of e-beam [m]
    sigYp = bl['ElectronBeamDivergenceV']  # vertical RMS angular divergence [rad]
    sigEperE = bl['ElectronEnergySpread']  

    # 2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX * sigX   # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0             # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp * sigXp # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY * sigY   # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0             # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp * sigYp # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE * sigEperE  # <(E-<E>)^2>/<E>^2

    return eBeam

def set_magnetic_structure(bl: dict) -> srwlib.SRWLMagFldC:
    """
    Sets up the magnetic field container.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        magfield_central_position (float): Longitudinal position of the magnet center [m]

    Returns:
        srwlib.SRWLMagFldC: Magnetic field container.

    """

    if bl['Class'].startswith('u'):
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
                                        _arZc=srwlib.array('d', [bl['MagFieldCenter']]))
    if bl['Class'].startswith('bm'):

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
                                       _arZc=srwlib.array('d', [bl['MagFieldCenter']]))
        
    if bl['Class'].startswith('arb'):

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
                                        _arZc=srwlib.array('d', [bl['MagFieldCenter']]))

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
                          partTraj: srwlib.SRWLPrtTrj) -> srwlib.SRWLWfr:
    """
    Calculates the electric field for synchrotron radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy (np.array): Photon energy array (np.array) or enerfy point (float) [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        bending magnet (bm) or arbitrary (arb).

    Returns:
        srwlib.SRWLWfr: Object containing the calculated wavefront
    """
    tzero = time()
    arPrecPar = [0]*7
    if bl['Class'] in ['bm', 'w', 'arb']:
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

    return srwlib.srwl.CalcElecFieldSR(wfr, partTraj, magFldCnt, arPrecPar), time()-tzero

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
                                   ) -> dict:
    """
    Calculates the electric field spectrum over an energy array.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy (np.ndarray): Photon energy array [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        id_type (str): Type of magnetic structure, e.g., undulator (u), wiggler (w), bending magnet (bm), or arbitrary (arb).
        parallel (bool): Whether to compute in parallel across available CPU cores.
        selected_polarisations (list): List of polarisations to extract.
        number_macro_electrons (int): Number of macro electrons used in the calculation.
        verbose (bool, optional): Whether to print elapsed time and status messages.

    Returns:
        dict: Dictionary containing the computed wavefront data for each selected polarisation over the energy range.
    """

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
                       v_slit_points: int) -> srwlib.SRWLStokes:
    """
    Calculates the undulator radiation Stokes parameters over an energy array.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy (np.array): Photon energy array [eV].
        resonant_energy (float): Resonant energy of the undulator [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.

    Returns:
        tuple: (srwlib.SRWLStokes, float) containing the calculated Stokes object and elapsed computation time [s].
    """
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
                                ) -> dict:
    """
    Calculates the Stokes parameters spectrum for undulator radiation over an energy array.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy (np.ndarray): Photon energy array [eV].
        resonant_energy (float): Resonant energy of the undulator [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        parallel (bool): Whether to compute in parallel across available CPU cores.
        selected_polarisations (list): List of polarisations to extract.
        verbose (bool, optional): Whether to print elapsed time and status messages.

    Returns:
        dict: Dictionary containing the computed Stokes data for each selected polarisation over the energy range.
    """

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
    stk.mesh.xStart =  -bl['slitH']/2-bl['slitHcenter']
    stk.mesh.xFin =     bl['slitH']/2-bl['slitHcenter']
    stk.mesh.yStart = -bl['slitV']/2-bl['slitVcenter']
    stk.mesh.yFin =    bl['slitV']/2-bl['slitVcenter']

    return srwlib.srwl.CalcPowDenSR(stk, eBeam, 0, magFldCnt, arPrecPar)

#***********************************************************************************
# WOFRY interface functions
#***********************************************************************************

def wofrySourceCMD(bl: dict, energy: float, scan_direction: str, **kwargs) -> UndulatorCoherentModeDecomposition1D:
    """
    Calculates the 1D coherent mode decomposition source using WOFRY.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        energy (float): Photon energy [eV].
        scan_direction (str): Direction of the scan, either 'H' (horizontal) or 'V' (vertical).

    Optional Args (kwargs):
        magnification_forward (float): Forward magnification factor. Defaults to 100.
        magnification_backward (float): Backward magnification factor. Defaults to 1 / magnification_forward.
        distance_to_screen (float): Distance from source to screen [m]. Defaults to magnification_forward.
        nsigma (float): Number of standard deviations defining the extent of the spatial domain. Defaults to 10.
        number_of_points (int): Number of points for spatial sampling. If None, it is computed automatically.

    Returns:
        UndulatorCoherentModeDecomposition1D: Object containing the coherent mode decomposition result.
    """

    magnification_forward = kwargs.get('magnification_forward', 100)
    magnification_backward = kwargs.get('magnification_backward', 1/magnification_forward)
    distance_to_screen = kwargs.get('magnification_forward', 100)
    nsigma = kwargs.get('nsigma', 10)
    number_of_points = kwargs.get('number_of_points', None)

    wavelength = energy_wavelength(energy, 'eV')

    sigma_up = 0.69*np.sqrt(wavelength/(bl['NPeriods']*bl['PeriodID']))
    sigma = np.sqrt(sigma_up**2 + bl[f'ElectronBeamDivergence{scan_direction.upper()}']**2)

    abscissas_interval = (np.tan(sigma / 2) * distance_to_screen * 2)*nsigma*magnification_backward

    if number_of_points is None:
        number_of_points = int(1.25*nsigma*(sigma**2)* distance_to_screen/wavelength)

    coherent_mode_decomposition = UndulatorCoherentModeDecomposition1D(
        electron_energy=bl['ElectronEnergy'],
        electron_current=bl['ElectronCurrent'],
        undulator_period=bl['PeriodID'],
        undulator_nperiods=bl['NPeriods'],
        K=np.sqrt(bl['Kv']**2 + bl['Kh']**2),
        photon_energy=energy,
        abscissas_interval=abscissas_interval,
        number_of_points=number_of_points,
        distance_to_screen=distance_to_screen,
        magnification_x_forward=magnification_forward,
        magnification_x_backward=magnification_backward,
        scan_direction=scan_direction.upper(),
        sigmaxx=bl[f'ElectronBeamSize{scan_direction.upper()}'],
        sigmaxpxp=bl[f'ElectronBeamDivergence{scan_direction.upper()}'],
        useGSMapproximation=False,
        e_energy_dispersion_flag=1,
        e_energy_dispersion_sigma_relative=bl['ElectronEnergySpread'],
        e_energy_dispersion_interval_in_sigma_units=6,
        e_energy_dispersion_points=11)
    
    cmd = coherent_mode_decomposition.calculate()

    return coherent_mode_decomposition


def wofrySlitCMD(src_cmd, window, observation_point, nmodes=-1):

    ff_dist = 100
    angular_window = 2 * np.arctan(window / (2 * observation_point))
    ff_window = np.tan(angular_window/2)*ff_dist*2

    if nmodes == -1:
        nmodes = len(src_cmd.eigenvalues)-1

    tally = TallyCoherentModes()

    for current_mode_index in range(nmodes):

        input_wavefront = src_cmd.get_eigenvector_wavefront(current_mode_index).duplicate()

        optical_element = WOSlit1D(boundary_shape=Rectangle(-ff_window/2, ff_window/2,
                                                            -ff_window/2, ff_window/2))
        propagation_elements = PropagationElements()
        beamline_element = BeamlineElement(optical_element=optical_element,
                                           coordinates=ElementCoordinates(p=ff_dist, q=0, 
                                                                          angle_radial=np.radians(0), 
                                                                          angle_azimuthal=np.radians(0)))
        propagation_elements.add_beamline_element(beamline_element)

        propagation_parameters = PropagationParameters(wavefront=input_wavefront,
                                                       propagation_elements=propagation_elements)
        propagation_parameters.set_additional_parameters('magnification_x', 100)

        propagator = PropagationManager.Instance()
        try:
            propagator.add_propagator(FresnelZoom1D())
        except:
            pass
        tally.append(propagator.do_propagation(propagation_parameters=propagation_parameters,
                                               handler_name='FRESNEL_ZOOM_1D'))

    return tally


#***********************************************************************************
# auxiliary functions 
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
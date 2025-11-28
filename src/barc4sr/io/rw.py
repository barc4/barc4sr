# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Read/Write helpers for barc4sr results in HDF5 format.
"""

from __future__ import annotations

import os
import pickle
from array import array
from copy import copy, deepcopy

import h5py as h5
import numpy as np
import scipy.integrate as integrate
from scipy.constants import physical_constants

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

CHARGE = physical_constants["atomic unit of charge"][0]

# ---------------------------------------------------------------------------
# electron trajectory
# ---------------------------------------------------------------------------

def write_electron_trajectory(file_name:str, eTraj: srwlib.SRWLPrtTrj) -> dict:
    """
    Saves electron trajectory data to an HDF5 file and returns a dictionary containing the trajectory data.

    This function processes the trajectory data from an `SRWLPrtTrj` object and stores it in both an HDF5 file 
    and a Python dictionary. 

    Parameters:
        file_name (str): Base file path for saving the trajectory data. The data will be saved 
                         in a file with the suffix '_eTraj.h5'.
        eTraj (SRWLPrtTrj): SRW library object containing the electron trajectory data. The object must include:
            - `arX`: Array of horizontal positions [m].
            - `arXp`: Array of horizontal relative velocities (trajectory angles) [rad].
            - `arY`: Array of vertical positions [m].
            - `arYp`: Array of vertical relative velocities (trajectory angles) [rad].
            - `arZ`: Array of longitudinal positions [m].
            - `arZp`: Array of longitudinal relative velocities (trajectory angles) [rad].
            - `arBx` (optional): Array of horizontal magnetic field components [T].
            - `arBy` (optional): Array of vertical magnetic field components [T].
            - `arBz` (optional): Array of longitudinal magnetic field components [T].
            - `np`: Number of trajectory points.
            - `ctStart`: Start value of the independent variable (c*t) for the trajectory [m].
            - `ctEnd`: End value of the independent variable (c*t) for the trajectory [m].

    Returns:
        dict: A dictionary containing the trajectory data with the following keys:
              - "ct": List of time values corresponding to the trajectory points.
              - "X", "Y", "Z": Lists of positions in the respective axes.
              - "Xp", "Yp", "Zp": Lists of velocity components (trajectory angles) in the respective axes.
              - "Bx", "By", "Bz" (optional): Lists of magnetic field components in the respective axes, if present.
    """

    eTrajDict = {"eTraj":{
        "ct": [],
        "X": [],
        "Xp": [],
        "Y": [],
        "Yp": [],
        "Z": [],
        "Zp": [],
    }}

    if hasattr(eTraj, 'arBx'):
        eTrajDict["eTraj"]["Bx"] = []
    if hasattr(eTraj, 'arBy'):
        eTrajDict["eTraj"]["By"] = []
    if hasattr(eTraj, 'arBz'):
        eTrajDict["eTraj"]["Bz"] = []

    if file_name is not None:
        with h5.File(f"{file_name}_eTraj.h5", "w") as f:
            group = f.create_group("XOPPY_ETRAJ")
            intensity_group = group.create_group("eTraj")
            
            intensity_group.create_dataset("ct", data=np.zeros(eTraj.np))
            intensity_group.create_dataset("X", data=eTraj.arX)
            intensity_group.create_dataset("Xp", data=eTraj.arXp)
            intensity_group.create_dataset("Y", data=eTraj.arY)
            intensity_group.create_dataset("Yp", data=eTraj.arYp)
            intensity_group.create_dataset("Z", data=eTraj.arZ)
            intensity_group.create_dataset("Zp", data=eTraj.arZp)
            if hasattr(eTraj, 'arBx'):
                intensity_group.create_dataset("Bx", data=eTraj.arBx)
            if hasattr(eTraj, 'arBy'):
                intensity_group.create_dataset("By", data=eTraj.arBy)
            if hasattr(eTraj, 'arBz'):
                intensity_group.create_dataset("Bz", data=eTraj.arBz)

    eTrajDict["eTraj"]["ct"] = np.zeros(eTraj.np)
    eTrajDict["eTraj"]["X"] = np.asarray(eTraj.arX)
    eTrajDict["eTraj"]["Xp"] = np.asarray(eTraj.arXp)
    eTrajDict["eTraj"]["Y"] = np.asarray(eTraj.arY)
    eTrajDict["eTraj"]["Yp"] = np.asarray(eTraj.arYp)
    eTrajDict["eTraj"]["Z"] = np.asarray(eTraj.arZ)
    eTrajDict["eTraj"]["Zp"] = np.asarray(eTraj.arZ)
    eTrajDict["eTraj"]["Bx"] = np.asarray(eTraj.arBx)
    eTrajDict["eTraj"]["By"] = np.asarray(eTraj.arBy)
    eTrajDict["eTraj"]["Bz"] = np.asarray(eTraj.arBz)

    return eTrajDict
    

def read_electron_trajectory(file_path: str) -> dict:
    """
    Reads SRW electron trajectory data from a .h5 file (XOPPY_ETRAJ format).

    Args:
        file_path (str): The path to the .h5 file containing electron trajectory data.

    Returns:
        dict: A dictionary where keys are the column names (ct, X, Xp, Y, Yp, Z, Zp, Bx, By, Bz),
            and values are lists containing the corresponding column data from the file.
    """
    result = {"eTraj": {}}

    with h5.File(file_path, "r") as f:
        try:
            trajectory_group = f["XOPPY_ETRAJ"]["eTraj"]
        except KeyError:
            raise ValueError(f"Invalid file structure: {file_path} does not contain 'XOPPY_ETRAJ/eTraj'.")

        # Read datasets
        for key in trajectory_group.keys():
            result["eTraj"][key] = trajectory_group[key][:].tolist()

    return result


def read_electron_trajectory_dat(file_path: str) -> dict:
    """
    Reads SRW electron trajectory data from a .dat file (SRW native format).

    Args:
        file_path (str): The path to the .dat file containing electron trajectory data.

    Returns:
        dict: A dictionary where keys are the column names extracted from the header
            (ct, X, Xp, Y, Yp, Z, Zp, Bx, By, Bz),
            and values are lists containing the corresponding column data from the file.
    """
    data = []
    header = None
    with open(file_path, 'r') as file:
        header_line = next(file).strip()
        header = [col.split()[0] for col in header_line.split(',')]
        header[0] = header[0].replace("#","")
        for line in file:
            values = line.strip().split('\t')
            values = [float(value) if value != '' else None for value in values]
            data.append(values)
            
    eTrajDict = {}
    for i, key in enumerate(header):
        eTrajDict[key] = np.asarray([row[i] for row in data])

    return eTrajDict

# ---------------------------------------------------------------------------
# Wavefront
# ---------------------------------------------------------------------------
   
def write_wavefront(file_name: str, wfr: srwlib.SRWLWfr, selected_polarisations: list, 
                    number_macro_electrons: int, propagation_distance: float=None) -> dict:
    """
    Writes wavefront data (intensity, phase, and wavefront object) to an HDF5 file.

    Parameters:
        file_name (str): Base file path for saving the wavefront data. The data will be stored
                         in a file named '<file_name>_undulator_wfr.h5'.
        wfr (srwlib.SRWLWfr): The SRW wavefront object containing the simulated electric field.
        selected_polarisations (list or str): List of polarisations to export. Can be a single
                         string or a list of strings. Accepted values include: 'LH', 'LV', 
                         'L45', 'L135', 'CR', 'CL', 'T'.
        number_macro_electrons (int): Number of macro electrons.

    Returns:
        dict: A dictionary containing the wavefront object, computed axes, intensities for selected
              polarisations, and phase image.
    """

    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings.")
    
    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    wfr_qpt = deepcopy(wfr)
    wfrDict = {'wfr': wfr}

    wfrDict.update({'axis':{'x': np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx),
                            'y': np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)}})

    all_polarisations = ['LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T']
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return write_wavefront(file_name, wfr, ['T'], number_macro_electrons)
    
    if propagation_distance is None:
        Rx, Ry = wfr.Rx, wfr.Ry
    else:
        Rx, Ry = propagation_distance, propagation_distance

    wfrDict.update({'energy':wfr.mesh.eStart})
    wfrDict.update({'intensity':{}})
    wfrDict.update({'phase':{}})
    wfrDict.update({'Rx':Rx, 'Ry':Ry})

    _inIntType = int(number_macro_electrons)
    _inDepType = 3

    # X, Y = np.meshgrid(wfrDict['axis']['x'], wfrDict['axis']['y'])
    # spherical_phase = Rx - np.sqrt(Rx**2 - X**2 - (Rx/Ry)**2 * Y**2)
    # amplitude_transmission = np.ones((wfr.mesh.ny, wfr.mesh.nx), dtype='float64')
    # arTr = np.empty((2 * wfr.mesh.nx * wfr.mesh.ny), dtype=spherical_phase.dtype)
    # arTr[0::2] = np.reshape(amplitude_transmission,(wfr.mesh.nx*wfr.mesh.ny))
    # arTr[1::2] = np.reshape(-spherical_phase,(wfr.mesh.nx*wfr.mesh.ny))
    # spherical_wave = srwlib.SRWLOptT(wfr.mesh.nx, wfr.mesh.ny, 
    #                                  wfrDict['axis']['x'][-1]-wfrDict['axis']['x'][0],
    #                                  wfrDict['axis']['y'][-1]-wfrDict['axis']['y'][0],
    #                                  _arTr=arTr, _extTr=1, _Fx=Rx, _Fy=Ry, _x=0, _y=0)

    quadratic_phase_term = srwlib.SRWLOptL(_Fx=Rx, _Fy=Ry)
    pp_spherical_wave =  [0, 0, 1.0, 1, 0, 1., 1., 1., 1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    OE = [quadratic_phase_term]
    PP = [pp_spherical_wave]
    
    optBL = srwlib.SRWLOptC(OE, PP)
    srwlib.srwl.PropagElecField(wfr_qpt, optBL)

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index

        arInt = array('f', [0]*wfr_qpt.mesh.nx*wfr_qpt.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arInt, wfr_qpt, _inPol, _inIntType, _inDepType, wfr_qpt.mesh.eStart, 0, 0)
        wfrDict['intensity'].update({polarisation:np.asarray(arInt, dtype="float64").reshape((wfr_qpt.mesh.ny, wfr_qpt.mesh.nx))})

        arPh = array('d', [0]*wfr_qpt.mesh.nx*wfr_qpt.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arPh, wfr_qpt, _inPol, 4, _inDepType, wfr_qpt.mesh.eStart, 0, 0)
        phase = np.asarray(arPh, dtype="float64").reshape((wfr_qpt.mesh.ny, wfr_qpt.mesh.nx))
        wfrDict['phase'].update({polarisation:phase})

    if file_name is not None:
        with h5.File(f'{file_name}_undulator_wfr.h5', 'w') as f:
            group = f.create_group('XOPPY_WAVEFRONT')

            group.create_dataset('axis_x', data=wfrDict['axis']['x'] * 1e3)  # mm
            group.create_dataset('axis_y', data=wfrDict['axis']['y'] * 1e3)

            intensity_group = group.create_group('Intensity')
            for pol, img in wfrDict['intensity'].items():
                intensity_group.create_dataset(pol, data=img)

            phase_group = group.create_group('Phase')
            for pol, img in wfrDict['phase'].items():
                phase_group.create_dataset(pol, data=img)

            wfr_pickled = pickle.dumps(wfr)
            group.create_dataset('wfr', data=np.void(wfr_pickled))
    
    return wfrDict

def read_wavefront(file_name: str) -> dict:
    """
    Reads wavefront data from an HDF5 file and reconstructs the full wavefront dictionary.

    Parameters:
        file_name (str): Path to the HDF5 file containing wavefront data.

    Returns:
        dict: Dictionary with keys:
            - 'wfr': the SRW wavefront object (unpickled).
            - 'axis': dict with 'x' and 'y' numpy arrays (in meters).
            - 'energy': photon energy (float).
            - 'Rx': curvature radius in x (float).
            - 'Ry': curvature radius in y (float).
            - 'intensity': dict with polarisation labels as keys and 2D numpy arrays as values.
            - 'phase': dict with polarisation labels as keys and 2D numpy arrays as values.
    """
    if not (file_name.endswith("h5") or file_name.endswith("hdf5")):
        raise ValueError("Only HDF5 format supported for this function.")

    with h5.File(file_name, "r") as f:
        group = f["XOPPY_WAVEFRONT"]

        x = group["axis_x"][()] * 1e-3  # back to meters
        y = group["axis_y"][()] * 1e-3

        intensity = {}
        for pol in group["Intensity"]:
            intensity[pol] = group["Intensity"][pol][()]

        phase = {}
        for pol in group["Phase"]:
            phase[pol] = group["Phase"][pol][()]

        wfr = pickle.loads(group["wfr"][()])

    Rx = getattr(wfr, "Rx", None)
    Ry = getattr(wfr, "Ry", None)
    energy = getattr(wfr.mesh, "eStart", None)

    return {
        "wfr": wfr,
        "axis": {
            "x": x,
            "y": y,
        },
        "energy": energy,
        "Rx": Rx,
        "Ry": Ry,
        "intensity": intensity,
        "phase": phase
    }

# ---------------------------------------------------------------------------
# Power density
# ---------------------------------------------------------------------------

def write_power_density(file_name: str, stks: srwlib.SRWLStokes, selected_polarisations: list) -> dict:
    """
    Writes power density data () to an HDF5 file.

    Parameters:
        file_name (str): Base file path for saving the wavefront data. The data will be stored
                         in a file named '<file_name>_undulator_wfr.h5'.
        wfr (srwlib.SRWLStokes): The SRW Stokes object containing the simulated power density.
        selected_polarisations (list or str): List of polarisations to export. Can be a single
                         string or a list of strings. Accepted values include: 'LH', 'LV', 
                         'L45', 'L135', 'CR', 'CL', 'T'.

    Returns:
        dict: A dictionary containing the wavefront object, computed axes, intensities for selected
              polarisations, and phase image.
    """
    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings.")
    
    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    pwrDict = {}

    pwrDict.update({'axis':{'x': np.linspace(stks.mesh.xStart, stks.mesh.xFin, stks.mesh.nx),
                            'y': np.linspace(stks.mesh.yStart, stks.mesh.yFin, stks.mesh.ny)}})

    all_polarisations = ['LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T']
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return write_wavefront(file_name, stks, selected_polarisations=['T'])
    
    dx = (pwrDict['axis']['x'][1]-pwrDict['axis']['x'][0])*1E3
    dy = (pwrDict['axis']['y'][1]-pwrDict['axis']['y'][0])*1E3

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index
        pow_map = np.reshape(stks.to_int(_inPol), (stks.mesh.ny, stks.mesh.nx))
        cum_pow = pow_map.sum()*dx*dy
        pwrDict.update({polarisation:{'map': pow_map,
                                      'integrated': cum_pow,
                                      'peak': pow_map.max()}})

    if file_name is not None:
        with h5.File(f'{file_name}_power_density.h5', 'w') as f:
            group = f.create_group('XOPPY_POWERDENSITY')
            group.create_dataset('axis_x', data=pwrDict['axis']['x'] * 1e3)  # mm
            group.create_dataset('axis_y', data=pwrDict['axis']['y'] * 1e3)  # mm

            for pol in selected_polarisations:
                pol_group = group.create_group(pol)
                pol_group.create_dataset('map', data=pwrDict[pol]['map'])
                pol_group.attrs['integrated'] = pwrDict[pol]['integrated']
                pol_group.attrs['peak'] = pwrDict[pol]['peak']

    return pwrDict

def read_power_density(file_name: str) -> dict:
    """
    Reads power density data from an HDF5 file and reconstructs the power dictionary.

    Parameters:
        file_name (str): Path to the HDF5 file containing power density data.

    Returns:
        dict: Dictionary with keys:
            - 'axis': dict with 'x' and 'y' numpy arrays (in meters).
            - <polarisation>: for each polarisation, a dict with:
                - 'map': 2D numpy array of power density.
                - 'integrated': scalar total power [W].
                - 'peak': scalar peak power density [W/m^2].
    """
    if not (file_name.endswith("h5") or file_name.endswith("hdf5")):
        raise ValueError("Only HDF5 format supported for this function.")

    with h5.File(file_name, "r") as f:
        group = f["XOPPY_POWERDENSITY"]

        x = group["axis_x"][()] * 1e-3  # back to meters
        y = group["axis_y"][()] * 1e-3

        pwrDict = {"axis": {"x": x, "y": y}}

        for pol in group:
            if pol.startswith("axis_"):
                continue  # skip axes

            pol_group = group[pol]
            pwrDict[pol] = {
                "map": pol_group["map"][()],
                "integrated": pol_group.attrs["integrated"],
                "peak": pol_group.attrs["peak"],
            }

    return pwrDict

# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def write_spectrum(file_name: str, spectrum: dict) -> dict:
    """
    Saves computed spectrum data to an HDF5 file and returns a processed spectrum dictionary.

    This function processes the input `spectrum` dictionary to compute:
        - Flux [ph/s/0.1%bw]
        - Spectral power [W/eV]
        - Cumulated power [W] (integrated from minimum energy up to each point)
        - Integrated power [W] (total power over the entire spectrum)

    The data is saved in an HDF5 file under the 'XOPPY_SPECTRUM/Spectrum' group, with one subgroup per polarisation
    containing the computed arrays.

    Parameters:
        file_name (str): Base file path for saving the spectrum data. The data will be stored
                         in a file named '<file_name>_spectrum.h5'.
        spectrum (dict): Dictionary containing the simulated spectrum results with keys:
            - 'energy': Energy axis array [eV].
            - '<polarisation>': Data arrays per polarisation.

    Returns:
        dict: Processed spectrum dictionary with:
            - 'energy': Energy axis array [eV].
            - For each polarisation:
                - 'flux': Flux array [ph/s/0.1%bw].
                - 'spectral_power': Spectral power array [W/eV].
                - 'cumulated_power': Cumulative integrated power array [W].
                - 'integrated_power': Total integrated power scalar [W].

    Example:
        spectrumDict = write_spectrum("myfile", spectrum)
    """

    spectrumDict = {
        'energy': spectrum['energy'],
        'window': {
            'dx': spectrum['axis']['x'],
            'dy': spectrum['axis']['y'],
        }
    }

    for polarisation, data in spectrum.items():
        if polarisation in ['energy', 'axis']:
            continue

        flux = data.reshape(len(spectrum['energy']))
        spectral_power = flux * CHARGE * 1E3
        cumulated_power = integrate.cumulative_trapezoid(spectral_power, spectrum['energy'], initial=0)
        integrated_power = integrate.trapezoid(spectral_power, spectrum['energy'])

        spectrumDict[polarisation] = {
            'flux': flux,
            'spectral_power': spectral_power,
            'cumulated_power': cumulated_power,
            'integrated_power': integrated_power,
        }

    if file_name is not None:
        with h5.File(f"{file_name}_spectrum.h5", "w") as f:
            group = f.create_group("XOPPY_SPECTRUM")
            spec_group = group.create_group("Spectrum")

            spec_group.create_dataset("energy", data=spectrumDict['energy'])

            window_group = spec_group.create_group("window")
            window_group.create_dataset("dx", data=spectrumDict['window']['dx'])
            window_group.create_dataset("dy", data=spectrumDict['window']['dy'])

            for pol, pol_data in spectrumDict.items():
                if pol in ["energy", "window"]:
                    continue

                pol_group = spec_group.create_group(pol)
                for key, array in pol_data.items():
                    pol_group.create_dataset(key, data=array)

    return spectrumDict


def read_spectrum(file_name: str) -> dict:
    """
    Reads a spectrum HDF5 file saved by write_spectrum and returns the dictionary.

    Parameters:
        file_name (str): Path to the spectrum HDF5 file (without '_spectrum.h5' extension).

    Returns:
        dict: Spectrum dictionary with energy and polarisation data.
    """
    spectrumDict = {}
    with h5.File(f"{file_name}_spectrum.h5", "r") as f:
        spec_group = f["XOPPY_SPECTRUM"]["Spectrum"]

        spectrumDict["energy"] = spec_group["energy"][:]

        # Read window group
        window_group = spec_group["window"]
        spectrumDict["window"] = {
            "dx": window_group["dx"][:],
            "dy": window_group["dy"][:],
        }

        for pol in spec_group:
            if pol in ["energy", "window"]:
                continue

            pol_group = spec_group[pol]
            spectrumDict[pol] = {}
            for key in pol_group:
                spectrumDict[pol][key] = pol_group[key][:]

    return spectrumDict

# ---------------------------------------------------------------------------
# Coherent mode decomposition
# ---------------------------------------------------------------------------

def write_cmd(file_name: str, cmd: dict) -> dict:
    """
    Saves coherent mode decomposition (CMD) data to an HDF5 file and returns a processed CMD dictionary.

    Parameters:
        file_name (str): Base file path for saving the CMD data. The data will be stored
                         in a file named '<file_name>_cmd.h5'.
        cmd (dict): Dictionary containing the CMD results with keys:
            - 'energy': photon energy [eV].
            - 'src_h_cmd' and 'src_v_cmd': CMD objects with attributes 'eigenvalues', 'abscissas', 'CSD'.

    Returns:
        dict: Processed CMD dictionary with energy and, for each direction ('H', 'V'):
            - 'eigenvalues': Eigenvalues array.
            - 'axis': Abscissas array.
            - 'occupation': Normalised occupation array.
            - 'cumulated': Cumulative sum of occupation array.
            - 'CF': Coherent fraction (first mode occupation).
            - 'CSD': Cross-spectral density matrix (absolute value).
    """
    cmdDict = {'energy': cmd['energy'], 'source': {}}

    for direction in ['h', 'v']:
        eigenvalues = cmd[f'src_{direction}_cmd'].eigenvalues
        axis = cmd[f'src_{direction}_cmd'].abscissas 
        occupation = eigenvalues / eigenvalues.sum()
        cumulated = np.cumsum(occupation)
        CF = occupation[0]
        CSD = np.abs(cmd[f'src_{direction}_cmd'].CSD)

        cmdDict['source'].update({
            f'{direction.upper()}': {
                'eigenvalues': eigenvalues,
                'axis': axis,
                'occupation': occupation,
                'cumulated': cumulated,
                'CF': CF,
                'CSD': CSD
            }
        })

    if file_name is not None:
        with h5.File(f"{file_name}_cmd.h5", "w") as f:
            group = f.create_group("XOPPY_CMD")
            group.attrs["energy"] = cmdDict['energy']

            for direction in ['H', 'V']:
                dir_group = group.create_group(direction)
                dir_group.create_dataset("eigenvalues", data=cmdDict['source'][direction]['eigenvalues'])
                dir_group.create_dataset("axis", data=cmdDict['source'][direction]['axis'])
                dir_group.create_dataset("occupation", data=cmdDict['source'][direction]['occupation'])
                dir_group.create_dataset("cumulated", data=cmdDict['source'][direction]['cumulated'])
                dir_group.attrs["CF"] = cmdDict['source'][direction]['CF']
                dir_group.create_dataset("CSD", data=cmdDict['source'][direction]['CSD'])

    return cmdDict

def read_cmd(file_name: str) -> dict:
    """
    Reads CMD data from an HDF5 file and reconstructs the CMD dictionary.

    Parameters:
        file_name (str): Path to the HDF5 file containing CMD data (ending with '_cmd.h5').

    Returns:
        dict: CMD dictionary with energy and, for each direction ('H', 'V'):
            - 'eigenvalues': Eigenvalues array.
            - 'axis': Abscissas array.
            - 'occupation': Normalised occupation array.
            - 'cumulated': Cumulative sum of occupation array.
            - 'CF': Coherent fraction (first mode occupation).
            - 'CSD': Cross-spectral density matrix.
    """
    cmdDict = {'source': {}}

    with h5.File(file_name, "r") as f:
        group = f["XOPPY_CMD"]
        cmdDict['energy'] = group.attrs["energy"]

        for direction in ['H', 'V']:
            dir_group = group[direction]
            cmdDict['source'][direction] = {
                'eigenvalues': dir_group["eigenvalues"][:],
                'axis': dir_group["axis"][:],
                'occupation': dir_group["occupation"][:],
                'cumulated': dir_group["cumulated"][:],
                'CF': dir_group.attrs["CF"],
                'CSD': dir_group["CSD"][:]
            }

    return cmdDict

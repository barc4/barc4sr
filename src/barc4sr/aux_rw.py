#!/bin/python

"""
This module provides a collection of functions for rading and saving barc4sr calculations
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '26/JUN/2025'

import os
import pickle
from array import array
from copy import copy

import h5py as h5
import numpy as np

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def write_electron_trajectory(file_name:str, eTraj: srwlib.SRWLPrtTrj) -> None:
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

#***********************************************************************************
# Wavevfront
#***********************************************************************************
   
def write_wavefront(file_name: str, wfr: srwlib.SRWLWfr, selected_polarisations: list, number_macro_electrons: int) -> dict:
    """
    Writes wavefront data (intensity, phase, and wavefront object) to an HDF5 file.

    Parameters:
        file_name (str): Base file path for saving the wavefront data. The data will be stored
                         in a file named '<file_name>_undulator_wfr.h5'.
        wfr (srwlib.SRWLWfr): The SRW wavefront object containing the simulated electric field.
        selected_polarisations (list or str): List of polarisations to export. Can be a single
                         string or a list of strings. Accepted values include: 'LH', 'LV', 
                         'L45', 'L135', 'CR', 'CL', 'T'.
        number_macro_electrons (int): Number of macro electrons. Default is -1.

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

    wfrDict = {'wfr': wfr}

    wfrDict.update({'axis':{'x': np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx),
                            'y': np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)}})

    all_polarisations = ['LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T']
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return write_wavefront(file_name, wfr, selected_polarisations=['T'])
    
    wfrDict.update({'energy':wfr.mesh.eStart})
    wfrDict.update({'intensity':{}})
    wfrDict.update({'phase':{}})
    wfrDict.update({'Rx':wfr.Rx, 'Ry':wfr.Ry})

    _inIntType = int(number_macro_electrons)
    _inDepType = 3

    Rx, Ry = copy(wfr.Rx), copy(wfr.Ry)

    X, Y = np.meshgrid(wfrDict['axis']['x'], wfrDict['axis']['y'])
    shperical_phase = Rx - np.sqrt(Rx**2 - X**2 - (Rx/Ry)**2 * Y**2)
    amplitude_transmission = np.ones((wfr.mesh.ny, wfr.mesh.nx), dtype='float64')
    arTr = np.empty((2 * wfr.mesh.nx * wfr.mesh.ny), dtype=shperical_phase.dtype)
    arTr[0::2] = np.reshape(amplitude_transmission,(wfr.mesh.nx*wfr.mesh.ny))
    arTr[1::2] = np.reshape(-shperical_phase,(wfr.mesh.nx*wfr.mesh.ny))
    spherical_wave = srwlib.SRWLOptT(wfr.mesh.nx, wfr.mesh.ny, 
                                     wfrDict['axis']['x'][-1]-wfrDict['axis']['x'][0],
                                     wfrDict['axis']['y'][-1]-wfrDict['axis']['y'][0],
                                     _arTr=arTr, _extTr=1, _Fx=Rx, _Fy=Ry, _x=0, _y=0)
    pp_spherical_wave =  [0, 0, 1.0, 1, 0, 1., 1., 1., 1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    OE = [spherical_wave]
    PP = [pp_spherical_wave]
    
    optBL = srwlib.SRWLOptC(OE, PP)
    srwlib.srwl.PropagElecField(wfr, optBL)

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index

        arInt = array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arInt, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
        wfrDict['intensity'].update({polarisation:np.asarray(arInt, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx))})

        arPh = array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arPh, wfr, _inPol, 4, _inDepType, wfr.mesh.eStart, 0, 0)
        # pahse = unwrap_phase(np.asarray(arPh, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)))
        phase = np.asarray(arPh, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx))
        wfrDict['phase'].update({polarisation:phase})

    wfr.Rx, wfr.Ry = Rx, Ry

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

    # Extract Rx, Ry, energy from wavefront object
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

#***********************************************************************************
# Power density
#***********************************************************************************

def write_power_density(file_name: str, stks: srwlib.SRWLStokes, selected_polarisations: list):
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


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")

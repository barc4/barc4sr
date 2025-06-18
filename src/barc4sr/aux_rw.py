#!/bin/python

"""
This module provides a collection of functions for rading and saving barc4sr calculations
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '17/JUN/2025'

import os

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
# magnetic measurments
#***********************************************************************************

def write_magnetic_field(mag_field_array: np.ndarray, file_path: str = None) -> srwlib.SRWLMagFld3D:
    """
    Generate a 3D magnetic field object based on the input magnetic field array.

    Parameters:
        mag_field_array (np.ndarray): Array containing magnetic field data. Each row corresponds to a point in the 3D space,
                                      where the first column represents the position along the longitudinal axis, and subsequent 
                                      columns represent magnetic field components (e.g., Bx, By, Bz).
        file_path (str): File path to save the generated magnetic field object. If None, the object won't be saved.

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

if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")

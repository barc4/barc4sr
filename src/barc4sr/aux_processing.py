#!/bin/python

"""
This module provides a collection of functions for processing and analyzing data related
 to undulator radiation, power density, and spectra. It includes functions for reading 
 data from XOPPY files (HDF5 format), SPECTRA files (JSON format), and processing the 
 data to calculate various properties such as spectral power, cumulated power, 
 integrated power, and power density. Additionally, it offers functions for selecting 
 specific energy ranges within 3D data sets, spatially trimming data, and generating 
 animated GIFs of energy scans in the 3D data-sets. Overall, this module facilitates 
 the analysis and visualization of data obtained from simulations related to synchrotron 
 radiation sources.
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '25/NOV/2024'

import json
import multiprocessing
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from PIL import Image, ImageDraw, ImageFont
from scipy.constants import physical_constants
from scipy.interpolate import RegularGridInterpolator

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

CHARGE = physical_constants["atomic unit of charge"][0]

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
# Spectrum
#***********************************************************************************

def write_spectrum(file_name: str, flux: np.array, energy: np.array) -> None:
    """
    Writes spectrum data to an HDF5 file.

    This function writes the provided energy and flux data to an HDF5 file. The data is stored 
    in the 'XOPPY_SPECTRUM' group within the file, with a subgroup for 'Spectrum'.

    Parameters:
        file_name (str): Base file path for saving the spectrum data. The file will be saved 
                         with the suffix '_spectrum.h5'.
        flux (np.array): 1D numpy array containing the flux data.
        energy (np.array): 1D numpy array containing the energy data.

    """
    if file_name is not None:
        with h5.File('%s_spectrum.h5'%file_name, 'w') as f:
            group = f.create_group('XOPPY_SPECTRUM')
            intensity_group = group.create_group('Spectrum')
            intensity_group.create_dataset('energy', data=energy)
            intensity_group.create_dataset('flux', data=flux) 

    spectral_power = flux*CHARGE*1E3

    cumulated_power = integrate.cumulative_trapezoid(spectral_power, energy, initial=0)
    integrated_power = integrate.trapezoid(spectral_power, energy)

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


def read_spectrum(file_list: List[str]) -> Dict:
    """
    Reads and processes spectrum data from files.

    This function reads spectrum data from files specified in 'file_list' and processes 
    it to compute spectral power, cumulated power, and integrated power.

    Parameters:
        file_list (List[str]): A list of file paths containing spectrum data.

    Returns:
        Dict: A dictionary containing processed spectrum data with the following keys:
            - 'spectrum': A dictionary containing various properties of the spectrum including:
                - 'energy': Array containing energy values.
                - 'flux': Array containing spectral flux data.
                - 'spectral_power': Array containing computed spectral power.
                - 'cumulated_power': Cumulated power computed using cumulative trapezoid integration.
                - 'integrated_power': Integrated power computed using trapezoid integration.
    """
    energy = []
    flux = []

    if isinstance(file_list, List) is False:
        file_list = [file_list]

    # new and official barc4sr format
    if file_list[0].endswith("h5") or file_list[0].endswith("hdf5"):
        for sim in file_list:
            print(sim)
            f = h5.File(sim, "r")
            energy = np.concatenate((energy, f["XOPPY_SPECTRUM"]["Spectrum"]["energy"][()]))
            flux = np.concatenate((flux, f["XOPPY_SPECTRUM"]["Spectrum"]["flux"][()]))

    # backwards compatibility - old barc4sr format
    elif file_list[0].endswith("pickle"):
        for sim in file_list:
            print(sim)
            f = open(sim, "rb")
            data = np.asarray(pickle.load(f))
            f.close()

            energy = np.concatenate((energy, data[0, :]))
            flux = np.concatenate((flux, data[1, :]))

    # SPECTRA format
    elif file_list[0].endswith("json"):
        for jsonfile in file_list:
            f = open(jsonfile)
            data = json.load(f)
            f.close()
            energy = np.concatenate((energy, data['Output']['data'][0]))
            flux = np.concatenate((flux, data['Output']['data'][1]))
            if 'Angular Flux Density' in data["Input"]["Configurations"]["Type"]:
                dist = data["Input"]["Configurations"]["Distance from the Source (m)"]
                flux /= (np.tan(0.5E-3)*dist*2*1E3)**2
    else:
        raise ValueError("Invalid file extension.")

    spectral_power = flux*CHARGE*1E3

    cumulated_power = integrate.cumulative_trapezoid(spectral_power, energy, initial=0)
    integrated_power = integrate.trapezoid(spectral_power, energy)

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

#***********************************************************************************
# Power density
#***********************************************************************************

def write_power_density(file_name: str, power_density: np.array, h_axis: np.array, v_axis: np.array) -> None:
    """
    Writes power density data to an HDF5 file.

    This function writes the provided power density data along with the corresponding 
    horizontal (h_axis) and vertical (v_axis) axes to an HDF5 file. The data is stored in 
    the 'XOPPY_POWERDENSITY' group within the file, with a subgroup for 'PowerDensity'.

    Parameters:
        file_name (str): Base file path for saving the power density data. The file will be 
                         saved with the suffix '_power_density.h5'.
        power_density (np.array): 2D numpy array containing the power density data.
        h_axis (np.array): 1D numpy array containing the horizontal axis data.
        v_axis (np.array): 1D numpy array containing the vertical axis data.

    """
    if file_name is not None:
        with h5.File('%s_power_density.h5' % file_name, 'w') as f:
            group = f.create_group('XOPPY_POWERDENSITY')
            sub_group = group.create_group('PowerDensity')
            sub_group.create_dataset('image_data', data=power_density)
            sub_group.create_dataset('axis_x', data=h_axis * 1e3)  # axis in [mm]
            sub_group.create_dataset('axis_y', data=v_axis * 1e3)

    dx = (h_axis[1]-h_axis[0])*1E3
    dy = (v_axis[1]-v_axis[0])*1E3

    CumPow = power_density.sum()*dx*dy

    print(f"Total received power: {CumPow:.3f} W")
    print(f"Peak power density: {power_density.max():.3f} W/mm^2")

    powDenSRdict = {
        "axis": {
            "x": h_axis,
            "y": v_axis,
            },
        "power_density": {
            "map":power_density,
            "CumPow": CumPow,
            "PowDenSRmax": power_density.max()
            }
        }
    
    return powDenSRdict

def read_power_density(file_name: str) -> Dict:
    """
    Reads power density data from an HDF5 (barc4sr) or JSON (SPECTRA) file and processes it.

    This function reads power density data from either an  HDF5 (barc4sr) or JSON (SPECTRA)
    file specified by 'file_name'. It extracts the power density map along with 
    corresponding x and y axes from the file.

    Parameters:
        file_name (str): File path containing power density data.

    Returns:
        Dict: A dictionary containing processed power density data with the following keys:
            - 'axis': A dictionary containing 'x' and 'y' axes arrays.
            - 'power_density': A dictionary containing power density-related data, including the power density map,
              total received power, and peak power density.
    """
    if file_name.endswith("h5") or file_name.endswith("hdf5"):
        f = h5.File(file_name, "r")
        PowDenSR = f["XOPPY_POWERDENSITY"]["PowerDensity"]["image_data"][()]

        x = f["XOPPY_POWERDENSITY"]["PowerDensity"]["axis_x"][()]
        y = f["XOPPY_POWERDENSITY"]["PowerDensity"]["axis_y"][()]

    elif file_name.endswith("json"):
        f = open(file_name)
        data = json.load(f)
        f.close()

        PowDenSR = np.reshape(data['Output']['data'][2],
                            (len(data['Output']['data'][1]), 
                             len(data['Output']['data'][0])))

        if "mrad" in data['Output']['units'][2]:
            dist = data["Input"]["Configurations"]["Distance from the Source (m)"]
            dx = (data["Input"]["Configurations"]["x Range (mm)"][1]-data["Input"]["Configurations"]["x Range (mm)"][0])*1e-3
            dy = (data["Input"]["Configurations"]["y Range (mm)"][1]-data["Input"]["Configurations"]["y Range (mm)"][0])*1e-3

            dtx = 2*np.arctan(dx/dist/2)*1e3    # mrad
            dty = 2*np.arctan(dy/dist/2)*1e3

            PowDenSR *= 1e3 * (dtx*dty)/(dx*dy*1e3*1e3)
            x = data['Output']['data'][0]
            y = data['Output']['data'][1]
        else:
            PowDenSR *= 1e3
    else:
        raise ValueError("Invalid file extension.")

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    CumPow = PowDenSR.sum()*dx*dy

    print(f"Total received power: {CumPow:.3f} W")
    print(f"Peak power density: {PowDenSR.max():.3f} W/mm^2")

    powDenSRdict = {
        "axis": {
            "x": x,
            "y": y,
            },
        "power_density": {
            "map":PowDenSR,
            "CumPow": CumPow,
            "PowDenSRmax": PowDenSR.max()
            }
        }
    
    return powDenSRdict


def trim_and_resample_power_density(powDenSRdict: Dict, **kwargs: Union[float, bool]) -> Dict:
    """
    Trims and optionally resamples the power density data map.

    This function trims the power density data map based on specified criteria and optionally resamples
    it using interpolation if new axis values are provided. It returns the trimmed and resampled power density map
    along with cumulative power and maximum power density.

    Parameters:
        - powDenSRdict (Dict): A dictionary containing power density data with the following keys:
            - 'axis': A dictionary containing 'x' and 'y' axes arrays.
            - 'power_density': A dictionary containing the power density map.
        - **kwargs (Union[float, bool]): Additional keyword arguments for optional trimming and resampling:
            - 'dx' (float): Width in the x-direction for trimming.
            - 'dy' (float): Width in the y-direction for trimming.
            - 'xc' (float): Center of the trimming region along the x-axis.
            - 'yc' (float): Center of the trimming region along the y-axis.
            - 'X' (array_like): New x-axis values for resampling.
            - 'Y' (array_like): New y-axis values for resampling.

    Returns:
        Dict: A dictionary containing trimmed and resampled power density data with the following keys:
            - 'axis': A dictionary containing trimmed and resampled 'x' and 'y' axes arrays.
            - 'power_density': A dictionary containing the trimmed and resampled power density map,
              along with cumulative power and maximum power density.

    """

    PowDenSR = powDenSRdict["power_density"]["map"]

    x, y = powDenSRdict["axis"]["x"], powDenSRdict["axis"]["y"]
    xc, yc = 0, 0
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]

    if kwargs:
        dx = kwargs.get("dx", dx)
        dy = kwargs.get("dy", dy)
        xc = kwargs.get("xc", xc)
        yc = kwargs.get("yc", yc)
        X = kwargs.get("X", x)
        Y = kwargs.get("Y", y)
        interpol = "X" in kwargs or "Y" in kwargs

    if interpol:
        print("Interpolation of PowDenSR")
        ygrid, xgrid = np.meshgrid(Y, X, indexing='ij')
        f = RegularGridInterpolator((y, x), PowDenSR, bounds_error=False, fill_value=0)
        PowDenSR = f((ygrid, xgrid))
        x, y = X, Y
    else:
        deltax, deltay = x[1] - x[0], y[1] - y[0]

        bool_x = np.logical_and((x - xc <=  dx / 2 + deltax / 20),
                                (x - xc >= -dx / 2 - deltax / 20))

        bool_y = np.logical_and((y - yc <=  dy / 2 + deltay / 20),
                                (y - yc >= -dy / 2 - deltay / 20)) 

        PowDenSR = PowDenSR[bool_y, :][:, bool_x]
        x = np.delete(x, np.logical_not(bool_x)) - xc
        y = np.delete(y, np.logical_not(bool_y)) - yc

    dx, dy = x[1] - x[0], y[1] - y[0]

    CumPow = PowDenSR.sum() * dx * dy

    print(f"Total received power: {CumPow:.3f} W")
    print(f"Peak power density: {PowDenSR.max():.3f} W/mm^2")

    return {
        "axis": {"x": x, "y": y},
        "power_density": {"map": PowDenSR, "CumPow": CumPow, "PowDenSRmax": PowDenSR.max()}
    }

#***********************************************************************************
# Spatial-spectral distribution
#***********************************************************************************

def write_emitted_radiation(file_name: str, intensity: np.array, energy: np.array, 
                            h_axis: np.array, v_axis: np.array, parallel_processing: bool = False) -> None:
    """
    Writes synchrotron radiation (I vs x vs y vs E) data to an HDF5 file.

    This function writes the provided intensity data along with the corresponding energy, 
    horizontal (h_axis), and vertical (v_axis) axes to an HDF5 file. The data is stored 
    in the 'XOPPY_RADIATION' group within the file, with a subgroup for 'Radiation'.

    Parameters:
        file_name (str): Base file path for saving the undulator radiation data. The file will be 
                         saved with the suffix '_undulator_radiation.h5'.
        intensity (np.array): 3D numpy array containing the intensity data.
        energy (np.array): 1D numpy array containing the energy axis data.
        h_axis (np.array): 1D numpy array containing the horizontal axis data.
        v_axis (np.array): 1D numpy array containing the vertical axis data.
        parallel_processing (bool, optional): Whether to use parallel processing. Defaults to False.

    """
    if file_name is not None:
        with h5.File('%s_undulator_radiation.h5' % file_name, 'w') as f:
            group = f.create_group('XOPPY_RADIATION')
            radiation_group = group.create_group('Radiation')
            radiation_group.create_dataset('stack_data', data=intensity)
            radiation_group.create_dataset('axis0', data=energy)
            radiation_group.create_dataset('axis1', data=h_axis * 1e3)
            radiation_group.create_dataset('axis2', data=v_axis * 1e3)

    if parallel_processing:
        return proc_spatial_spectral_dist_parallel(intensity, energy, h_axis, v_axis)
    else:
        return proc_spatial_spectral_dist(intensity, energy, h_axis, v_axis)


def read_emitted_radiation(file_list: List[str], parallel_processing: bool = False) -> Dict:
    """
    Reads synchrotron radiation emission (I vs x vs y vs E) data from from a list of files and processes it.

    This function reads emitted radiation data from a list of HDF5 files, concatenates the spectral 
    flux data, and processes it using either the proc_undulator_radiation function or the 
    proc_undulator_radiation_parallel function based on the value of parallel_processing.

    Parameters:
        - file_list (List[str]): A list of file paths containing synchrotron radiation data.
        - parallel_processing (bool, optional): Whether to use parallel processing. Defaults to False.

    Returns:
        Dict: A dictionary containing processed undulator radiation data.

    Notes:
        - The input HDF5 files should contain the following datasets:
            - 'XOPPY_RADIATION/Radiation/stack_data': 3D array representing the spectral flux data.
            - 'XOPPY_RADIATION/Radiation/axis0': 1D array representing the energy axis.
            - 'XOPPY_RADIATION/Radiation/axis1': 1D array representing the x-axis.
            - 'XOPPY_RADIATION/Radiation/axis2': 1D array representing the y-axis.
        - The spectral flux data from different files will be concatenated along the 0-axis.
        - The x and y axes are assumed to be the same for all files in the file_list.
    """

    if isinstance(file_list, List) is False:
        file_list = [file_list]
        
    energy = []
    spectral_flux_3D = []

    k = 0

    for sim in file_list:
        print(sim)
        f = h5.File(sim, "r")

        if k == 0:
            spectral_flux_3D = f["XOPPY_RADIATION"]["Radiation"]["stack_data"][()]
            k+=1
        else:
            spectral_flux_3D = np.concatenate((spectral_flux_3D, f["XOPPY_RADIATION"]["Radiation"]["stack_data"][()]), 0)
        energy = np.concatenate((energy, f["XOPPY_RADIATION"]["Radiation"]["axis0"][()]))

    print("UR files loaded")

    spectral_flux_3D = spectral_flux_3D.swapaxes(1, 2)

    x = f["XOPPY_RADIATION"]["Radiation"]["axis1"][()]
    y = f["XOPPY_RADIATION"]["Radiation"]["axis2"][()]

    if parallel_processing:
        return proc_spatial_spectral_dist_parallel(spectral_flux_3D, energy, x, y)
    else:
        return proc_spatial_spectral_dist(spectral_flux_3D, energy, x, y)


def proc_spatial_spectral_dist(spectral_flux_3D: np.ndarray, energy: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Processes the synchotron radiation data.

    This function calculates various properties of synchotron radiation based on the provided 3D spectral flux data.

    Parameters:
        - spectral_flux_3D (np.ndarray): A 3D numpy array representing the spectral flux data. Shape: (n_slices, ny, nx)
        - energy (np.ndarray): A 1D numpy array containing the energy values.
        - x (np.ndarray): A 1D numpy array containing the x-axis values.
        - y (np.ndarray): A 1D numpy array containing the y-axis values.

    Returns:
        Dict: A dictionary containing processed synchotron radiation data with the following keys:
            - 'axis': Dictionary containing the x and y axis values.
            - 'spectral_power_3D': 3D numpy array of spectral power.
            - 'power_density': Dictionary containing the power density information with keys:
                - 'map': 2D numpy array representing the power density map.
                - 'CumPow': Cumulative power.
                - 'PowDenSRmax': Maximum power density.
            - 'spectrum': Dictionary containing spectral information with keys:
                - 'energy': 1D numpy array of energy values.
                - 'flux': Flux values.
                - 'spectral_power': Spectral power.
                - 'cumulated_power': Cumulative power.
                - 'integrated_power': Integrated power.

    Notes:
        - The input spectral_flux_3D should have dimensions (n_slices, ny, nx) where:
            - n_slices: Number of sample images.
            - ny: Number of points along the y-axis.
            - nx: Number of points along the x-axis.
        - The energy array should correspond to the energy values for each slice in spectral_flux_3D.
        - The x and y arrays represent the coordinates of the grid on which the data is sampled.

    """
    print("Processing undulator radiation")
    n_slices = spectral_flux_3D.shape[0]
    ny = spectral_flux_3D.shape[1]
    nx = spectral_flux_3D.shape[2]

    print(f"> {n_slices} sample images ({ny} x {nx}) found ({ sys.getsizeof(spectral_flux_3D) / (1024 ** 3):.2f} Gb in memory)")

    dx = (x[1]-x[0])
    dy = (y[1]-y[0])

    flux = dx*dy*np.sum(spectral_flux_3D, axis=(1, 2))
    spectral_power = flux*CHARGE*1E3
    cumulated_power = integrate.cumulative_trapezoid(spectral_power, energy, initial=0)
    integrated_power = integrate.trapezoid(spectral_power, energy)

    PowDenSR = integrate.trapezoid(spectral_flux_3D*CHARGE*1E3, energy, axis=0)

    CumPow = dx*dy*PowDenSR.sum()

    print(f"Puissance totale reçue : {CumPow:.3f} W")
    print(f"Puissance crête reçue (incidence normale): {PowDenSR.max():.3f} W/mm^2")

    URdict = {
        "axis": {
            "x": x,
            "y": y,
        },
        "spectral_power_3D":spectral_flux_3D*CHARGE*1E3,
        "power_density": {
            "map":PowDenSR,
            "CumPow": CumPow,
            "PowDenSRmax": PowDenSR.max()
        },
        "spectrum":{
            "energy":energy,
            "flux": flux,
            "spectral_power": spectral_power,
            "cumulated_power": cumulated_power,
            "integrated_power": integrated_power
        }
    }
    print("Dictionary written")
    return URdict


def proc_spatial_spectral_dist_parallel(spectral_flux_3D: np.ndarray, energy: np.ndarray, x: np.ndarray, y: np.ndarray, chunk_size: int = 25) -> Dict:
    """
    Process synchotron radiation data in parallel.

    This function calculates various properties of synchotron radiation based on the provided 3D spectral flux data.

    Parameters:
        - spectral_flux_3D (np.ndarray): A 3D numpy array representing the spectral flux data. Shape: (n_slices, ny, nx)
        - energy (np.ndarray): A 1D numpy array containing the energy values.
        - x (np.ndarray): A 1D numpy array containing the x-axis values.
        - y (np.ndarray): A 1D numpy array containing the y-axis values.
        - chunk_size (int, optional): Size of each chunk of spectral flux data for parallel processing. Default is 25.

    Returns:
        Dict: A dictionary containing processed synchotron radiation data with the following keys:
            - 'axis': Dictionary containing the x and y axis values.
            - 'spectral_power_3D': 3D numpy array of spectral power.
            - 'power_density': Dictionary containing the power density information with keys:
                - 'map': 2D numpy array representing the power density map.
                - 'CumPow': Cumulative power.
                - 'PowDenSRmax': Maximum power density.
            - 'spectrum': Dictionary containing spectral information with keys:
                - 'energy': 1D numpy array of energy values.
                - 'flux': Flux values.
                - 'spectral_power': Spectral power.
                - 'cumulated_power': Cumulative power.
                - 'integrated_power': Integrated power.

    Notes:
        - The input spectral_flux_3D should have dimensions (n_slices, ny, nx) where:
            - n_slices: Number of sample images.
            - ny: Number of points along the y-axis.
            - nx: Number of points along the x-axis.
        - The energy array should correspond to the energy values for each slice in spectral_flux_3D.
        - The x and y arrays represent the coordinates of the grid on which the data is sampled.

    """
    print("Processing undulator radiation (parallel)")
    n_slices = spectral_flux_3D.shape[0]
    ny = spectral_flux_3D.shape[1]
    nx = spectral_flux_3D.shape[2]

    print(f"> {n_slices} sample images ({ny} x {nx}) found ({ sys.getsizeof(spectral_flux_3D) / (1024 ** 3):.2f} Gb in memory)")

    dx = (x[1]-x[0])
    dy = (y[1]-y[0])

    # Divide the data into chunks
    chunks = [(spectral_flux_3D[i:i + chunk_size+1], energy[i:i + chunk_size+1], x, y) for i in range(0, n_slices, chunk_size)]
    
    # Create a multiprocessing Pool
    with multiprocessing.Pool() as pool:
        # Process each chunk in parallel
        processed_chunks = pool.map(process_chunk, chunks)
    
    # Concatenate the processed chunks
    PowDenSR = np.zeros((ny, nx))
    flux = []

    for i, (PowDenSR_chunk, flux_chunk, energy_chunck) in enumerate(processed_chunks):
        PowDenSR += PowDenSR_chunk
        if i == 0:
            flux.extend(flux_chunk)
            previous_energy_chunck = energy_chunck
        else: 
            if energy_chunck[0] == previous_energy_chunck[-1]:
                flux.extend(flux_chunk[1:])
            else:
                flux.extend(flux_chunk)
        previous_energy_chunck = energy_chunck

    spectral_power = np.asarray(flux)*CHARGE*1E3
    cumulated_power = integrate.cumulative_trapezoid(spectral_power, energy, initial=0)
    integrated_power = integrate.trapezoid(spectral_power, energy)
    CumPow = dx*dy*PowDenSR.sum()

    print(f"Puissance totale reçue : {CumPow:.3f} W")
    print(f"Puissance crête reçue (incidence normale): {PowDenSR.max():.3f} W/mm^2")

    URdict = {
        "axis": {
            "x": x,
            "y": y,
        },
        "spectral_power_3D":spectral_flux_3D*CHARGE*1E3,
        "power_density": {
            "map":PowDenSR,
            "CumPow": CumPow,
            "PowDenSRmax": PowDenSR.max()
        },
        "spectrum":{
            "energy":energy,
            "flux": np.asarray(flux),
            "spectral_power": spectral_power,
            "cumulated_power": cumulated_power,
            "integrated_power": integrated_power
        }
    }
    print("Dictionary written")
    return URdict


def process_chunk(args):
    """
    Process a chunk of spectral flux data.

    This function calculates the power density and flux for a chunk of spectral flux data.

    Parameters:
        args (tuple): A tuple containing the following elements:
            - spectral_flux_3D_chunk (np.ndarray): A chunk of spectral flux data.
            - energy_chunk (np.ndarray): A chunk of energy values.
            - x (np.ndarray): A 1D numpy array containing the x-axis values.
            - y (np.ndarray): A 1D numpy array containing the y-axis values.

    Returns:
        tuple: A tuple containing the calculated power density and flux.
    """
    spectral_flux_3D_chunk, energy_chunk, x, y = args

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    flux = dx * dy * np.sum(spectral_flux_3D_chunk, axis=(1, 2))
    PowDenSR = integrate.trapezoid(spectral_flux_3D_chunk * CHARGE * 1E3, energy_chunk, axis=0)

    return PowDenSR, flux, energy_chunk


def select_energy_range(URdict: Dict, ei: float, ef: float, **kwargs: Union[float, bool]) -> Dict:
    """
    Selects a specific energy range from the synchotron radiation data and returns processed data within that range.

    This function selects a specific energy range from the given synchotron radiation data dictionary (URdict)
    and returns processed data within that range. Optionally, it allows trimming the data based on specified criteria.

    Parameters:
        - URdict (Dict): A dictionary containing synchotron radiation data with the following keys:
            - 'axis': A dictionary containing 'x' and 'y' axes arrays.
            - 'spectrum': A dictionary containing energy-related data, including 'energy' array.
            - 'spectral_power_3D': A 3D array representing spectral power density.
        - ei (float): The initial energy of the selected range.
        - ef (float): The final energy of the selected range.
        - **kwargs (Union[float, bool]): Additional keyword arguments for optional trimming:
            - 'dx' (float): Width in the x-direction for trimming.
            - 'dy' (float): Width in the y-direction for trimming.
            - 'xc' (float): Center of the trimming region along the x-axis.
            - 'yc' (float): Center of the trimming region along the y-axis.

    Returns:
        Dict: A dictionary containing processed synchotron radiation data within the selected energy range,
        trimmed based on the specified criteria (if any).

    Notes:
        - If 'ei' or 'ef' is set to -1, the function selects the minimum or maximum energy from the available data, respectively.
        - If 'ei' is equal to 'ef', the function duplicates the data for that energy and increments the energy values by 1.

    """
    x = URdict["axis"]["x"]
    y = URdict["axis"]["y"]

    dx = kwargs.get("dx", x[-1] - x[0])
    dy = kwargs.get("dy", y[-1] - y[0])
    xc = kwargs.get("xc", 0)
    yc = kwargs.get("yc", 0)

    deltax, deltay = x[1] - x[0], y[1] - y[0]

    bool_x = np.logical_and((x - xc <= dx / 2 + deltax / 20),
                            (x - xc >= -dx / 2 - deltax / 20))

    bool_y = np.logical_and((y - yc <= dy / 2 + deltay / 20),
                            (y - yc >= -dy / 2 - deltay / 20))

    if ei == -1:
        ei = URdict["spectrum"]["energy"][0]
    if ef == -1:
        ef = URdict["spectrum"]["energy"][-1]

    crop_map = np.logical_not(np.logical_and((URdict["spectrum"]["energy"] <= ef),
                                             (URdict["spectrum"]["energy"] >= ei)))
    energy = np.delete(URdict["spectrum"]["energy"], crop_map)
    spectral_flux_3D = np.delete(URdict["spectral_power_3D"], crop_map, axis=0) / (CHARGE * 1E3)

    if ei == ef:
        spectral_flux_3D = np.concatenate((spectral_flux_3D, spectral_flux_3D), axis=0)
        energy = np.concatenate((energy, energy + 1))

    if kwargs:  # Check if trimming is requested
        spectral_flux_3D = spectral_flux_3D[:, bool_y, :][:, :, bool_x]
        x = np.delete(x, np.logical_not(bool_x)) - xc
        y = np.delete(y, np.logical_not(bool_y)) - yc

    return proc_spatial_spectral_dist_parallel(spectral_flux_3D, energy, x, y)


def animate_energy_scan(URdict: Dict, file_name: str, **kwargs: Any) -> None:
    """
    Generate an animated GIF of an energy scan.

    Args:
        URdict (Dict): Dictionary containing spectral power data.
        file_name (str): Name of the output GIF file.
        **kwargs: Additional keyword arguments.
            duration_per_frame (float): Duration of each frame in seconds. Defaults to 0.05.
            frame_rate (float): Frame rate in frames per second. Overrides duration_per_frame if provided.
            cmap (str): Colormap for visualization. Defaults to 'plasma'.
            cumSum (bool): If True, generate an additional GIF with cumulative sum. Defaults to False.
            ScaleBar (bool): If True, add a scale bar to the GIF. Defaults to False.
            ScaleBarLength (float): Length of the scale bar in the unit of theScaleBarUnit.
            ScaleBarUnit (str): Unit of measurement for the scale bar.
            group (bool): If True, group the energy scan with the cumulative sum. Defaults to False.

    """
    duration_per_frame = 0.05
    cmap = 'plasma'
    cumSum = False
    ScaleBar = False
    group = False

    if bool(kwargs):

        if "duration_per_frame" in kwargs.keys():
            duration_per_frame = kwargs["duration_per_frame"]
        if "frame_rate" in kwargs.keys():
            duration_per_frame = 1/kwargs["frame_rate"]
        if "cmap" in kwargs.keys():
            cmap = kwargs["cmap"]
        if "cumSum" in kwargs.keys():
            cumSum = kwargs["cumSum"]

        ScaleBarLength = None
        ScaleBarUnit = None
        if "ScaleBar" in kwargs.keys():
            ScaleBar = kwargs["ScaleBar"]
        if "ScaleBarLength" in kwargs.keys():
            ScaleBar = True
            ScaleBarLength = kwargs["ScaleBarLength"]
        if "ScaleBarUnit" in kwargs.keys():
            ScaleBar = True
            ScaleBarUnit = kwargs["ScaleBarUnit"]

        if ScaleBar is True and ScaleBarUnit is None:
            # warnings.warn(">> Scale bar unit nor provided. No scale bar will be displayed", Warning)
            print(">> Warning: Scale bar unit nor provided. No scale bar will be displayed")
            ScaleBar = False

        if ScaleBar:
            if ScaleBarLength is None:
                dh = np.round((URdict["axis"]["x"][-1]-URdict["axis"]["x"][0])/4)
                ScaleBarLength = dh-dh%2
            PixelsPerLengthUnit = len(URdict["axis"]["x"])/(URdict["axis"]["x"][-1]-URdict["axis"]["x"][0])
            ScaleLengthPixels = int(ScaleBarLength * PixelsPerLengthUnit)
        
        if "group" in kwargs.keys():
            group = kwargs["group"]

    global_min = np.min(URdict["spectral_power_3D"])
    global_max = np.max(URdict["spectral_power_3D"])

    if group:
        cumulated_power = np.cumsum(URdict["spectral_power_3D"], axis=0)
        with imageio.get_writer(file_name + ".gif", mode='I', duration=duration_per_frame) as writer:
            for i, frame in enumerate(URdict["spectral_power_3D"]):

                cp_frame_min = np.min(cumulated_power[i, :, :])
                cp_frame_max = np.max(cumulated_power[i, :, :])
                cp_frame_normalized = (cumulated_power[i, :, :] - cp_frame_min) / (cp_frame_max - cp_frame_min)

                dnan = 5
                frame_normalized = (frame - global_min) / (global_max - global_min)
                frame_normalized = np.hstack((frame_normalized, np.full((frame_normalized.shape[0], dnan), np.nan)))
                frame_normalized = np.concatenate((frame_normalized, cp_frame_normalized), axis=1)

                frame_colored = plt.cm.get_cmap(cmap)(frame_normalized)
                # Convert the colormap to uint8 format (0-255)
                frame_colored_uint8 = (frame_colored[:, :, :3] * 255).astype(np.uint8)
                
                # Convert NumPy array to PIL Image
                pil_image = Image.fromarray(frame_colored_uint8)
                
                # Add frame number as text overlay
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype("arial.ttf", 20)
                draw.text((15, 15), f"E = {URdict['spectrum']['energy'][i]:.2f} eV", fill=(255, 255, 255), font=font)
                draw.text((15+frame.shape[1]+dnan, 15), "cummulated", fill=(255, 255, 255), font=font)

                if ScaleBar:
                    # Add scale bar
                    scale_bar_x0 = pil_image.width - 10 - ScaleLengthPixels
                    scale_bar_y0 = pil_image.height - 20
                    scale_bar_x1 = pil_image.width - 10
                    scale_bar_y1 = pil_image.height - 15
                    draw.rectangle([scale_bar_x0, scale_bar_y0, scale_bar_x1, scale_bar_y1], fill=(255, 255, 255))
                    # Add text for scale bar length
                    scale_text = f"{ScaleBarLength} {ScaleBarUnit}"
                    text_left, text_top, text_right, text_bottom = draw.textbbox(xy=(0,0), text=scale_text, font=font)
                    text_width, text_height = (text_right - text_left, text_bottom - text_top)
                    text_x = scale_bar_x0 + (scale_bar_x1 - scale_bar_x0 - text_width) // 2
                    text_y = scale_bar_y0 - text_height - 10
                    draw.text((text_x, text_y), scale_text, fill=(255, 255, 255), font=font)

                # Convert PIL Image back to NumPy array
                frame_with_tag = np.array(pil_image)
                
                writer.append_data(frame_with_tag)

        print(f"GIF created successfully: {file_name.split('/')[-1]}")
    else:
        with imageio.get_writer(file_name + ".gif", mode='I', duration=duration_per_frame) as writer:
            for i, frame in enumerate(URdict["spectral_power_3D"]):

                frame_normalized = (frame - global_min) / (global_max - global_min)
                frame_colored = plt.cm.get_cmap(cmap)(frame_normalized)

                # Convert the colormap to uint8 format (0-255)
                frame_colored_uint8 = (frame_colored[:, :, :3] * 255).astype(np.uint8)
                
                # Convert NumPy array to PIL Image
                pil_image = Image.fromarray(frame_colored_uint8)
                
                # Add frame number as text overlay
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype("arial.ttf", 20)
                draw.text((15, 15), f"E = {URdict['spectrum']['energy'][i]:.2f} eV", fill=(255, 255, 255), font=font)
                        
                if ScaleBar:
                    # Add scale bar
                    scale_bar_x0 = pil_image.width - 10 - ScaleLengthPixels
                    scale_bar_y0 = pil_image.height - 20
                    scale_bar_x1 = pil_image.width - 10
                    scale_bar_y1 = pil_image.height - 15
                    draw.rectangle([scale_bar_x0, scale_bar_y0, scale_bar_x1, scale_bar_y1], fill=(255, 255, 255))
                    # Add text for scale bar length
                    scale_text = f"{ScaleBarLength} {ScaleBarUnit}"
                    text_left, text_top, text_right, text_bottom = draw.textbbox(xy=(0,0), text=scale_text, font=font)
                    text_width, text_height = (text_right - text_left, text_bottom - text_top)
                    text_x = scale_bar_x0 + (scale_bar_x1 - scale_bar_x0 - text_width) // 2
                    text_y = scale_bar_y0 - text_height - 10
                    draw.text((text_x, text_y), scale_text, fill=(255, 255, 255), font=font)

                # Convert PIL Image back to NumPy array
                frame_with_tag = np.array(pil_image)
                
                writer.append_data(frame_with_tag)

        print(f"GIF created successfully: {file_name.split('/')[-1]}")

        if cumSum:
            cumulated_power = np.cumsum(URdict["spectral_power_3D"], axis=0)
            with imageio.get_writer(file_name + "_CumSum.gif", mode='I', duration=duration_per_frame) as writer:
                for i, frame in enumerate(cumulated_power):

                    frame_min = np.min(frame)
                    frame_max = np.max(frame)

                    frame_normalized = (frame - frame_min) / (frame_max - frame_min)
                    frame_colored = plt.cm.get_cmap(cmap)(frame_normalized)

                    # Convert the colormap to uint8 format (0-255)
                    frame_colored_uint8 = (frame_colored[:, :, :3] * 255).astype(np.uint8)
                    
                    # Convert NumPy array to PIL Image
                    pil_image = Image.fromarray(frame_colored_uint8)
                    
                    # Add frame number as text overlay
                    draw = ImageDraw.Draw(pil_image)
                    font = ImageFont.truetype("arial.ttf", 20)
                    draw.text((15, 15), f"E = {URdict['spectrum']['energy'][i]:.2f} eV (cummulated)", fill=(255, 255, 255), font=font)
                    if ScaleBar:
                        # Add scale bar
                        scale_bar_x0 = pil_image.width - 10 - ScaleLengthPixels
                        scale_bar_y0 = pil_image.height - 20
                        scale_bar_x1 = pil_image.width - 10
                        scale_bar_y1 = pil_image.height - 15
                        draw.rectangle([scale_bar_x0, scale_bar_y0, scale_bar_x1, scale_bar_y1], fill=(255, 255, 255))
                        # Add text for scale bar length
                        scale_text = f"{ScaleBarLength} {ScaleBarUnit}"
                        text_left, text_top, text_right, text_bottom = draw.textbbox(xy=(0,0), text=scale_text, font=font)
                        text_width, text_height = (text_right - text_left, text_bottom - text_top)
                        text_x = scale_bar_x0 + (scale_bar_x1 - scale_bar_x0 - text_width) // 2
                        text_y = scale_bar_y0 - text_height - 10
                        draw.text((text_x, text_y), scale_text, fill=(255, 255, 255), font=font)
                    # Convert PIL Image back to NumPy array
                    frame_with_tag = np.array(pil_image)
                    
                    writer.append_data(frame_with_tag)

            print(f"GIF created successfully: {file_name.split('/')[-1]+'_CumSum'}")

#***********************************************************************************
# Tuning curves
#***********************************************************************************

def write_tuning_curve(file_name: str, flux: np.array, Kh: np.array, Kv: np.array, energy: np.array) -> None:
    """
    Writes tuning curve data to an HDF5 file.

    This function writes the provided energy and flux data to an HDF5 file. The data is stored 
    in the 'XOPPY_SPECTRUM' group within the file, with a subgroup for 'TC'.

    Parameters:
        file_name (str): Base file path for saving the tuning curve data. The file will be saved 
                         with the suffix '_tc.h5'.
        flux (np.array): 1D numpy array containing the flux data.
        energy (np.array): 1D numpy array containing the energy data.

    """
    if file_name is not None:
        with h5.File('%s_tc.h5'%file_name, 'w') as f:
            group = f.create_group('XOPPY_SPECTRUM')
            intensity_group = group.create_group('TC')
            intensity_group.create_dataset('energy', data=energy)
            intensity_group.create_dataset('flux', data=flux) 
            intensity_group.create_dataset('Kh', data=Kh) 
            intensity_group.create_dataset('Kv', data=Kv) 

    tcSRdict = {
            "energy": energy,
            "flux": flux,
            "Kh": Kh,
            "Kv": Kv
    }

    return tcSRdict

def read_tuning_curve(file_name: str) -> Dict:
    """
    Reads and processes tuning curve data from files.

    Parameters:
        file_name (str): A file path containing tuning curve data.

    Returns:
        Dict: A dictionary containing processed tuning curve data with the following keys:
            - 'TC': A dictionary containing various properties of the tuning curve including:
                - 'energy': Array containing energy values.
                - 'flux': Array containing spectral flux data.
    """

    if file_name.endswith("h5") or file_name.endswith("hdf5"):
        print(file_name)
        with h5.File(file_name, "r") as f:
            energy =  f["XOPPY_SPECTRUM"]["TC"]["energy"][()]
            flux = f["XOPPY_SPECTRUM"]["TC"]["flux"][()]
            Kh = f["XOPPY_SPECTRUM"]["TC"]["Kh"][()]
            Kv = f["XOPPY_SPECTRUM"]["TC"]["Kv"][()]

    tcSRdict = {
            "energy": energy,
            "flux": flux,
            "Kh": Kh,
            "Kv": Kv
    }

    return tcSRdict

#***********************************************************************************
# Wavevfront
#***********************************************************************************
   
def write_wavefront(file_name: str, intensity:np.array, phase:np.array, h_axis:np.array,
                    v_axis:np.array) -> None:
    """
    Writes wavefront data to an HDF5 file.

    This function writes the provided intensity and phase maps along with the corresponding 
    horizontal (h_axis) and vertical (v_axis) axes to an HDF5 file. The data is stored in the 
    'XOPPY_WAVEFRONT' group within the file, with subgroups for 'Intensity' and 'Phase'.

    Parameters:
        file_name (str): Base file path for saving the wavefront data. The file will be saved with 
                         the suffix '_undulator_wft.h5'.
        intensity (np.array): 2D numpy array containing the intensity data.
        phase (np.array): 2D numpy array containing the phase data.
        h_axis (np.array): 1D numpy array containing the horizontal axis data.
        v_axis (np.array): 1D numpy array containing the vertical axis data.

    """
    if file_name is not None:
        with h5.File('%s_undulator_wft.h5'%file_name, 'w') as f:
            group = f.create_group('XOPPY_WAVEFRONT')
            intensity_group = group.create_group('Intensity')
            intensity_group.create_dataset('image_data', data=intensity)
            intensity_group.create_dataset('axis_x', data=h_axis*1e3) 
            intensity_group.create_dataset('axis_y', data=v_axis*1e3)
            intensity_group = group.create_group('Phase')
            intensity_group.create_dataset('image_data', data=phase)
            intensity_group.create_dataset('axis_x', data=h_axis*1e3) 
            intensity_group.create_dataset('axis_y', data=v_axis*1e3)

    wftDict = {
        "axis": {
            "x": h_axis,
            "y": v_axis,
            },
        "wavefront": {
            "intensity":intensity,
            "phase": phase,
            }
        }
    
    return wftDict

def read_wavefront(file_name: str) -> Dict:
    """
    Reads wavefront data from an HDF5 or JSON (SPECTRA) file and processes it.

    This function reads wavefront data from an HDF5 file (barc4sr) or JSON (SPECTRA) file specified by 
    'file_name'. It extracts the intensity and phase maps along with  corresponding x 
    and y axes from the file.

    Parameters:
        file_name (str): File path containing wavefront data.

    Returns:
        Dict: A dictionary containing processed wavefront data with the following keys:
            - 'axis': A dictionary containing 'x' and 'y' axes arrays.
            - 'intensity': 2D numpy array of intensity data.
            - 'phase': 2D numpy array of phase data.
    """
    if file_name.endswith("h5") or file_name.endswith("hdf5"):
        f = h5.File(file_name, "r")
        phase = f["XOPPY_WAVEFRONT"]["Phase"]["image_data"][()]
        intensity = f["XOPPY_WAVEFRONT"]["Intensity"]["image_data"][()]

        x = f["XOPPY_WAVEFRONT"]["Phase"]["axis_x"][()]
        y = f["XOPPY_WAVEFRONT"]["Phase"]["axis_y"][()]

    elif file_name.endswith("json"):
        f = open(file_name)
        data = json.load(f)
        f.close()

        intensity = np.reshape(data['Output']['data'][2],
                            (len(data['Output']['data'][1]), 
                             len(data['Output']['data'][0])))
        x = np.asarray(data['Output']['data'][0])*1e-3
        y = np.asarray(data['Output']['data'][1])*1e-3
        if "mr" in data['Output']['units'][2]:
            dist = data["Input"]["Configurations"]["Distance from the Source (m)"]
            dx = x[1]-x[0]
            dy = y[1]-y[0]

            dtx = np.arctan(dx/dist)*1e3    # mrad
            dty = np.arctan(dy/dist)*1e3
            intensity *= (dtx*dty)/(dx*dy*1e3*1e3)

        phase = np.zeros(intensity.shape)
    else:
        raise ValueError("Invalid file extension.")
        
    wftDict = {
        "axis": {
            "x": x,
            "y": y,
            },
        "wavefront": {
            "intensity":intensity,
            "phase": phase,
            }
        }
    
    return wftDict



#***********************************************************************************
# magnetic measurments
#***********************************************************************************

def write_magnetic_field(mag_field_array: np.ndarray, file_path: Optional[str] = None) -> srwlib.SRWLMagFld3D:
    """
    Generate a 3D magnetic field object based on the input magnetic field array.

    Parameters:
        mag_field_array (np.ndarray): Array containing magnetic field data. Each row corresponds to a point in the 3D space,
                                      where the first column represents the position along the longitudinal axis, and subsequent 
                                      columns represent magnetic field components (e.g., Bx, By, Bz).
        file_path (str, optional): File path to save the generated magnetic field object. If None, the object won't be saved.

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


#***********************************************************************************
# Mutual intensity, (Cross-) Spectral Density and Degree of Coherence
#***********************************************************************************

if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for processing and analyzing data related to synchrotron radiation, power density, and spectra.")


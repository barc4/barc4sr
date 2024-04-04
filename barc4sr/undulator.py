#!/bin/python

"""
This module provides a collection of functions for performing calculations related to 
undulators used in synchrotron radiation sources.
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '12/MAR/2024'
__changed__ = '29/MAR/2024'

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.integrate as integrate
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from barc4sr.utils import energy_wavelength, get_gamma

USE_SRWLIB = False
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
RMS = np.sqrt(2)/2

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def read_electron_trajectory(file_path: str) -> Dict[str, List[Union[float, None]]]:
    """
    Reads SRW electron trajectory data from a .dat file.

    Args:
        file_path (str): The path to the .dat file containing electron trajectory data.

    Returns:
        dict: A dictionary where keys are the column names extracted from the header
            (ct, X, BetaX, Y, BetaY, Z, BetaZ, Bx, By, Bz),
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
# read/write functions for magnetic fields
#***********************************************************************************

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


def generate_magnetic_measurement(und_per: float, B: float, num_und_per: int, 
                                  step_size: float, add_terminations: bool = True, 
                                  field_direction: str="v",
                                  und_per_disp: float = 0, B_disp: float = 0, 
                                  initial_phase_disp: float = 0, seed: int = 69, 
                                  num_samples: int = 1000, file_path: str=None, save_srw: bool=True) -> tuple:
    """
    Generate a magnetic measurement.

    This function generates a magnetic measurement with optional variations and noise.

    Args:
        und_per (float): Period of the undulator.
        B (float): Amplitude of the magnetic field.
        num_und_per (int): Number of undulator periods.
        step_size (float): Step size between samples.
        add_terminations (bool, optional): Whether to add magnetic terminations. Defaults to True.
        und_per_disp (float, optional): Standard deviation for random variation in undulator period. Defaults to 0.
        B_disp (float, optional): Standard deviation for random variation in magnetic field amplitude. Defaults to 0.
        initial_phase_disp (float, optional): Standard deviation for random variation in initial phase (in degrees). Defaults to 0.
        seed (int, optional): Seed for the random number generator. Defaults to 69.
        num_samples (int, optional): Number of samples for averaging. Defaults to 1000.
        file_path (str, optional): File path to save the generated magnetic field object. If None, the object won't be saved.
        save_srw (bool, optional): Whether to save the data in SRW format. Defaults to True.

    Returns:
        tuple: A tuple containing the magnetic field array and the axis array.
    """

    pad = 10
    nsteps = int((num_und_per+pad) * und_per / step_size)
    axis = np.linspace(-(num_und_per + pad) * und_per / 2, (num_und_per + pad) * und_per / 2, nsteps)
    
    if und_per_disp != 0 or B_disp != 0 or initial_phase_disp != 0:
        print("Adding phase errors")
        Bd = B + B_disp * np.sin(2 * np.pi * axis / (und_per * np.random.uniform(7.5, 12.5)) + np.random.normal(0, 1))
        magnetic_field = np.zeros((num_samples, len(axis)))
        dund_per = np.random.normal(loc=und_per, scale=und_per_disp, size=num_samples)
        dphase = np.random.normal(loc=0, scale=initial_phase_disp * np.pi / 180, size=num_samples)
        for i in range(num_samples):
            magnetic_field[i,:] = np.sin(2 * np.pi * axis / dund_per[i] + dphase[i])
        magnetic_field = Bd*np.mean(magnetic_field, axis=0)
    else:
        magnetic_field = B * np.sin(2 * np.pi * axis / und_per)

    if add_terminations:
        magnetic_field[axis < -(num_und_per) * und_per / 2] *= 3/4
        magnetic_field[axis > (num_und_per) * und_per / 2] *= 3/4
        magnetic_field[axis < -(num_und_per + 1) * und_per / 2] *= 1/3
        magnetic_field[axis > (num_und_per + 1) * und_per / 2] *= 1/3         
        magnetic_field[axis < -(num_und_per + 2) * und_per / 2] = 0 
        magnetic_field[axis > (num_und_per + 2) * und_per / 2] = 0  

    else:
        magnetic_field[axis < -(num_und_per) * und_per / 2] = 0
        magnetic_field[axis > (num_und_per) * und_per / 2] = 0

    if field_direction == "v":
        magnetic_field_vertical = magnetic_field
        magnetic_field_horizontal = magnetic_field*0
    elif field_direction == "h":
        magnetic_field_vertical = magnetic_field*0
        magnetic_field_horizontal = magnetic_field
    else:
        raise ValueError("Not valid field direction given.")
    
    if file_path is not None:
        if save_srw:
            magFldCnt = np.zeros([len(axis), 3])
            for i in range(len(axis)):
                magFldCnt[i,0] = axis[i]*1e3    # field is measured in mm on the benchtest
                magFldCnt[i,1] = magnetic_field_horizontal[i]
                magFldCnt[i,2] = magnetic_field_vertical[i]
            magFldCnt = generate_srw_magnetic_field(magFldCnt, file_path.replace(".txt", ".dat"))

        else:
            print(field_direction)
            print(np.amax(magnetic_field_vertical))
            print(np.amax(magnetic_field_horizontal))

            print(f">>> saving {file_path}")
            with open(file_path, 'w') as file:
                file.write("# Magnetic field data\n")
                file.write("# Axis_position   Horizontal_field   Vertical_field\n")
                for i in range(len(axis)):
                    file.write(f"{axis[i]}   {magnetic_field_horizontal[i]}   {magnetic_field_vertical[i]}\n")

    return magnetic_field, axis


def generate_srw_magnetic_field(mag_field_array: np.ndarray, file_path: Optional[str] = None) -> srwlib.SRWLMagFld3D:
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

#***********************************************************************************
# magnetic field properties
#***********************************************************************************

def get_magnetic_field_properties(mag_field_component: np.ndarray, 
                                  field_axis: np.ndarray, 
                                  threshold: float = 0.7,
                                  **kwargs: Any) -> Dict[str, Any]:
    """
    Perform analysis on magnetic field data.

    Parameters:
        mag_field_component (np.ndarray): Array containing magnetic field component data.
        field_axis (np.ndarray): Array containing axis data corresponding to the magnetic field component.
        threshold (float, optional): Peak detection threshold as a fraction of the maximum value.
        **kwargs: Additional keyword arguments.
            - pad (bool): Whether to pad the magnetic field component array for FFT analysis. Default is False.
            - positive_side (bool): Whether to consider only the positive side of the FFT. Default is True.

    Returns:
        Dict[str, Any]: Dictionary containing magnetic field properties including mean, standard deviation,
                        undulator period mean, undulator period standard deviation, frequency, FFT data, 
                        and first and second field integrals.
                        
        Dictionary structure:
            {
                "field": np.ndarray,                  # Array containing the magnetic field component data.
                "axis": np.ndarray,                   # Array containing the axis data corresponding to the magnetic field component.
                "mag_field_mean": float,              # Mean value of the magnetic field component.
                "mag_field_std": float,               # Standard deviation of the magnetic field component.
                "und_period_mean": float,             # Mean value of the undulator period.
                "und_period_std": float,              # Standard deviation of the undulator period.
                "first_field_integral": np.ndarray,   # Array containing the first field integral.
                "second_field_integral": np.ndarray,  # Array containing the first field integral.
                "freq": np.ndarray,                   # Array containing frequency data for FFT analysis.
                "fft": np.ndarray                     # Array containing FFT data.
            }
    """

    field_axis -= np.mean(field_axis)
    peaks, _ = find_peaks(mag_field_component, height=np.amax(mag_field_component) *threshold)

    # Calculate distances between consecutive peaks
    periods = np.diff(field_axis[peaks])
    average_period = np.mean(periods)
    period_dispersion = np.std(periods)
    average_peak = np.mean(mag_field_component[peaks])
    peak_dispersion = np.std(mag_field_component[peaks])

    print(f"Number of peaks over 0.7*Bmax: {len(peaks)}")
    print(f"Average period: {average_period * 1e3:.3f}+-{period_dispersion * 1e3:.3f} [mm]")
    print(f"Average peak: {average_peak:.3f}+-{peak_dispersion:.3f} [T]")

    mag_field_properties = {
        "field": mag_field_component,
        "axis": field_axis,
        "mag_field_mean": average_peak,
        "mag_field_std": peak_dispersion,
        "und_period_mean": average_period,
        "und_period_std": period_dispersion,
    }
    # # RC20240315: debug
    # import matplotlib.pyplot as plt
    # plt.plot(mag_field_component)
    # plt.plot(peaks, mag_field_component[peaks], "x")
    # plt.plot(np.zeros_like(mag_field_component), "--", color="gray")
    # plt.show()

    # First and second field integrals

    first_field_integral = integrate.cumulative_trapezoid(mag_field_component, field_axis, initial=0)
    second_field_integral = integrate.cumulative_trapezoid(first_field_integral, field_axis, initial=0)

    mag_field_properties["first_field_integral"] = first_field_integral
    mag_field_properties["second_field_integral"] = second_field_integral

    # Frequency analysis of the magnetic field
    pad = kwargs.get("pad", False)
    positive_side = kwargs.get("positive_side", True)

    if pad:
        mag_field_component = np.pad(mag_field_component, 
                                     (int(len(mag_field_component) / 2), 
                                      int(len(mag_field_component) / 2)))
    
    naxis = len(mag_field_component)
    daxis = field_axis[1] - field_axis[0]

    fftfield = np.abs(np.fft.fftshift(np.fft.fft(mag_field_component)))
    freq = np.fft.fftshift(np.fft.fftfreq(naxis, daxis))

    if positive_side:
        fftfield = 2 * fftfield[freq > 0]
        freq = freq[freq > 0]

    mag_field_properties["freq"] = freq
    mag_field_properties["fft"] = fftfield

    return mag_field_properties

#***********************************************************************************
# undulator auxiliary functions
#***********************************************************************************

def get_K_from_B(B: float, und_per: float) -> float:
    """
    Calculate the undulator parameter K from the magnetic field B and the undulator period.

    Parameters:
    B (float): Magnetic field in Tesla.
    und_per (float): Undulator period in meters.

    Returns:
    float: The undulator parameter K.
    """
    K = CHARGE * B * und_per / (2 * PI * MASS * LIGHT)
    return K


def get_B_from_K(K: float, und_per: float) -> float:
    """
    Calculate the undulator magnetic field in Tesla from the undulator parameter K and the undulator period.

    Parameters:
    K (float): The undulator parameter K.
    und_per (float): Undulator period in meters.

    Returns:
    float: Magnetic field in Tesla.
    """
    B = K * 2 * PI * MASS * LIGHT/(CHARGE * und_per)
    return B

#***********************************************************************************
# field-gap relationship
#***********************************************************************************

def fit_gap_field_relation(gap_table: List[float], B_table: List[float], 
                           und_per: float) -> Tuple[float, float, float]:
    """
    Fit parameters coeff0, coeff1, and coeff2 for an undulator from the given tables:

    B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

    Parameters:
        gap_table (List[float]): List of gap sizes in meters.
        B_table (List[float]): List of magnetic field values in Tesla corresponding to the gap sizes.
        und_per (float): Undulator period in meters.

    Returns:
        Tuple[float, float, float]: Fitted parameters (coeff0, coeff1, coeff2).
    """
    def _model(gp, c0, c1, c2):
        return c0 * np.exp(c1*gp + c2*gp**2)

    def _fit_parameters(gap, und_per, B):
        gp = gap / und_per
        popt, pcov = curve_fit(_model, gp, B, p0=(1, 1, 1)) 
        return popt

    popt = _fit_parameters(np.asarray(gap_table), und_per, np.asarray(B_table))
    coeff0_fit, coeff1_fit, coeff2_fit = popt

    print("Fitted parameters:")
    print("coeff0:", coeff0_fit)
    print("coeff1:", coeff1_fit)
    print("coeff2:", coeff2_fit)

    return coeff0_fit, coeff1_fit, coeff2_fit


def get_B_from_gap(gap: Union[float, np.ndarray], und_per: float, coeff: Tuple[float, float, float]) -> Union[float, np.ndarray, None]:
    """
    Calculate the magnetic field B from the given parameters:
       B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

    Parameters:
        gap (Union[float, np.ndarray]): Gap size(s) in meters.
        und_per (float): Undulator period in meters.
        coeff (Tuple[float, float, float]): Fit coefficients.

    Returns:
        Union[float, np.ndarray, None]: Calculated magnetic field B if gap and period are positive, otherwise None.
    """
    if isinstance(gap, np.ndarray):
        if np.any(gap <= 0) or und_per <= 0:
            return None
        gp = gap / und_per
    else:
        if gap <= 0 or und_per <= 0:
            return None
        gp = gap / und_per

    B = coeff[0] * np.exp(coeff[1] * gp + coeff[2] * gp**2)
    return B

#***********************************************************************************
# undulator emission auxiliary functions
#***********************************************************************************

def get_emission_energy(und_per: float, K: float, ring_e: float, n: int = 1, theta: float = 0) -> float:
    """
    Calculate the energy of an undulator emission in a storage ring.

    Parameters:
        und_per (float): Undulator period in meters.
        K (float): Undulator parameter.
        ring_e (float): Energy of electrons in GeV.
        n (int, optional): Harmonic number (default is 1).
        theta (float, optional): Observation angle in radians (default is 0).

    Returns:
        float: Emission energy in electron volts.
    """
    gamma = get_gamma(ring_e)
    emission_wavelength = und_per * (1 + (K ** 2) / 2 + (gamma * theta) ** 2) / (2 * n * gamma ** 2)

    return energy_wavelength(emission_wavelength, "m")


def find_emission_harmonic_and_K(energy: float, und_per: float, ring_e: float, Kmin: float = 0.1, theta: float = 0) -> Tuple[int, float]:
    """
    Find the emission harmonic number and undulator parameter K for a given energy in a storage ring.

    Parameters:
        - energy (float): Energy of the emitted radiation in electron volts.
        - und_per (float): Undulator period in meters.
        - ring_e (float): Energy of electrons in GeV.
        - Kmin (float, optional): Minimum value of the undulator parameter (default is 0.1).
        - theta (float, optional): Observation angle in radians (default is 0).

    Returns:
        Tuple[int, float]: A tuple containing the emission harmonic number and the undulator parameter K.

    Raises:
        ValueError: If no valid harmonic is found.
    """
    count = 0
    harm = 0
    gamma = get_gamma(ring_e)
    wavelength = energy_wavelength(energy, 'eV')

    while harm == 0:
        n = 2 * count + 1
        K = np.sqrt(2 * ((2 * n * wavelength * gamma ** 2) / und_per - 1 - (gamma * theta) ** 2))
        if K >= Kmin:
            harm = int(n)
        count += 1
        # Break loop if no valid harmonic is found
        if count > 21:
            raise ValueError("No valid harmonic found.")

    return harm, K
        
#***********************************************************************************
# power calculation
#***********************************************************************************

def total_power(ring_e: float, ring_curr: float, und_per: float, und_n_per: int,
              B: Optional[float] = None, K: Optional[float] = None,
              verbose: bool = False) -> float:
    """ 
    Calculate the total power emitted by a planar undulator in kilowatts (kW) based on Eq. 56 
    from K. J. Kim, "Optical and power characteristics of synchrotron radiation sources"
    [also Erratum 34(4)1243(Apr1995)], Opt. Eng 34(2), 342 (1995). 

    :param ring_e: Ring energy in gigaelectronvolts (GeV).
    :param ring_curr: Ring current in amperes (A).
    :param und_per: Undulator period in meters (m).
    :param und_n_per: Number of periods.
    :param B: Magnetic field in tesla (T). If not provided, it will be calculated based on K.
    :param K: Deflection parameter. Required if B is not provided.
    :param verbose: Whether to print intermediate calculation results. Defaults to False.
    
    :return: Total power emitted by the undulator in kilowatts (kW).
    """

    if B is None:
        if K is None:
            raise TypeError("Please, provide either B or K for the undulator")
        else:
            B = get_B_from_K(K, und_per)
            if verbose:
                print(">>> B = %.5f [T]"%B)

    return 0.63*(ring_e**2)*(B**2)*ring_curr*und_per*und_n_per


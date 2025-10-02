#!/bin/python

"""
This module provides several auxiliary magnetic field functions
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '26/JUL/2025'

import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


#***********************************************************************************
# Arbitrary magnetic fields
#***********************************************************************************

def check_magnetic_field_dictionary(magnetic_field_dictionary: dict) -> bool:
    """
    Validate a magnetic field dictionary.

    Parameters
    ----------
    magnetic_field_dictionary : dict
        Magnetic field definition with at least the following keys:
        - 's' : 1D array-like, positions [m], shape (N,)
        - 'B' : 2D array-like, magnetic field vectors [T], shape (N, 3)
        Extra keys are allowed.

    Returns
    -------
    bool
        True if the dictionary is valid, False otherwise.

    Raises
    ------
    TypeError
        If `magnetic_field_dictionary` is not a dict.
    KeyError
        If `magnetic_field_dictionary` does not contain keys 's' and 'B'.
    ValueError
        If 's' is not 1D, if 'B' is not 2D with shape (N, 3),
        or if any values in 's' or 'B' are non-finite.
    """
    if not isinstance(magnetic_field_dictionary, dict):
        raise TypeError("Input must be a dictionary.")

    required = {'s', 'B'}
    if not required.issubset(magnetic_field_dictionary.keys()):
        raise KeyError(f"Missing required keys: {required - set(magnetic_field_dictionary.keys())}")

    s = magnetic_field_dictionary['s']
    B = magnetic_field_dictionary['B']

    try:
        s_arr = np.asarray(s, dtype=float)
        B_arr = np.asarray(B, dtype=float)
    except Exception as e:
        raise ValueError("Failed to convert 's' or 'B' to arrays.") from e

    if s_arr.ndim != 1:
        raise ValueError("'s' must be a 1D array.")
    if B_arr.ndim != 2 or B_arr.shape != (len(s_arr), 3):
        raise ValueError("'B' must be a 2D array with shape (N, 3).")
    if not (np.isfinite(s_arr).all() and np.isfinite(B_arr).all()):
        raise ValueError("'s' and 'B' must not contain NaN or Inf values.")

    return True


def DBA_magnetic_field(
    B: float,
    L: float,
    straight_length: float,
    fringe: float = 0,
    step_size: float = 1e-3,
    gaussian_kernel: float = 0,
) -> dict:
    """
    Generate a double-bend magnetic field profile.

    Parameters
    ----------
    B : float
        Magnetic field strength [T].
    L : float
        Length of each bending magnet [m].
    straight_length : float
        Length of the central zero-field region [m].
    fringe : float, optional
        Length of the fringe field region on each magnet edge [m]. Default is 0.
    step_size : float, optional
        Spatial step size along the lattice [m]. Default is 1e-3.
    gaussian_kernel : float, optional
        Standard deviation [m] of a Gaussian kernel used to smooth the
        magnetic field profile. No smoothing is applied if 0. Default is 0.

    Returns
    -------
    dict
        Magnetic field dictionary with:
        - 's' : array of shape (N,), positions along the magnetic lattice [m],
          centered at 0.
        - 'B' : array of shape (N, 3), magnetic field vectors [T] with
          components (Bx=0, By=B, Bz=0).

    Raises
    ------
    ValueError
        If `L`, `straight_length`, `fringe`, or `step_size` are not positive.
    """
    if L <= 0:
        raise ValueError("Bending magnet length L must be positive.")
    if straight_length < 0:
        raise ValueError("Straight section length must be non-negative.")
    if fringe < 0:
        raise ValueError("Fringe length must be non-negative.")
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    if gaussian_kernel < 0:
        raise ValueError("Gaussian kernel must be non-negative.")

    total_length = 2 * (fringe + L) + straight_length
    s = np.arange(-total_length / 2, total_length / 2 + step_size, step_size)
    B_field = np.zeros(s.size)

    bump1_start = s[0] + fringe
    bump1_end = bump1_start + L
    bump2_start = bump1_end + straight_length
    bump2_end = bump2_start + L

    B_field[(s >= bump1_start) & (s <= bump1_end)] = B
    B_field[(s >= bump2_start) & (s <= bump2_end)] = B

    if gaussian_kernel > 0:
        B_field = gaussian_filter1d(B_field, sigma=gaussian_kernel / step_size)

    return {
        's': s.T,
        'B': np.asarray([np.zeros(len(s)), B_field, np.zeros(len(s))]).T,
    }


def MBA_magnetic_field(
    bends: dict,
    fringe: float = 0.0,
    step_size: float = 1e-3,
    default_gaussian_kernel: float = 0.0,
) -> dict:
    """
    Generate a multi‑bend achromat (MBA) magnetic field.

    Parameters:
        bends (dict): Bend specification with keys:
            - 'B' (list[float] | np.ndarray): Plateau field(s) [T].
            - 'L' (list[float] | np.ndarray): Plateau length(s) [m].
            - 's0' (list[float] | np.ndarray): Center position(s) [m] on a global s-axis.
            - 'gaussian_kernel' (float | list[float] | np.ndarray, optional):
                Per‑bend Gaussian sigma(s) [m] for edge smoothing. If omitted, uses
                `default_gaussian_kernel`. A scalar applies to all bends.
        fringe (float): Extra padding [m] added to both ends of the global s-axis.
        step_size (float): Spatial step size [m].
        default_gaussian_kernel (float): Gaussian sigma [m] applied when
            'gaussian_kernel' is not provided for a bend. Set 0 for sharp edges.

    Returns:
        dict:
            - 's' (np.ndarray): Position array along the beamline [m], centered at 0.
            - 'B' (np.ndarray): Magnetic field vectors shaped (len(s), 3) [T],
              with the field along the y-component.
    """

    for key in ['B', 'L', 's0']:
        if key not in bends:
            raise ValueError(f"Missing required key '{key}' in bends dictionary.")
    
    B_list  = np.atleast_1d(bends['B']).astype(float)
    L_list  = np.atleast_1d(bends['L']).astype(float)
    s0_list = np.atleast_1d(bends['s0']).astype(float)

    if not (len(B_list) == len(L_list) == len(s0_list)):
        raise ValueError("'B', 'L', and 's0' lists must have the same length.")

    N_bends = len(B_list)

    if 'gaussian_kernel' in bends:
        gk_data = bends['gaussian_kernel']
        if np.isscalar(gk_data):
            gk_list = np.full(N_bends, float(gk_data))
        else:
            gk_list = np.atleast_1d(gk_data).astype(float)
            if len(gk_list) != N_bends:
                raise ValueError("'gaussian_kernel' list must match length of 'B'.")
    else:
        gk_list = np.full(N_bends, float(default_gaussian_kernel))

    lefts  = s0_list - L_list/2
    rights = s0_list + L_list/2
    s_min_raw = np.min(lefts)  - abs(fringe)
    s_max_raw = np.max(rights) + abs(fringe)

    half_span = max(abs(s_min_raw), abs(s_max_raw))
    s_min = -half_span
    s_max =  half_span

    n_steps = int(np.floor((s_max - s_min) / step_size))
    s = np.linspace(s_min, s_min + n_steps * step_size, n_steps + 1)

    B_total_y = np.zeros_like(s, dtype=float)
    for B_val, L_val, s0_val, gk_val in zip(B_list, L_list, s0_list, gk_list):
        field = np.zeros_like(s, dtype=float)
        left  = s0_val - L_val/2
        right = s0_val + L_val/2
        mask = (s >= left) & (s <= right)
        field[mask] = B_val
        if gk_val > 0:
            sigma_samples = gk_val / step_size
            field = gaussian_filter1d(field, sigma=sigma_samples, mode='nearest')
        B_total_y += field

    B_vec = np.column_stack([np.zeros_like(s), B_total_y, np.zeros_like(s)])
    return {'s': s, 'B': B_vec}


#***********************************************************************************
# periodic signal treatment
#***********************************************************************************
    
def treat_periodic_signal(signal, axis, threshold=0.5):
    """
    Count the number of periods in a sinusoidal signal with a threshold on amplitude.
    
    Parameters:
    signal (np.array): The signal array.
    threshold (float): Fraction of the maximum amplitude to consider valid zero crossings.
                       Default is 0.5 (50% of max amplitude).
    
    Returns:
    int: Number of complete periods in the signal.
    """

    axis -= np.mean(axis)

    peaks, _ = find_peaks(np.abs(signal), height=np.amax(signal) *threshold)

    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    num_periods = len(zero_crossings)

    if zero_crossings[0] < peaks[0]:
        num_periods -= 1
    if zero_crossings[-1] > peaks[-1]:
        num_periods -= 1

    num_periods = num_periods // 2

    periods = np.diff(axis[peaks])
    average_period = np.mean(periods)
    period_dispersion = np.std(periods)
    average_peak = np.mean(np.abs(signal[peaks]))
    peak_dispersion = np.std(np.abs(signal[peaks]))

    # # RC 2024/11/25 - quick debug
    # import matplotlib.pyplot as plt
    # plt.plot(signal)
    # plt.plot(peaks, signal[peaks], "o")
    # plt.plot(zero_crossings, signal[zero_crossings], "x")
    # plt.plot(np.zeros_like(signal), "--", color="gray")
    # plt.show()

    if threshold == 0:
        print(f"Number of peaks: {len(peaks)/2}")
    else:
        print(f"Number of peaks over {100*threshold:.2f} %: {len(peaks)/2}")
    print(f"Number of periods (based on zero crossing): {num_periods}")
    print(f"Average period: {average_period*2 * 1e3:.3f}+-{period_dispersion * 1e3:.3f} [mm]")
    print(f"Average peak value: {average_peak:.3e}+-{peak_dispersion:.3e} au")

    periodic_signal_properties = {
        "signal": signal,
        "axis": axis,
        "number_of_peaks": len(peaks)/2,
        "number_of_periods": num_periods,
        "mean_period": average_period*2,
        "std_period": period_dispersion,
        "threshold": threshold
    }

    return periodic_signal_properties



# def fit_gap_field_relation(gap_table: List[float], B_table: List[float], 
#                            und_per: float) -> Tuple[float, float, float]:
#     """
#     Fit parameters coeff0, coeff1, and coeff2 for an undulator from the given tables:

#     B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

#     Parameters:
#         gap_table (List[float]): List of gap sizes in meters.
#         B_table (List[float]): List of magnetic field values in Tesla corresponding to the gap sizes.
#         und_per (float): Undulator period in meters.

#     Returns:
#         Tuple[float, float, float]: Fitted parameters (coeff0, coeff1, coeff2).
#     """
#     def _model(gp, c0, c1, c2):
#         return c0 * np.exp(c1*gp + c2*gp**2)

#     def _fit_parameters(gap, und_per, B):
#         gp = gap / und_per
#         popt, pcov = curve_fit(_model, gp, B, p0=(1, 1, 1)) 
#         return popt

#     popt = _fit_parameters(np.asarray(gap_table), und_per, np.asarray(B_table))
#     coeff0_fit, coeff1_fit, coeff2_fit = popt

#     print("Fitted parameters:")
#     print("coeff0:", coeff0_fit)
#     print("coeff1:", coeff1_fit)
#     print("coeff2:", coeff2_fit)

#     return coeff0_fit, coeff1_fit, coeff2_fit


# def get_B_from_gap(gap: Union[float, np.ndarray], und_per: float, coeff: Tuple[float, float, float]) -> Union[float, np.ndarray, None]:
#     """
#     Calculate the magnetic field B from the given parameters:
#        B0 = c0 * exp[c1(gap/und_per) + c2(gap/und_per)**2]

#     Parameters:
#         gap (Union[float, np.ndarray]): Gap size(s) in meters.
#         und_per (float): Undulator period in meters.
#         coeff (Tuple[float, float, float]): Fit coefficients.

#     Returns:
#         Union[float, np.ndarray, None]: Calculated magnetic field B if gap and period are positive, otherwise None.
#     """
#     if isinstance(gap, np.ndarray):
#         if np.any(gap <= 0) or und_per <= 0:
#             return None
#         gp = gap / und_per
#     else:
#         if gap <= 0 or und_per <= 0:
#             return None
#         gp = gap / und_per

#     B = coeff[0] * np.exp(coeff[1] * gp + coeff[2] * gp**2)
#     return B
#     
if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
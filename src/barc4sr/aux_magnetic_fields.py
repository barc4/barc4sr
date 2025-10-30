#!/bin/python

"""
This module provides several auxiliary magnetic field functions
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '2024.11.25'
__changed__ = '2025.10.23'

import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


#***********************************************************************************
# Arbitrary magnetic fields
#***********************************************************************************

def check_magnetic_field_dictionary(magnetic_field_dictionary: dict) -> bool:
    """
    Validate a magnetic field dictionary for structural and numerical consistency.

    Parameters
    ----------
    magnetic_field_dictionary : dict
        Dictionary describing a magnetic field, with at least:
          - 's' : array_like, shape (N,)
              Positions along the optical axis [m].
          - 'B' : array_like, shape (N, 3)
              Magnetic field vectors [T], ordered as (Bx, By, Bz).
        Extra keys are allowed and ignored during validation.

    Returns
    -------
    bool
        True if the dictionary passes all validation checks.

    Raises
    ------
    TypeError
        If `magnetic_field_dictionary` is not a dictionary.
    KeyError
        If required keys ('s', 'B') are missing.
    ValueError
        If:
          - 's' is not 1D,
          - 'B' is not 2D with shape (N, 3),
          - or any element in 's' or 'B' is NaN or Inf.

    Notes
    -----
    This function performs only consistency checks; it does not modify
    or normalize the data.
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

#***********************************************************************************
# Bending magnets
#***********************************************************************************

def bm_magnetic_field(
    magnet: dict,
    *,
    fringe: float = 0.0,
    step_size: float = 1e-3,
    gaussian_kernel: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Generate the 1D magnetic field profile of a single bending magnet.

    Parameters
    ----------
    magnet : dict
        Dictionary defining the bending magnet with:
          - 'B' : float
              Magnetic field amplitude [T].
          - 'length' : float
              Effective magnetic length [m].
          - 's0' : float, optional
              Center position [m]. Defaults to 0 if omitted.
    fringe : float, optional
        Zero-field padding [m] added on both sides of the magnet. Default 0.
    step_size : float, optional
        Sampling step [m]. Default 1e-3.
    gaussian_kernel : float, optional
        Gaussian σ [m] for smoothing. 0 disables smoothing.
    verbose : bool, optional
        Print magnet info if True.

    Returns
    -------
    dict
        {
            's': np.ndarray,  # 1D coordinate [m]
            'B': np.ndarray,  # (N,3), with (Bx=0, By, Bz=0)
        }
    """

    if not isinstance(magnet, dict):
        raise TypeError("magnet must be a dictionary with keys 'B', 'length', and optionally 's0'.")

    B = float(magnet["B"])
    length = float(magnet["length"])
    center = float(magnet.get("s0", 0.0))

    if length <= 0:
        raise ValueError("length must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if fringe < 0:
        raise ValueError("fringe must be non-negative.")
    if gaussian_kernel < 0:
        raise ValueError("gaussian_kernel must be non-negative.")

    half_active = 0.5 * length
    half_total = half_active + fringe
    s_min = center - half_total
    s_max = center + half_total

    total_span = s_max - s_min
    n_steps = max(1, int(round(total_span / step_size)))
    s_max = s_min + n_steps * step_size
    s = np.linspace(s_min, s_max, n_steps + 1)

    left = center - half_active
    right = center + half_active

    By = np.zeros_like(s, dtype=float)
    mask = (s >= left) & (s <= right)
    By[mask] = B

    if gaussian_kernel > 0:
        sigma_samples = gaussian_kernel / step_size
        By = gaussian_filter1d(By, sigma=sigma_samples, mode='nearest')

    B_vec = np.column_stack([np.zeros_like(s), By, np.zeros_like(s)])

    if verbose:
        print(f"Grid center (array midpoint): s = {center:.6f} m")
        print(f"Grid span: [{s[0]:.6f}, {s[-1]:.6f}] m")
        print(f"Active BM region: [{left:.6f}, {right:.6f}] m  (length = {length:.6f} m)")
        print(f"Padding per side: {fringe:.6f} m; step_size: {step_size:.6g} m; N = {s.size}\n")

    return {'s': s, 'B': B_vec}

def multi_bm_magnetic_field(
    magnets: list,
    fringe: float = 0.0,
    step_size: float = 1e-3,
    gaussian_kernel: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Construct a composite magnetic field from multiple bending magnet elements.

    Each bending magnet is specified by its magnetic field amplitude, effective length,
    and center position along a shared global axis. The total field is the superposition
    of all bending magnet contributions.

    Parameters
    ----------
    magnets : list of dict
        Each dictionary defines one bending magnet with:
          - 'B' : float, field amplitude [T]
          - 'length' : float, magnetic length [m]
          - 's0' : float, center position [m]
          Optional:
          - 'fringe' : float, per-magnet padding [m]
          - 'gaussian_kernel' : float, per-magnet σ [m]
    fringe : float, optional
        Global padding [m] at both ends of the assembled field. Default is 0.
    step_size : float, optional
        Sampling step along the global s-axis [m]. Default is 1e-3.
    gaussian_kernel : float, optional
        Default Gaussian sigma [m] if not defined per magnet. Default is 0.
    verbose : bool, optional
        If True, prints individual magnet parameters during assembly.

    Returns
    -------
    dict of str : np.ndarray
        Composite magnetic field dictionary:
          - 's' : ndarray, shape (N,)
              Global s-axis centered at 0 [m].
          - 'B' : ndarray, shape (N, 3)
              Superimposed magnetic field vectors [T], with (Bx=0, By, Bz=0).
    """
    import numpy as np

    if not isinstance(magnets, (list, tuple)) or not magnets:
        raise ValueError("magnets must be a non-empty list of dictionaries.")

    s_min_all, s_max_all = np.inf, -np.inf
    for m in magnets:
        if not all(k in m for k in ("B", "length", "s0")):
            raise ValueError("Each magnet must have 'B', 'length', and 's0'.")
        L, s0 = float(m["length"]), float(m["s0"])
        s_min_all = min(s_min_all, s0 - L / 2)
        s_max_all = max(s_max_all, s0 + L / 2)

    s_min_raw = s_min_all - abs(fringe)
    s_max_raw = s_max_all + abs(fringe)
    half_span = max(abs(s_min_raw), abs(s_max_raw))
    s = np.arange(-half_span, half_span + step_size / 2, step_size)

    By_total = np.zeros_like(s, dtype=float)

    for m in magnets:
        bm = bm_magnetic_field(
            magnet=m,
            fringe=m.get("fringe", fringe),
            step_size=step_size,
            gaussian_kernel=m.get("gaussian_kernel", gaussian_kernel),
            verbose=verbose,
        )

        local_s = bm["s"]
        local_By = bm["B"][:, 1]
        idx = np.round((local_s - s[0]) / step_size).astype(int)
        good = (idx >= 0) & (idx < s.size)
        np.add.at(By_total, idx[good], local_By[good])

    B_vec = np.column_stack([np.zeros_like(s), By_total, np.zeros_like(s)])

    return {"s": s, "B": B_vec}

def multi_arb_magnetic_field(
    magnets: list,
    fringe: float = 0.0,
    step_size: float = 1e-3,
    gaussian_kernel: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Assemble several arbitrary magnetic-field profiles (from text files or
    other sources) into one composite field along a shared global axis.

    Each element defines its own 1D field array and center position.
    All fields are resampled onto a common, symmetric grid and summed.

    Parameters
    ----------
    magnets : list of dict
        Each dict represents one magnetic element with:
          - 's'  : ndarray, local coordinate axis [m], typically centered at 0.
          - 'B'  : ndarray, (N,) or (N,3); vertical field in By or column 1.
          - 's0' : float,  center position on the global axis [m].

    fringe : float, optional
        Extra padding [m] on both sides of the global grid. Default is 0.
    step_size : float, optional
        Step size [m] for the uniform global grid. Default is 1e-3.
    gaussian_kernel : float, optional
        Gaussian σ [m] for optional smoothing of the *composite* field.
        If 0, no smoothing. Default is 0.
    verbose : bool, optional
        If True, prints per-element diagnostics and global summary.

    Returns
    -------
    dict
        {
            's': np.ndarray,  # uniform, centered at 0 [m]
            'B': np.ndarray,  # (N,3) composite field (Bx=0, By, Bz=0)
        }
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if fringe < 0:
        raise ValueError("fringe must be non-negative.")
    if gaussian_kernel < 0:
        raise ValueError("gaussian_kernel must be non-negative.")
    if not magnets:
        raise ValueError("magnets list is empty.")

    s_all = []
    for m in magnets:
        if not all(k in m for k in ("s", "B", "s0")):
            raise ValueError("Each magnet must have keys 's', 'B', and 's0'.")
        s_local = np.asarray(m["s"], float)
        s_all.append(s_local + float(m["s0"]))
    s_all = np.concatenate(s_all)
    s_min_raw, s_max_raw = s_all.min() - fringe, s_all.max() + fringe
    half_span = max(abs(s_min_raw), abs(s_max_raw))
    s = np.arange(-half_span, half_span + step_size/2, step_size)

    By_total = np.zeros_like(s, float)

    for i, m in enumerate(magnets, start=1):
        s_local = np.asarray(m["s"], float)
        s0 = float(m["s0"])
        B_local = np.asarray(m["B"], float)
        if B_local.ndim == 2:
            B_local = B_local[:, 1]
        order = np.argsort(s_local)
        s_local, B_local = s_local[order], B_local[order]
        s_shift = s_local + s0

        By_interp = np.interp(s, s_shift, B_local, left=0.0, right=0.0)
        By_total += By_interp

        if verbose:
            ds = np.median(np.diff(s_local))
            print(f"[Magnet {i}]")
            print(f"  Center: s0 = {s0:+.6f} m")
            print(f"  Points: N = {s_local.size}, Δs ≈ {ds:.3e} m")
            print(f"  s-range (shifted): [{s_shift.min():+.3f}, {s_shift.max():+.3f}] m")
            print(f"  B-range: [{B_local.min():+.3e}, {B_local.max():+.3e}] T\n")

    if gaussian_kernel > 0:
        sigma_samples = gaussian_kernel / step_size
        By_total = gaussian_filter1d(By_total, sigma=sigma_samples, mode="nearest")

    B_vec = np.column_stack([np.zeros_like(By_total), By_total, np.zeros_like(By_total)])

    if verbose:
        print(f"Global grid: [{s[0]:+.3f}, {s[-1]:+.3f}] m")
        print(f"  step_size = {step_size:.3e} m, total N = {s.size}")
        print(f"  Gaussian σ = {gaussian_kernel:.3e} m\n")

    return {"s": s, "B": B_vec}

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
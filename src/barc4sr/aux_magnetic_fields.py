#!/bin/python

"""
This module provides several auxiliary magnetic field functions
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '2024.11.25'
__changed__ = '2025.11.04'

import math
import os
from typing import List

import numpy as np
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
    padding: float = 0.0,
    step_size: float = 1e-3,
    verbose: bool = False,
) -> dict:
    """
    Generate the 1D magnetic-field profile of a single bending magnet with analytic soft edges.

    Parameters
    ----------
    magnet : dict
        Magnet specification:
            {
                "B": float,           # plateau field [T]
                "length": float,      # effective magnetic length [m]
                "s0": float, optional # center position [m] (default 0.0)
                "soft_edge": {        # optional, see below
                    "mode": {"none","tanh","erf","arctan","gompertz","srw"},
                    "edge_length": float,  # 10-90% rise distance [m]
                    "center_on": {"midpoint","left","right"}  # default "midpoint"
                }
            }
        Notes on modes:
        - "tanh": symmetric Bashmakov-like edge (tanh in dimensional x; no explicit gap h).
        - "erf" : Gaussian CDF edge (equivalent to Gaussian-convolved step).
        - "arctan": smooth with slightly longer far tails than erf.
        - "gompertz": flipped Gompertz with directional asymmetry:
            rising 0→1 has long left tail; falling 1 to 0 has long right tail.
        - "srw": SRW default - super-Lorentzian (order 2) CDF.
                 S(u) = 1/2 + (1/pi)[ arctan(u) + u/(1+u^2) ], u = x/d.
                 We map d from the requested 10-90% edge_length.

    padding : float, optional
        Zero-field padding [m] added on both sides of the magnet (default 0).
    step_size : float, optional
        Sampling step [m] (default 1e-3).
    verbose : bool, optional
        If True, prints a short summary.

    Returns
    -------
    dict
        {
            "s": np.ndarray,      # 1D axis [m]
            "B": np.ndarray,      # shape (N,3): (Bx=0, By, Bz=0)
        }

    Raises
    ------
    TypeError
        If `magnet` is not a dict with required keys.
    ValueError
        If parameters are invalid (e.g., non-positive length/step) or soft_edge fails validation.
    """

    if not isinstance(magnet, dict):
        raise TypeError("magnet must be a dictionary with keys 'B', 'length', and optionally 's0'/'soft_edge'.")
    try:
        B = float(magnet["B"])
        length = float(magnet["length"])
    except Exception as e:
        raise TypeError("magnet['B'] and magnet['length'] must be provided and castable to float.") from e

    center = float(magnet.get("s0", 0.0))
    if length <= 0:
        raise ValueError("length must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")

    soft_edge = magnet.get("soft_edge", None)

    def _validate_soft_edge(se: dict | None) -> dict:
        if not se:
            return {"mode": "none"}
        if not isinstance(se, dict):
            raise ValueError("magnet['soft_edge'] must be a dict.")
        mode = str(se.get("mode", "none")).lower()
        if mode not in {"none","tanh","erf","arctan","gompertz","srw"}:
            raise ValueError("soft_edge['mode'] must be one of {'none','tanh','erf','arctan','gompertz','srw'}.")
        if mode == "none":
            return {"mode": "none"}
        if "edge_length" not in se:
            raise ValueError("soft_edge requires 'edge_length' when mode != 'none'.")
        edge_length = float(se["edge_length"])
        if edge_length <= 0:
            # treat as hard edge for robustness
            return {"mode": "none"}
        center_on = str(se.get("center_on", "midpoint")).lower()
        if center_on not in {"midpoint","left","right"}:
            raise ValueError("soft_edge['center_on'] must be 'midpoint', 'left', or 'right'.")
        return {"mode": mode, "edge_length": edge_length, "center_on": center_on}

    se = _validate_soft_edge(soft_edge)

    half_active = 0.5 * length
    half_total = half_active + padding
    s_min = center - half_total
    s_max = center + half_total
    total_span = s_max - s_min
    n_steps = max(1, int(round(total_span / step_size)))
    s_max = s_min + n_steps * step_size
    s = np.linspace(s_min, s_max, n_steps + 1)

    L_edge = center - half_active
    R_edge = center + half_active

    # -------------------------
    # Soft-edge constants (edge_length is the 10–90 width for a centered edge)
    # -------------------------
    C_TANH = 2.197224        # tanh      : edge_length = C_TANH * lam
    C_ERF  = 2.563103        # erf       : edge_length = C_ERF  * sigma
    C_ATAN = 6.155367        # arctan    : edge_length = C_ATAN * a
    C_GOMP = 3.085323        # Gompertz  : edge_length = C_GOMP / c  (flipped)
    C_SRW  = 1.89110428694   # SRW CDF   : edge_length = C_SRW  * d
    LN2    = float(np.log(2.0))

    # -------------------------
    # Edge factories: return (S_rise, S_fall)
    # -------------------------
    def _sigmoid_factory_pair(mode: str, edge_length_val: float):
        """
        Return two vectorized callables mapping R -> [0,1]:
          S_rise(x): increasing  0→1 with the given 10-90 width when centered at x=0
          S_fall(x): decreasing 1→0 with mirrored asymmetry (important for Gompertz)
        """
        delta = float(edge_length_val)
        m = mode.lower()
        if m == "none" or delta <= 0:
            return None, None

        if m == "tanh":
            lam = delta / C_TANH
            S_rise = lambda x: 0.5 * (1.0 + np.tanh(x / lam))
            S_fall = lambda x: 0.5 * (1.0 + np.tanh(-x / lam))
            return S_rise, S_fall

        if m == "erf":
            sigma = delta / C_ERF
            denom = sigma * np.sqrt(2.0)
            verf = np.vectorize(math.erf)
            S_rise = lambda x: 0.5 * (1.0 + verf(x / denom))
            S_fall = lambda x: 0.5 * (1.0 + verf(-x / denom))
            return S_rise, S_fall

        if m == "arctan":
            a = delta / C_ATAN
            S_rise = lambda x: 0.5 + (1.0 / np.pi) * np.arctan(x / a)
            S_fall = lambda x: 0.5 + (1.0 / np.pi) * np.arctan(-x / a)
            return S_rise, S_fall

        if m == "gompertz":
            # Flipped Gompertz with correct opposite asymmetry on fall:
            #   rise: S↑(x) = 1 - exp(-ln2 * exp(+c x))  → long left, short right
            #   fall: S↓(x) = 1 - exp(-ln2 * exp(-c x))  → short left, long right
            c = C_GOMP / delta
            S_rise = lambda x: 1.0 - np.exp(-LN2 * np.exp(+c * x))
            S_fall = lambda x: 1.0 - np.exp(-LN2 * np.exp(-c * x))
            return S_rise, S_fall

        if m == "srw":
            # SRW default: super-Lorentzian (order 2) CDF
            d = delta / C_SRW
            def S_cdf(x):
                u = x / d
                return 0.5 + (1.0 / np.pi) * (np.arctan(u) + u / (1.0 + u * u))
            S_rise = lambda x: S_cdf(x)
            S_fall = lambda x: S_cdf(-x)
            return S_rise, S_fall

        raise ValueError(f"Unknown soft-edge mode: {mode}")

    By = np.zeros_like(s, dtype=float)

    if se["mode"] == "none":
        By[(s >= L_edge) & (s <= R_edge)] = B
    else:
        S_rise, S_fall = _sigmoid_factory_pair(se["mode"], se["edge_length"])
        if S_rise is None or S_fall is None:
            By[(s >= L_edge) & (s <= R_edge)] = B
        else:
            delta = se["edge_length"]
            center_on = se["center_on"]
            if center_on == "midpoint":
                cL, cR = L_edge, R_edge                       # 50% at nominal edges
            elif center_on == "left":
                cL, cR = L_edge + 0.5 * delta, R_edge + 0.5 * delta   # 10% at nominal edges
            else:  # "right"
                cL, cR = L_edge - 0.5 * delta, R_edge - 0.5 * delta   # 90% at nominal edges

            FL = S_rise(s - cL)   # left edge: 0 to 1
            FR = S_fall(s - cR)   # right edge: 1 to 0 (mirrored asymmetry for gompertz)
            By = B * FL * FR

    B_vec = np.column_stack([np.zeros_like(s), By, np.zeros_like(s)])

    if verbose:
        ds = np.median(np.diff(s))
        print(f"s-axis center: {center:.6f} m | span: [{s[0]:.6f}, {s[-1]:.6f}] m")
        print(f"sampling: N = {s.size} | step: {ds:.3e} m")
        print(f"nominal BM length: length = {length:.6f} m | span: [{L_edge:.6f}, {R_edge:.6f}] m")
        print(f"mag. field range: [{By.min():+.3e}, {By.max():+.3e}] T")
        print(f"soft_edge: {se}")
        print(f"padding = {padding:.6f} m")

    return {"s": s, "B": B_vec}

def multi_bm_magnetic_field(
    magnets: List[dict],
    padding: float = 0.0,
    step_size: float = 1e-3,
    verbose: bool = False,
) -> dict:
    """
    Construct a composite magnetic field from multiple bending-magnet elements.

    Each magnet is defined like in `bm_magnetic_field`, including an optional
    per-magnet soft-edge specification nested inside the magnet dict.

    Parameters
    ----------
    magnets : list of dict
        Each dictionary defines one bending magnet:
          {
            "B": float,            # plateau field [T]
            "length": float,       # effective magnetic length [m]
            "s0": float,           # center position [m]
            # optional:
            "padding": float,       # per-magnet padding [m] (defaults to global `padding`)
            "soft_edge": {          # OPTIONAL, same structure as in bm_magnetic_field
               "mode": "none" | "tanh" | "erf" | "arctan" | "gompertz" | "srw",
               "edge_length": float,  # 10-90 width [m] (required if mode != "none")
               "center_on": "midpoint" | "left" | "right"
            }
          }

        Notes
        -----
        - "tanh" is a symmetric Bashmakov-like edge (dimensional tanh; no gap h).
        - "srw" uses the super-Lorentzian (order 2) CDF (SRW default padding family).
        - For preserving the full plateau length, set `soft_edge.center_on="right"`.

    padding : float, optional
        Global zero-field padding [m] added at both ends of *each* magnet's local grid
        unless a per-magnet "padding" overrides it. Default 0.
    step_size : float, optional
        Sampling step along the global s-axis [m]. Default 1e-3.
    verbose : bool, optional
        If True, prints per-magnet summaries from `bm_magnetic_field`.

    Returns
    -------
    dict
        {
            "s": np.ndarray,      # global axis [m], symmetric around 0
            "B": np.ndarray,      # shape (N,3): (Bx=0, By, Bz=0), composite field
        }

    Raises
    ------
    ValueError
        If input list is empty or malformed.
    """

    if not isinstance(magnets, (list, tuple)) or not magnets:
        raise ValueError("magnets must be a non-empty list of dictionaries.")

    s_min_all, s_max_all = np.inf, -np.inf
    for m in magnets:
        if not all(k in m for k in ("B", "length", "s0")):
            raise ValueError("Each magnet must define 'B', 'length', and 's0'.")
        L = float(m["length"])
        s0 = float(m["s0"])
        f_loc = float(m.get("padding", padding))
        half_total = 0.5 * L + max(f_loc, 0.0)
        s_min_all = min(s_min_all, s0 - half_total)
        s_max_all = max(s_max_all, s0 + half_total)

    s_min_raw = s_min_all
    s_max_raw = s_max_all
    half_span = max(abs(s_min_raw), abs(s_max_raw))
    s = np.arange(-half_span, half_span + 0.5 * step_size, step_size)

    By_total = np.zeros_like(s, dtype=float)

    for i, m in enumerate(magnets):
        if verbose:
            print(f"\n[Magnet {i+1}]")
        f_loc = float(m.get("padding", padding))
        bm = bm_magnetic_field(
            magnet=m,
            padding=f_loc,
            step_size=step_size,
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
    padding: float = 0.0,
    step_size: float = 1e-3,
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

    padding : float, optional
        Extra padding [m] on both sides of the global grid. Default is 0.
    step_size : float, optional
        Step size [m] for the uniform global grid. Default is 1e-3.
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
    if padding < 0:
        raise ValueError("padding must be non-negative.")
    if not magnets:
        raise ValueError("magnets list is empty.")

    s_all = []
    for m in magnets:
        if not all(k in m for k in ("s", "B", "s0")):
            raise ValueError("Each magnet must have keys 's', 'B', and 's0'.")
        s_local = np.asarray(m["s"], float)
        s_all.append(s_local + float(m["s0"]))
    s_all = np.concatenate(s_all)
    s_min_raw, s_max_raw = s_all.min() - padding, s_all.max() + padding
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
            print(f"  s-axis center: {s0:+.6f} m | span: [{s_shift.min():+.3f}, {s_shift.max():+.3f}] m")
            print(f"  sampling: N = {s_local.size} | step: {ds:.3e} m")
            print(f"  mag. field range: length = [{B_local.min():+.3e}, {B_local.max():+.3e}] T")
            print(f"  padding = {padding:.6f} m")

    B_vec = np.column_stack([np.zeros_like(By_total), By_total, np.zeros_like(By_total)])

    if verbose:
        print(f"Global grid: [{s[0]:+.3f}, {s[-1]:+.3f}] m")
        print(f"  step_size = {step_size:.3e} m, total N = {s.size}")

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
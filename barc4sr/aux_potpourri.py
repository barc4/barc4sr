#!/bin/python

"""
This module provides several auxiliary functions
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

import numpy as np
from scipy.signal import find_peaks


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
    
       
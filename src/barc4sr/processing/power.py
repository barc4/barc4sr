# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Module for processing power density maps.
"""

from __future__ import annotations

from copy import copy

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def integrate_power_density_window(pwrDict, hor_slit, ver_slit):
    """
    Integrates the power density maps in pwrDict within a rectangular window.

    Parameters:
        pwrDict (dict): Dictionary containing power density data as produced by write_power_density.
        hor_slit (tuple or float): (x_min, x_max) limits of window, or positive float interpreted as (-w/2, +w/2).
        ver_slit (tuple or float): (y_min, y_max) limits of window, or positive float interpreted as (-w/2, +w/2).

    Returns:
        dict: Dictionary with structure {polarisation: integrated_value, ...}, plus the window used.
    """

    x_axis = pwrDict['axis']['x']
    y_axis = pwrDict['axis']['y']

    if isinstance(hor_slit, (int, float)):
        if hor_slit <= 0:
            raise ValueError("hor_slit must be positive if passed as float.")
        half_w = hor_slit / 2
        hor_slit = (-half_w, half_w)

    if isinstance(ver_slit, (int, float)):
        if ver_slit <= 0:
            raise ValueError("ver_slit must be positive if passed as float.")
        half_w = ver_slit / 2
        ver_slit = (-half_w, half_w)

    x_min, x_max = hor_slit
    y_min, y_max = ver_slit

    x_mask = (x_axis >= x_min) & (x_axis <= x_max)
    y_mask = (y_axis >= y_min) & (y_axis <= y_max)

    dx = (x_axis[1] - x_axis[0]) * 1e3  # mm
    dy = (y_axis[1] - y_axis[0]) * 1e3  # mm

    results = {'hor_slit': hor_slit, 'ver_slit': ver_slit}

    for pol in pwrDict:
        if pol == 'axis':
            continue

        map2d = copy(pwrDict[pol]['map'])
        subarray = map2d[np.ix_(y_mask, x_mask)]
        integrated = np.sum(subarray) * dx * dy
        results[pol] = integrated

    return results


def trim_and_resample_power_density(pwrDict,  **kwargs):
    """
    Trims and optionally resamples the power density maps for all polarisations in pwrDict.

    Parameters:
        pwrDict (Dict): Dictionary containing power density data with keys:
            - 'axis': dict with 'x' and 'y' axes arrays.
            - polarisations: each with 'map' (2D array), 'integrated' (W), and 'peak' (W/mm^2).
        **kwargs:
            - dx (float): Width in x-direction for trimming (default: full x range).
            - dy (float): Width in y-direction for trimming (default: full y range).
            - xc (float): Center of trimming region along x (default: 0).
            - yc (float): Center of trimming region along y (default: 0).
            - X (array_like): New x-axis values for resampling.
            - Y (array_like): New y-axis values for resampling.

    Returns:
        Dict: Trimmed and resampled power density dictionary, same structure as pwrDict.
    """

    x = pwrDict["axis"]["x"]
    y = pwrDict["axis"]["y"]

    # Extract kwargs with defaults
    dx = kwargs.get("dx", x[-1] - x[0])
    dy = kwargs.get("dy", y[-1] - y[0])
    xc = kwargs.get("xc", 0)
    yc = kwargs.get("yc", 0)
    X = kwargs.get("X", None)
    Y = kwargs.get("Y", None)

    interpol = X is not None or Y is not None

    resultDict = {'axis': {}}

    if interpol:
        print(">>>>> Interpolating power density maps...")

        if X is None:
            X = x
        if Y is None:
            Y = y

        ygrid, xgrid = np.meshgrid(Y, X, indexing='ij')
        new_x = X
        new_y = Y

        for pol in pwrDict:
            if pol == 'axis':
                continue

            pow_map = pwrDict[pol]['map']
            f = RegularGridInterpolator((y, x), pow_map, bounds_error=False, fill_value=0)
            pow_map_interp = f((ygrid, xgrid))

            dx_new = (new_x[1] - new_x[0]) * 1e3 if len(new_x) > 1 else 0
            dy_new = (new_y[1] - new_y[0]) * 1e3 if len(new_y) > 1 else 0

            integrated = np.sum(pow_map_interp) * dx_new * dy_new
            peak = pow_map_interp.max()

            resultDict[pol] = {'map': pow_map_interp, 'integrated': integrated, 'peak': peak}

        resultDict['axis']['x'] = new_x
        resultDict['axis']['y'] = new_y

    else:
        deltax = x[1] - x[0]
        deltay = y[1] - y[0]

        x_mask = (x >= xc - dx/2 - deltax/20) & (x <= xc + dx/2 + deltax/20)
        y_mask = (y >= yc - dy/2 - deltay/20) & (y <= yc + dy/2 + deltay/20)

        new_x = x[x_mask] - xc
        new_y = y[y_mask] - yc

        dx_new = (new_x[1] - new_x[0]) * 1e3 if len(new_x) > 1 else 0
        dy_new = (new_y[1] - new_y[0]) * 1e3 if len(new_y) > 1 else 0

        for pol in pwrDict:
            if pol == 'axis':
                continue

            pow_map = pwrDict[pol]['map']
            pow_map_trimmed = pow_map[np.ix_(y_mask, x_mask)]

            integrated = np.sum(pow_map_trimmed) * dx_new * dy_new
            peak = pow_map_trimmed.max()

            resultDict[pol] = {'map': pow_map_trimmed, 'integrated': integrated, 'peak': peak}

        resultDict['axis']['x'] = new_x
        resultDict['axis']['y'] = new_y

    for pol in resultDict:
        if pol == 'axis':
            continue
        print(f"{pol} - Total power: {resultDict[pol]['integrated']:.3f} W, "
              f"Peak density: {resultDict[pol]['peak']:.3f} W/mm^2")

    return resultDict
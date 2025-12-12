# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Module for processing wavefront data.
"""

from __future__ import annotations

from copy import copy

import numpy as np


def integrate_wavefront_window(wfrDict, hor_slit, ver_slit, observation_plane=None):
    """
    Integrates the wavefront intensity within a rectangular window.
    
    Parameters
    ----------
    wfrDict : dict
        Dictionary containing wavefront data as produced by write_wavefront.
    hor_slit : tuple or float
        If observation_plane is None: (x_min, x_max) limits of window [m], 
            or positive float interpreted as (-w/2, +w/2).
        If observation_plane is provided: (x_min, x_max) limits in [rad],
            or positive float interpreted as (-w/2, +w/2).
    ver_slit : tuple or float
        If observation_plane is None: (y_min, y_max) limits of window [m],
            or positive float interpreted as (-w/2, +w/2).
        If observation_plane is provided: (y_min, y_max) limits in [rad],
            or positive float interpreted as (-w/2, +w/2).
    observation_plane : float, optional
        Distance to observation plane in meters. If provided, hor_slit and ver_slit
        are interpreted as angles in radians and converted to meters (default: None).
    
    Returns
    -------
    dict
        Dictionary with structure {polarisation: integrated_intensity, ...}, 
        plus the window used.
    """
    x_axis = wfrDict['axis']['x']
    y_axis = wfrDict['axis']['y']
    
    if isinstance(hor_slit, (int, float)):
        if hor_slit <= 0:
            raise ValueError("hor_slit must be positive if passed as float.")
        half_w = hor_slit / 2
        hor_slit = (-half_w, half_w)
    
    if isinstance(ver_slit, (int, float)):
        if ver_slit <= 0:
            raise ValueError("ver_slit must be positive if passed as float.")
        half_w = ver_slit / 2
        ver_slit = (-half_w, +half_w)
    
    if observation_plane is not None:
        x_min = 2 * observation_plane * np.tan(hor_slit[0] / 2)
        x_max = 2 * observation_plane * np.tan(hor_slit[1] / 2)
        y_min = 2 * observation_plane * np.tan(ver_slit[0] / 2)
        y_max = 2 * observation_plane * np.tan(ver_slit[1] / 2)
        hor_slit_m = (x_min, x_max)
        ver_slit_m = (y_min, y_max)
    else:
        hor_slit_m = hor_slit
        ver_slit_m = ver_slit
        x_min, x_max = hor_slit
        y_min, y_max = ver_slit

    x_mask = (x_axis >= x_min) & (x_axis <= x_max)
    y_mask = (y_axis >= y_min) & (y_axis <= y_max)
    
    dx = (x_axis[1] - x_axis[0]) * 1e3
    dy = (y_axis[1] - y_axis[0]) * 1e3
    
    results = {'hor_slit': hor_slit_m, 'ver_slit': ver_slit_m}
    
    for pol in wfrDict['intensity']:
        map2d = copy(wfrDict['intensity'][pol])
        subarray = map2d[np.ix_(y_mask, x_mask)]
        integrated = np.sum(subarray) * dx * dy
        results[pol] = integrated
    
    return results
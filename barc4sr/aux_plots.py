#!/bin/python

"""
This module provides auxiliary functions for eplotting barc4sr data
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

import numpy as np
from scipy.constants import physical_constants

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]



def plot_electron_trajectory(eBeamTraj, **kwargs):
    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    file_name = kwargs.get("file_name", None)

    img = PlotManager(eBeamTraj["eTraj"]["X"]*1E6,  eBeamTraj["eTraj"]["Z"])
    img.additional_info('electron trajectory', "s [m]",  "[µm]", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    img.aesthetics(400, True, True, 0, 1, True, 4).info_1d_plot(0, 'hor.', 0, "-", False, 0, 1).plot_1d(enable=False)
    img.image = eBeamTraj["eTraj"]["Y"]*1E6
    img.info_1d_plot(1, 'ver.', 1, "-", False, 0, 1).plot_1d(file_name=file_name, enable=True, hold=True)
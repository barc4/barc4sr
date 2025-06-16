#!/bin/python

"""
This module provides auxiliary functions for plotting barc4sr data
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '13/JUN/2025'
__changed__ = '16/JUN/2025'

import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rcParamsDefault


def start_plotting(k: float = 1.0) -> None:
    """
    Set global Matplotlib plot parameters scaled by factor k.

    Args:
        k: Scaling factor for font sizes.
    """
    plt.rcParams.update(rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DeJavu Serif",
        "font.serif": ["Times New Roman"]
    })
    plt.rc('axes', titlesize=15 * k)
    plt.rc('axes', labelsize=14 * k)
    plt.rc('xtick', labelsize=13 * k)
    plt.rc('ytick', labelsize=13 * k)
    plt.rc('legend', fontsize=12 * k)


def plot_electron_trajectory(eBeamTraj: dict, direction: str, **kwargs) -> None:
    """
    Plot horizontal or vertical electron trajectory from simulation data.

    Args:
        eBeamTraj: Dictionary containing electron trajectory data.
        direction: 'horizontal', 'vertical', or 'both'.
        **kwargs: Optional keyword arguments, e.g., scaling factor `k`.
    """
    k = kwargs.get('k', 1)
    start_plotting(k)

    colors = ['firebrick', 'olive', 'steelblue']
    s = eBeamTraj['eTraj']['Z']

    if direction.lower() in ['x', 'h', 'hor', 'horizontal']:
        B = eBeamTraj['eTraj']['By']
        graph_title = 'Horizontal electron trajectory'
        axis = 'X'
    elif direction.lower() in ['y', 'v', 'ver', 'vertical']:
        B = eBeamTraj['eTraj']['Bx']
        graph_title = 'Vertical electron trajectory'
        axis = 'Y'
    elif direction.lower() in ['b', 'both']:
        plot_electron_trajectory(eBeamTraj, 'horizontal', **kwargs)
        plot_electron_trajectory(eBeamTraj, 'vertical', **kwargs)
        return
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'.")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8), height_ratios=[1, 1, 1])
    fig.suptitle(graph_title, fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    ys = [
        B,
        eBeamTraj['eTraj'][axis] * 1e3,   # mm
        eBeamTraj['eTraj'][f'{axis}p'] * 1e3  # mrad
    ]
    labels = ['B [T]', f'{axis.lower()} [mm]', f"{axis.lower()}' [mrad]"]

    for ax, y, color, label in zip(axes, ys, colors, labels):
        ax.set_facecolor('white')
        ax.plot(s, y, color=color, linewidth=1.5, label=label)
        ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
        ax.tick_params(direction='in', top=True, right=True)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.legend(loc='upper right', frameon=True)

    axes[-1].set_xlabel('[mm]', color='black')
    plt.tight_layout()
    plt.show()
            

def plot_magnetic_field(eBeamTraj: dict, direction: str, **kwargs) -> None:
    """
    Plot horizontal or vertical magnetic field from simulation data.

    Args:
        eBeamTraj: Dictionary containing electron trajectory data.
        direction: 'horizontal', 'vertical', or 'both'.
        **kwargs: Optional keyword arguments, e.g., scaling factor `k`.
    """
    k = kwargs.get('k', 1)
    start_plotting(k)

    colors = ['firebrick', 'olive', 'steelblue']
    s = eBeamTraj['eTraj']['Z']

    if direction.lower() in ['x', 'h', 'hor', 'horizontal']:
        B = eBeamTraj['eTraj']['By']
        graph_title = 'Horizontal magnetic field'
        axis = 'X'
    elif direction.lower() in ['y', 'v', 'ver', 'vertical']:
        B = eBeamTraj['eTraj']['Bx']
        graph_title = 'Vertical magnetic field'
        axis = 'Y'
    elif direction.lower() in ['b', 'both']:
        plot_magnetic_field(eBeamTraj, 'horizontal', **kwargs)
        plot_magnetic_field(eBeamTraj, 'vertical', **kwargs)
        return
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'.")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8), height_ratios=[1, 1, 1])
    fig.suptitle(graph_title, fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    frst_field_integral = integrate.cumulative_trapezoid(B, s, initial=0)
    scnd_field_integral = integrate.cumulative_trapezoid(frst_field_integral, s, initial=0)

    ys = [
        B,
        frst_field_integral,
        scnd_field_integral
    ]
    labels = ['B [T]', '$\int$B$\cdot$d$s$ [T$\cdot$m]', '$\iint$B$\cdot$d$s$² [T$\cdot$m²]']

    for ax, y, color, label in zip(axes, ys, colors, labels):
        ax.set_facecolor('white')
        ax.plot(s, y, color=color, linewidth=1.5, label=label)
        ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
        ax.tick_params(direction='in', top=True, right=True)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.legend(loc='upper right', frameon=True)

    axes[-1].set_xlabel('[mm]', color='black')
    plt.tight_layout()
    plt.show()

# def plot_magnetic_field(eBeamTraj_srw, **kwargs):
#     xmin = kwargs.get("xmin", None)
#     xmax = kwargs.get("xmax", None)
#     ymin = kwargs.get("ymin", None)
#     ymax = kwargs.get("ymax", None)

#     integrals = kwargs.get("integrals", False)

#     file_name = kwargs.get("file_name", None)

#     img = PlotManager(eBeamTraj_srw["eTraj"]["Bx"],  eBeamTraj_srw["eTraj"]["Z"])
#     img.additional_info('magnetic field', "s [m]",  "B [T]", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
#     img.aesthetics(400, True, True, 0, 1, True, 4).info_1d_plot(0, 'hor.', 0, "-", False, 0, 1).plot_1d(enable=False)
#     img.image = eBeamTraj_srw["eTraj"]["By"]
#     img.info_1d_plot(1, 'ver.', 1, "-", False, 0, 1).plot_1d(file_name=file_name, enable=True, hold=True)

#     if integrals:
        
#         frst_field_integral_Bx = integrate.cumulative_trapezoid(eBeamTraj_srw["eTraj"]["Bx"],
#                                                                 eBeamTraj_srw["eTraj"]["Z"], initial=0)
#         scnd_field_integral_Bx = integrate.cumulative_trapezoid(frst_field_integral_Bx,
#                                                                 eBeamTraj_srw["eTraj"]["Z"], initial=0)
        
#         frst_field_integral_By = integrate.cumulative_trapezoid(eBeamTraj_srw["eTraj"]["By"],
#                                                                 eBeamTraj_srw["eTraj"]["Z"], initial=0)
#         scnd_field_integral_By = integrate.cumulative_trapezoid(frst_field_integral_By,
#                                                                 eBeamTraj_srw["eTraj"]["Z"], initial=0)
        
#         img = PlotManager(frst_field_integral_Bx,  eBeamTraj_srw["eTraj"]["Z"])
#         img.additional_info('first field integral', "s [m]",  "[T$\cdot$m]", xmin=xmin, xmax=xmax, ymin=None, ymax=None)
#         img.aesthetics(400, True, True, 0, 1, True, 4).info_1d_plot(2, 'hor.', 0, "-", False, 0, 1).plot_1d(enable=False)
#         img.image = frst_field_integral_By
#         img.info_1d_plot(3, 'ver.', 1, "-", False, 0, 1).plot_1d(file_name=file_name, enable=True, hold=True)

#         img = PlotManager(scnd_field_integral_Bx,  eBeamTraj_srw["eTraj"]["Z"])
#         img.additional_info('second field integral', "s [m]",  "[T$\cdot$m²]", xmin=xmin, xmax=xmax, ymin=None, ymax=None)
#         img.aesthetics(400, True, True, 0, 1, True, 4).info_1d_plot(4, 'hor.', 0, "-", False, 0, 1).plot_1d(enable=False)
#         img.image = scnd_field_integral_By
#         img.info_1d_plot(5, 'ver.', 1, "-", False, 0, 1).plot_1d(file_name=file_name, enable=True, hold=True)
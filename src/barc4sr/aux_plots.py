#!/bin/python

"""
This module provides auxiliary functions for plotting barc4sr data
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '13/JUN/2025'
__changed__ = '26/JUN/2025'

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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
    s = eBeamTraj['eTraj']['Z']*1E3

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
    s = eBeamTraj['eTraj']['Z']*1E3

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

def plot_wavefront(wfr: dict, cuts: bool = True, phase: bool = True, **kwargs) -> None:
    """
    Plot wavefront intensity and optionally phase from a wavefront dictionary.

    For each polarisation in the dictionary, this plots:
        - If cuts=True: 2D intensity map + horizontal (y=0) and vertical (x=0) cuts.
        - If cuts=False: Only the 2D intensity map.
    Phase map is shown at the end if requested.

    Args:
        wfr (dict): Dictionary returned by `write_wavefront` or `read_wavefront`.
        cuts (bool): Whether to include 1D cuts in the plot.
        **kwargs: Optional keyword arguments, e.g., scaling factor `k`.
    """
    k = kwargs.get('k', 1)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    start_plotting(k)

    x = wfr['axis']['x'] * 1e3  # mm
    y = wfr['axis']['y'] * 1e3  # mm
    X, Y = np.meshgrid(x, y)

    dx = wfr['axis']['x'][1]-wfr['axis']['x'][0]
    dy = wfr['axis']['y'][1]-wfr['axis']['y'][0]

    fctr =(wfr['axis']['x'][-1]-wfr['axis']['x'][0])/(wfr['axis']['y'][-1]-wfr['axis']['y'][0])
   
    for pol, data in wfr['intensity'].items():

        flux = np.sum(data*dx*1E3*dy*1E3)
        fig = plt.figure(figsize=(4.2*fctr, 4))
        fig.suptitle(f"Intensity ({pol}) - integrated flux: {flux:.2e} ph/s/0.1%bw", fontsize=16 * k, x=0.5)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046 * 1, pad=0.04, format='%.0e')
        plt.show()


        phase = wfr['phase'][pol]
        phase -=phase[phase.shape[0]//2, phase.shape[1]//2]
        Rx = wfr['wfr'].Rx
        Ry = wfr['wfr'].Ry
        fig = plt.figure(figsize=(4.2*fctr, 4))
        fig.suptitle(f"Residual phase ({pol}) - Rx = {Rx:.3f}m, Ry = {Ry:.3f}m", fontsize=16 * k, x=0.5)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, phase, shading='auto', cmap='bwr')#, vmin=-np.pi, vmax=np.pi)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046 * 1, pad=0.04)

        # cb = plt.colorbar(im, ax=ax, fraction=0.046 * 1, pad=0.04, spacing='uniform',
        # ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        # cb.ax.set_yticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$'])
        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            ax1.plot(x, data[iy0, :], color='darkred', lw=1.5)
            ax1.set_title('Hor. cut (y=0)')
            ax1.set_xlabel('x [mm]')
            ax1.set_ylabel('ph/s/mm²/0.1%bw')
            ax1.grid(True, linestyle=':', linewidth=0.5)
            ax1.tick_params(direction='in', top=True, right=True)
            ax2.plot(y, data[:, ix0], color='darkred', lw=1.5)
            ax2.set_title('Ver. cut (x=0)')
            ax2.set_xlabel('y [mm]')
            ax2.grid(True, linestyle=':', linewidth=0.5)
            ax2.tick_params(direction='in', top=True, right=True)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()      

def plot_power_density(pwr: dict, cuts: bool = True, **kwargs) -> None:
    """
    Plot power density maps from a power dictionary.

    For each polarisation in the dictionary, this plots:
        - If cuts=True: 2D power map + horizontal (y=0) and vertical (x=0) cuts.
        - If cuts=False: Only the 2D power map.

    Args:
        pwr (dict): Dictionary returned by `write_power_density` or `read_power_density`.
        cuts (bool): Whether to include 1D cuts in the plot.
        **kwargs: Optional keyword arguments, e.g., scaling factor `k`, vmin, vmax.
    """
    k = kwargs.get('k', 1)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    start_plotting(k)

    x = pwr['axis']['x'] * 1e3  # mm
    y = pwr['axis']['y'] * 1e3  # mm
    X, Y = np.meshgrid(x, y)

    fctr = (x[-1] - x[0]) / (y[-1] - y[0])

    for pol in [p for p in pwr if p != 'axis']:
        data = pwr[pol]['map']
        integrated = pwr[pol]['integrated']
        peak = pwr[pol]['peak']

        fig = plt.figure(figsize=(4.2 * fctr, 4))
        fig.suptitle(f"{pol} - Total: {integrated:.3e} W | Peak: {peak:.2f} W/mm²",
                     fontsize=16 * k, x=0.5)

        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1e')
        cb.set_label('Power Density [W/mm²]')
        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            ax1.plot(x, data[iy0, :], color='darkred', lw=1.5)
            ax1.set_title('Hor. cut (y=0)')
            ax1.set_xlabel('x [mm]')
            ax1.set_ylabel('Power density [W/mm²]')
            ax1.grid(True, linestyle=':', linewidth=0.5)
            ax1.tick_params(direction='in', top=True, right=True)

            ax2.plot(y, data[:, ix0], color='darkred', lw=1.5)
            ax2.set_title('Ver. cut (x=0)')
            ax2.set_xlabel('y [mm]')
            ax2.grid(True, linestyle=':', linewidth=0.5)
            ax2.tick_params(direction='in', top=True, right=True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
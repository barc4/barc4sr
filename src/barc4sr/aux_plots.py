#!/bin/python

"""
This module provides auxiliary functions for plotting barc4sr data
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '13/JUN/2025'
__changed__ = '07/JUL/2025'

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import cm, rcParamsDefault
from matplotlib.colors import LinearSegmentedColormap
from skimage.restoration import unwrap_phase

#***********************************************************************************
# General style
#***********************************************************************************

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

#***********************************************************************************
# Electron Trajectory
#***********************************************************************************

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

    colors = ['darkred', 'olive', 'steelblue']
    s = eBeamTraj['eTraj']['Z']*1E3

    if direction.lower() in ['x', 'h', 'hor', 'horizontal']:
        B = eBeamTraj['eTraj']['By']
        graph_title = 'Horizontal electron trajectory'
        axis = 'X'
        label_B = 'By [T]'
    elif direction.lower() in ['y', 'v', 'ver', 'vertical']:
        B = eBeamTraj['eTraj']['Bx']
        graph_title = 'Vertical electron trajectory'
        axis = 'Y'
        label_B = 'Bx [T]'
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

    labels = [label_B, f'{axis.lower()} [mm]', f"{axis.lower()}' [mrad]"]

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
            
#***********************************************************************************
# Magnetic field
#***********************************************************************************

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

    colors = ['darkred', 'olive', 'steelblue']
    s = eBeamTraj['eTraj']['Z']*1E3

    if direction.lower() in ['x', 'h', 'hor', 'horizontal']:
        B = eBeamTraj['eTraj']['By']
        graph_title = 'Vertical magnetic field'
        axis = 'X'
    elif direction.lower() in ['y', 'v', 'ver', 'vertical']:
        B = eBeamTraj['eTraj']['Bx']
        graph_title = 'Horizontal magnetic field'
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

def plot_field_and_twiss(data: dict, **kwargs) -> None:
    """
    Plot magnetic field components and Twiss parameters vs s.

    Parameters
    ----------
    data : dict
        Dictionary with:
        - 's' : array, positions [m], shape (N,)
        - 'B' : array, magnetic field vectors [T], shape (N, 3) for (Bx, By, Bz)
        - 'Twiss' : dict with:
            * 'beta_x' : array, horizontal beta [m], shape (N,)
            * 'beta_y' : array, vertical beta [m], shape (N,)
            * 'eta_x' or 'eta' : array, horizontal dispersion [m], shape (N,)
    **kwargs
        k : float, optional
            Global scaling factor for fonts (default matches aux_plots).

    Returns
    -------
    None
        The function only generates the plots.

    Raises
    ------
    KeyError
        If required keys are missing in `data` or `data['Twiss']`.
    ValueError
        If array shapes are inconsistent, dimensions are incorrect,
        or any values are non-finite.
    """
    k = kwargs.get('k', 1)
    try:
        start_plotting(k)  
    except NameError:
        pass 

    if 's' not in data or 'B' not in data or 'Twiss' not in data:
        raise KeyError("Missing required keys: 's', 'B', and 'Twiss' must be present.")

    s = np.asarray(data['s'], dtype=float)
    B = np.asarray(data['B'], dtype=float)

    if s.ndim != 1:
        raise ValueError("'s' must be 1D.")
    if B.ndim != 2 or B.shape[1] != 3 or B.shape[0] != s.size:
        raise ValueError("'B' must have shape (N, 3) with N == len(s).")
    if not (np.isfinite(s).all() and np.isfinite(B).all()):
        raise ValueError("'s' and 'B' must be finite.")

    tw = data['Twiss']
    for key in ('beta_x', 'beta_y'):
        if key not in tw:
            raise KeyError(f"Missing Twiss key: '{key}'.")
    eta_key = 'eta_x' if 'eta_x' in tw else ('eta' if 'eta' in tw else None)
    if eta_key is None:
        raise KeyError("Missing Twiss key: 'eta_x' (or 'eta').")

    beta_x = np.asarray(tw['beta_x'], dtype=float)
    beta_y = np.asarray(tw['beta_y'], dtype=float)
    eta = np.asarray(tw[eta_key], dtype=float)

    for arr, name in ((beta_x, 'beta_x'), (beta_y, 'beta_y'), (eta, eta_key)):
        if arr.ndim != 1 or arr.size != s.size:
            raise ValueError(f"'{name}' must be 1D with len == len(s).")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{name}' must be finite.")

    color_Bx = 'darkred'
    color_By = 'olive'
    color_Bz = 'steelblue'
    color_beta_x = 'teal'
    color_beta_y = 'peru'
    color_eta = 'slategray'

    fig, (axB, axTw) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(9, 7),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle("Magnetic field and Twiss parameters vs s", fontsize=16 * k, x=0.5)

    def style_axis(ax):
        ax.set_facecolor('white')
        ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
        ax.tick_params(direction='in', top=True, right=True)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')

    style_axis(axB)
    labels = []
    comps = [
        (B[:, 0], 'Bx', color_Bx),
        (B[:, 1], 'By', color_By),
        (B[:, 2], 'Bz', color_Bz),
    ]
    for y, lbl, col in comps:
        if np.any(y != 0):
            axB.plot(s, y, color=col, lw=1.5, label=lbl)
            axB.fill_between(s, y, 0.0, facecolor=col, alpha=0.5)
            labels.append(lbl)
    if labels:
        axB.legend(loc='upper right', frameon=True)
    axB.set_ylabel('B [T]')

    # 2) Twiss with shared x, twin y-axes: betas on left, eta on right
    style_axis(axTw)
    axTw.plot(s, beta_x, color=color_beta_x, lw=1.5, label=r'$\beta_x$')
    axTw.plot(s, beta_y, color=color_beta_y, lw=1.5, label=r'$\beta_y$')
    axTw.set_ylabel(r'$\beta$ [m]', color='black')
    axTw.legend(loc='upper left', frameon=True)

    axTw2 = axTw.twinx()
    axTw2.tick_params(direction='in', top=True, right=True)
    axTw2.plot(s, eta, color=color_eta, lw=1.5, label=r'$\eta_x$')
    axTw2.set_ylabel(r'$\eta_x$ [m]', color=color_eta)
    # optional: separate legend for right axis if you prefer
    # axTw2.legend(loc='upper right', frameon=True)

    axTw.set_xlabel('s [m]')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
#***********************************************************************************
# Wavefront
#***********************************************************************************

def plot_wavefront(wfr: dict, cuts: bool = True, show_phase: bool = True, **kwargs) -> None:
    """
    Plot wavefront intensity and optionally the phase from a wavefront dictionary.

    For each polarisation in the dictionary, this plots:
        - If cuts=True: 2D intensity map + horizontal (y=0) and vertical (x=0) cuts.
        - If cuts=False: Only the 2D intensity map.
    Phase map is shown at the end if requested, with optional unwrapping.

    Args:
        wfr (dict): Dictionary returned by `write_wavefront` or `read_wavefront`.
        cuts (bool, optional): Whether to include 1D cuts in the plot (default: True).
        show_phase (bool, optional): Whether to plot the phase maps (default: True).
        **kwargs:
            k (float, optional): Scaling factor for fonts and titles (default: 1).
            vmin (float, optional): Minimum value for intensity color scale.
            vmax (float, optional): Maximum value for intensity color scale.
            unwrap (bool, optional): Whether to unwrap the phase before plotting (default: True).
    """
    k = kwargs.get('k', 1)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    unwrap= kwargs.get('unwrap', True)
    # mask_phase = kwargs.get('mask_phase', True)

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
        fig.suptitle(f"({pol}) | flux: {flux:.2e} ph/s/0.1%bw", fontsize=16 * k, x=0.5)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046 * 1, pad=0.04, format='%.0e')
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

        if show_phase:
            phase = wfr['phase'][pol]
            if unwrap:
                phase = unwrap_phase(phase)
                phase -=phase[phase.shape[0]//2, phase.shape[1]//2]
                cmapref = 'terrain'
            else:
                cmapref = 'coolwarm'                
            # if mask_phase:
            #     data /= np.max(data)
            #     phase = np.ma.masked_where(data < 1E-4, phase)
            Rx = wfr['wfr'].Rx
            Ry = wfr['wfr'].Ry
            fig = plt.figure(figsize=(4.2*fctr, 4))
            fig.suptitle(f"({pol}) | residual phase - Rx = {Rx:.2f}m, Ry = {Ry:.2f}m", fontsize=16 * k, x=0.5)
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(X, Y, phase, shading='auto', cmap=cmapref)#, vmin=-np.pi, vmax=np.pi)
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.grid(True, linestyle=':', linewidth=0.5)
            cb = plt.colorbar(im, ax=ax, fraction=0.046 * 1, pad=0.04)
            plt.show()

            if cuts:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
                ix0 = np.argmin(np.abs(x))
                iy0 = np.argmin(np.abs(y))

                ax1.plot(x, phase[iy0, :], color='darkred', lw=1.5)
                ax1.set_title('Hor. cut (y=0)')
                ax1.set_xlabel('x [mm]')
                ax1.set_ylabel('rad')
                ax1.grid(True, linestyle=':', linewidth=0.5)
                ax1.tick_params(direction='in', top=True, right=True)
                ax2.plot(y, phase[:, ix0], color='darkred', lw=1.5)
                ax2.set_title('Ver. cut (x=0)')
                ax2.set_xlabel('y [mm]')
                ax2.grid(True, linestyle=':', linewidth=0.5)
                ax2.tick_params(direction='in', top=True, right=True)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show() 

#***********************************************************************************
# Power Density
#***********************************************************************************

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
        fig.suptitle(f"({pol}) | power: {integrated:.3e} W | peak: {peak:.2f} W/mm²",
                     fontsize=16 * k, x=0.5)

        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)#, format='%.1e')
        cb.set_label('power density [W/mm²]')
        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            ax1.plot(x, data[iy0, :], color='darkred', lw=1.5)
            ax1.set_title('Hor. cut (y=0)')
            ax1.set_xlabel('x [mm]')
            ax1.set_ylabel('power density [W/mm²]')
            ax1.grid(True, linestyle=':', linewidth=0.5)
            ax1.tick_params(direction='in', top=True, right=True)

            ax2.plot(y, data[:, ix0], color='darkred', lw=1.5)
            ax2.set_title('Ver. cut (x=0)')
            ax2.set_xlabel('y [mm]')
            ax2.grid(True, linestyle=':', linewidth=0.5)
            ax2.tick_params(direction='in', top=True, right=True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

#***********************************************************************************
# spectrum
#***********************************************************************************

def plot_spectrum(spectrum: dict, logy: bool = True, spectral_power: bool = False, **kwargs) -> None:
    """
    Plot flux vs energy for each polarisation in a spectrum dictionary.

    Optionally, also plots spectral power (linear x linear) and cumulated power
    sharing the same x-axis as flux, in a subplot layout.

    Args:
        spectrum (dict): Spectrum dictionary as returned by write_spectrum or read_spectrum.
        logy (bool, optional): If True, plot flux with logarithmic y-axis (default: True).
        spectral_power (bool, optional): If True, plot spectral power and cumulated power (default: False).
        **kwargs: Optional keyword arguments, e.g., scaling factor `k`.
    """
    k = kwargs.get('k', 1)
    start_plotting(k)

    energy = spectrum['energy']

    for pol in spectrum:
        if pol in ["energy", "window"]:
            continue

        flux = spectrum[pol]['flux']

        if spectral_power:
            fig, (ax, ax1) = plt.subplots(
                nrows=2, ncols=1,
                sharex=True,
                figsize=(8, 8),
                gridspec_kw={'height_ratios': [1, 1]}
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        fig.suptitle(f"({pol}) | spectral flux", fontsize=16 * k, x=0.5)
        ax.plot(energy, flux, color='darkred', lw=1.5)
        ax.set_ylabel('flux [ph/s/0.1%bw]')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.tick_params(direction='in', top=True, right=True)
        if logy:
            ax.set_yscale('log')

        if spectral_power:
            spower = spectrum[pol]['spectral_power']
            cum_power = spectrum[pol]['cumulated_power']
            color1 = 'olive'
            ax1.plot(energy, spower, color=color1, lw=1.5, label='Spectral power')
            ax1.set_ylabel('spectral power [W/eV]', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, linestyle=':', linewidth=0.5)
            ax1.tick_params(direction='in', top=True, right=True)
            ax2 = ax1.twinx()
            color2 = 'steelblue'
            ax2.plot(energy, cum_power, color=color2, lw=1.5, linestyle='-', label='Cumulated power')
            ax2.set_ylabel('cumulated power [W]', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.tick_params(direction='in', top=True, right=True)
            ax1.set_xlabel('energy [eV]')

        else:
            ax.set_xlabel('energy [eV]')

        plt.tight_layout()
        plt.show()

#***********************************************************************************
# CMD
#***********************************************************************************

def plot_csd(cmdDict: dict, cuts: bool = True, **kwargs) -> None:
    """
    Plot cross-spectral density (CSD) and occupation/cumulative plots from a coherent mode decomposition dictionary.

    For each direction ('H', 'V'), this plots:
        - The 2D CSD map with optional horizontal and vertical cuts.
        - The eigenmode occupation vs mode number, with cumulative sum on a secondary y-axis.

    Args:
        cmdDict (dict): CMD dictionary as returned by write_cmd or read_cmd.
        cuts (bool, optional): Whether to include 1D cuts in the CSD plots (default: True).
        **kwargs:
            k (float, optional): Scaling factor for fonts and titles (default: 1).
            nmode_max (int, optional): Maximum mode number to plot. If None, plot all modes.
    """
    k = kwargs.get('k', 1)
    nmode_max = kwargs.get('nmodes', None)
    start_plotting(k)

    for direction, data in cmdDict['source'].items():
        axis = data['axis'] * 1e3  # convert to mm
        CSD = data['CSD']
        occupation = data['occupation']
        cumulative = data['cumulated']
        CF = data['CF']

        X, Y = np.meshgrid(axis, axis)

        fctr = (axis[-1] - axis[0]) / (axis[-1] - axis[0])

        # CSD 2D plot
        fig = plt.figure(figsize=(4.2 * fctr, 4))
        fig.suptitle(f"({direction}) | cross spectral density", fontsize=16 * k, x=0.5)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, CSD, shading='auto', cmap='terrain')
        ax.set_aspect('equal')
        ax.set_xlabel(f'{direction.lower()}$_1$ [mm]')
        ax.set_ylabel(f'{direction.lower()}$_2$ [mm]')
        ax.grid(True, linestyle=':', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.0e')
        plt.show()

        if cuts:
            ix0 = np.argmin(np.abs(axis))
            iy0 = np.argmin(np.abs(axis))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

            ax1.plot(axis, CSD[iy0, :], color='darkred', lw=1.5)
            ax1.set_title('Hor. cut (y=0)')
            ax1.set_xlabel(f'{direction.lower()}$_1$ [mm]')
            ax1.set_ylabel('CSD')
            ax1.grid(True, linestyle=':', linewidth=0.5)
            ax1.tick_params(direction='in', top=True, right=True)
            
            ax2.plot(axis, CSD[:, ix0], color='darkred', lw=1.5)
            ax2.set_title('Ver. cut (x=0)')
            ax2.set_xlabel(f'{direction.lower()}$_2$ [mm]')
            ax2.grid(True, linestyle=':', linewidth=0.5)
            ax2.tick_params(direction='in', top=True, right=True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        # occupation vs cumulative plot
        mode_numbers = np.arange(1, len(data['eigenvalues']) + 1)

        if nmode_max is not None:
            mode_numbers = mode_numbers[:nmode_max]
            occupation = occupation[:nmode_max]
            cumulative = cumulative[:nmode_max]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        color1 = 'darkred'
        ax1.plot(mode_numbers, occupation, color=color1, marker='o', linestyle='--', 
                 markerfacecolor='white', markeredgecolor=color1, markeredgewidth=1.5, 
                 lw=1.5, label='Occupation')
        ax1.set_xlabel('mode number')
        ax1.set_ylabel('occupation', color=color1)
        ax1.set_ylim(bottom=-0.05, top=1.05)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':', linewidth=0.5)
        ax1.tick_params(direction='in', top=True, right=True)

        ax2 = ax1.twinx()
        color2 = 'steelblue'
        ax2.plot(mode_numbers, cumulative, color=color2, marker='s', linestyle='--', 
                 markerfacecolor='white', markeredgecolor=color2, markeredgewidth=1.5,
                 lw=1.5, label='Cumulative')
        ax2.set_ylabel('cumulative', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.tick_params(direction='in', top=True, right=True)

        fig.suptitle(f"({direction}) | modes | CF {CF*100:.1f}%", fontsize=16 * k, x=0.5)
        plt.tight_layout()
        plt.show()
#***********************************************************************************
# scans
#***********************************************************************************

def plot_multiple_spectra(scans: list, polarisation: str, logy: bool = True, **kwargs) -> None:
    """
    Plots flux vs energy for a list of spectrum dictionaries (different window sizes) on the same graph.

    Parameters:
        scans (list): List of spectrum dictionaries (each as returned by write_spectrum).
        polarisation (str): The polarisation to plot ('total', 'horizontal', 'vertical', etc.).
        logy (bool, optional): If True, uses logarithmic y-axis (default: True).
        **kwargs: Additional keyword arguments passed to start_plotting (e.g. scaling factor 'k').

    Notes:
        Only first, middle (or before middle if even), and last scans are labelled with window size.
    """
    k = kwargs.get('k', 1)
    observation_point = kwargs.get('observation_point', None)
    start_plotting(k)

    my_cmap = LinearSegmentedColormap.from_list('my_cmap', 
                                                [(0.00, ( 14/255,  14/255, 120/255, 1)),
                                                 (0.17, ( 62/255, 117/255, 207/255, 1)),
                                                 (0.30, ( 91/255, 190/255, 243/255, 1)),
                                                 (0.43, (100/255, 200/255, 150/255, 1)),
                                                 (0.59, (244/255, 213/255, 130/255, 1)),
                                                 (0.71, (237/255, 158/255,  80/255, 1)),
                                                 (0.85, (204/255,  90/255,  41/255, 1)),
                                                 (1.00, (150/255,  20/255,  30/255, 1))])
    colors = my_cmap(np.linspace(0, 1, len(scans)))
    # colors = cm.jet(np.linspace(0, 1, len(scans)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"({polarisation}) | spectral flux", fontsize=16*k)

    indices_to_label = [0, len(scans)-1]
    # indices_to_label = [0, (len(scans)-1)//2, len(scans)-1] if len(scans)%2 else [0, (len(scans)//2)-1, len(scans)-1]

    if len(scans)>=5:
        lw = 1
    else:
        lw = 1.5

    for idx, (scan, color) in enumerate(zip(scans, colors)):

        energy = scan['energy']
        flux = scan[polarisation]['flux']
        window_dx = scan['window']['dx']
        window_dy = scan['window']['dy']
        label = None
        if idx in indices_to_label:
            if observation_point is None:
                label = f"{window_dx*1e3:.2f} x {window_dy*1e3:.2f} mm²"
            else:
                window_dx_rmad = 2*np.arctan(window_dx/2/observation_point)
                window_dy_rmad = 2*np.arctan(window_dy/2/observation_point)
                label = f"{window_dx_rmad*1e3:.2f} x {window_dy_rmad*1e3:.2f} mrad²"

        ax.plot(energy, flux, color=color, lw=lw, label=label)

    ax.set_xlabel('energy [eV]')
    ax.set_ylabel('flux [ph/s/0.1%bw]')
    if logy:
        ax.set_yscale('log')

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.tick_params(direction='in', top=True, right=True)

    if indices_to_label:
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_beamline_acceptance_scan(results: dict, observation_point: float, measurement: str, **kwargs) -> None:
    """
    Plots integrated flux and/or power vs slit size with twin y-axes if both are present.

    The lower x-axis is slit size [mm], the secondary bottom x-axis is divergence [mrad].

    Parameters:
        results (dict): Dictionary returned by integrate_flux_and_power_over_slits.
        observation_point (float): Observation point distance [m].
        **kwargs:
            k (float, optional): Scaling factor for fonts and titles (default: 1).
    """
    k = kwargs.get('k', 1)
    start_plotting(k)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_mm = results['axis'] * 1e3

    has_flux = 'flux' in results
    has_power = 'power' in results

    if has_flux and not has_power:
        ax.plot(
            x_mm,
            results['flux'],
            marker='s',
            linestyle=':',
            linewidth=1.5,
            color='darkred',
            markerfacecolor='white',
            markeredgecolor='darkred',
            markeredgewidth=1.5,
        )
        ax.set_ylabel('integrated flux [ph/s/0.1%bw]')
        ax.tick_params(axis='y')

    elif has_power and not has_flux:
        ax.plot(
            x_mm,
            results['power'],
            marker='o',
            linestyle='--',
            linewidth=1.5,
            color='steelblue',
            markerfacecolor='white',
            markeredgecolor='steelblue',
            markeredgewidth=1.5,
        )
        ax.set_ylabel('integrated power [W]')
        ax.tick_params(axis='y')

    elif has_flux and has_power:
        ax.plot(
            x_mm,
            results['flux'],
            marker='s',
            linestyle=':',
            linewidth=1.5,
            color='darkred',
            markerfacecolor='white',
            markeredgecolor='darkred',
            markeredgewidth=1.5,
            label='flux'
        )
        ax.set_ylabel('integrated flux [ph/s/0.1%bw]', color='darkred')
        ax.tick_params(axis='y', labelcolor='darkred')

        ax2 = ax.twinx()
        ax2.plot(
            x_mm,
            results['power'],
            marker='o',
            linestyle='--',
            linewidth=1.5,
            color='steelblue',
            markerfacecolor='white',
            markeredgecolor='steelblue',
            markeredgewidth=1.5,
            label='power'
        )
        ax2.set_ylabel('integrated power [W]', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')

    else:
        raise ValueError("No 'flux' or 'power' entries in results dictionary.")

    ax.set_xlabel("slit acceptance [mm]")
    ax.set_title(f"({measurement}) | beamline acceptance scan\n")
    ax.grid(True, linestyle=':', color='gray', linewidth=0.5)
    ax.tick_params(direction='in', top=True, right=True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    secax_bottom = ax.secondary_xaxis(-0.2)
    secax_bottom.set_xlabel('[mrad]')
    mrad_ticks = np.arange(0, 2.1, 0.1)
    mm_ticks = 2 * observation_point * np.tan(mrad_ticks / 2)
    secax_bottom.set_xticks(mm_ticks)
    secax_bottom.set_xticklabels([f"{tick:.2f}" for tick in mrad_ticks])

    plt.tight_layout()
    plt.show()

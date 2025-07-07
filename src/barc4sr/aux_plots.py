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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import rcParamsDefault
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
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
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

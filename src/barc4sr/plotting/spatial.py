# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Plots for 2D spatial maps: wavefronts, power density, and CSD.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.restoration import unwrap_phase

from .style import igor_cmap, srw_cmap, start_plotting

# ---------------------------------------------------------------------------
# Wavefront
# ---------------------------------------------------------------------------

def plot_wavefront(
    wfr: dict,
    cuts: bool = True,
    show_phase: bool = True,
    log_intensity: bool = False,
    **kwargs,
) -> None:
    """
    Plot wavefront intensity and optionally the phase from a wavefront dictionary.

    For each polarisation in the dictionary, this plots:
        - If cuts=True: 2D intensity map + horizontal (y=0) and vertical (x=0) cuts.
        - If cuts=False: Only the 2D intensity map.
    Phase map is shown at the end if requested, with optional unwrapping.

    Intensity can be shown either in linear scale (default) or logarithmic scale
    using Matplotlib's log normalization for the 2D map and semilogy for the cuts.

    Parameters
    ----------
    wfr : dict
        Dictionary returned by `write_wavefront` or `read_wavefront`.
    cuts : bool, optional
        Whether to include 1D cuts in the plot (default: True).
    show_phase : bool, optional
        Whether to plot the phase maps (default: True).
    log_intensity : bool, optional
        If True, use log scale for intensity (2D via LogNorm, cuts via semilogy).
        Phase is always plotted in linear scale (default: False).
    **kwargs :
        k : float, optional
            Scaling factor for fonts and titles (default: 1).
        vmin : float, optional
            Minimum value for intensity color scale (linear units).
        vmax : float, optional
            Maximum value for intensity color scale (linear units).
        unwrap : bool, optional
            Whether to unwrap the phase before plotting (default: True).
        cmap : str, optional
            Colormap for intensity. Can be any Matplotlib cmap name (e.g. 'viridis'),
            or the special keywords:
                - 'srw'  : black → white
                - 'igor' : black → navy → darkred → red → orange → yellow
            Default is 'jet'.
    """
    k = kwargs.get("k", 1)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    unwrap = kwargs.get("unwrap", True)
    cmap_name = kwargs.get("cmap", "jet")

    if cmap_name == "srw":
        cmap_intensity = srw_cmap
    elif cmap_name == "igor":
        cmap_intensity = igor_cmap
    else:
        cmap_intensity = cmap_name 

    start_plotting(k)

    x = wfr["axis"]["x"] * 1e3  # mm
    y = wfr["axis"]["y"] * 1e3  # mm
    X, Y = np.meshgrid(x, y)

    dx = wfr["axis"]["x"][1] - wfr["axis"]["x"][0]
    dy = wfr["axis"]["y"][1] - wfr["axis"]["y"][0]

    fctr = (wfr["axis"]["x"][-1] - wfr["axis"]["x"][0]) / (
        wfr["axis"]["y"][-1] - wfr["axis"]["y"][0]
    )

    for pol, data in wfr["intensity"].items():
        flux = np.sum(data * dx * 1e3 * dy * 1e3)

        if log_intensity:
            data_masked = np.ma.masked_less_equal(data, 0.0)
            positive = data[data > 0]

            if positive.size == 0:
                continue

            vmin_eff = vmin if (vmin is not None and vmin > 0) else positive.min()
            vmax_eff = vmax if (vmax is not None and vmax > vmin_eff) else data.max()
            norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
            vmin_lin = None
            vmax_lin = None
        else:
            data_masked = data
            norm = None
            vmin_lin = vmin
            vmax_lin = vmax

        fig = plt.figure(figsize=(4.2 * fctr, 4))
        fig.suptitle(
            f"({pol}) | flux: {flux:.2e} ph/s/0.1%bw",
            fontsize=16 * k,
            x=0.5,
        )
        ax = fig.add_subplot(111)

        im = ax.pcolormesh(
            X,
            Y,
            data_masked,
            shading="auto",
            cmap=cmap_intensity,
            vmin=vmin_lin,
            vmax=vmax_lin,
            norm=norm,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(True, linestyle=":", linewidth=0.5)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            hor = data[iy0, :]
            ver = data[:, ix0]

            if log_intensity:
                hor_plot = np.where(hor > 0, hor, np.nan)
                ver_plot = np.where(ver > 0, ver, np.nan)

                ax1.semilogy(x, hor_plot, color="darkred", lw=1.5)
                ax2.semilogy(y, ver_plot, color="darkred", lw=1.5)
            else:
                ax1.plot(x, hor, color="darkred", lw=1.5)
                ax2.plot(y, ver, color="darkred", lw=1.5)

            ax1.set_title("Hor. cut (y=0)")
            ax1.set_xlabel("x [mm]")
            ax1.set_ylabel(r"ph/s/mm$^2$/0.1%bw")
            ax1.grid(True, linestyle=":", linewidth=0.5)
            ax1.tick_params(direction="in", top=True, right=True)

            ax2.set_title("Ver. cut (x=0)")
            ax2.set_xlabel("y [mm]")
            ax2.grid(True, linestyle=":", linewidth=0.5)
            ax2.tick_params(direction="in", top=True, right=True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        if show_phase:
            phase = wfr["phase"][pol]
            if unwrap:
                phase = unwrap_phase(phase)
                phase -= phase[phase.shape[0] // 2, phase.shape[1] // 2]
                cmapref = "terrain"
            else:
                cmapref = "coolwarm"

            Rx = wfr["wfr"].Rx
            Ry = wfr["wfr"].Ry

            fig = plt.figure(figsize=(4.2 * fctr, 4))
            fig.suptitle(
                f"({pol}) | residual phase - Rx = {Rx:.2f} m, Ry = {Ry:.2f} m",
                fontsize=16 * k,
                x=0.5,
            )
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(X, Y, phase, shading="auto", cmap=cmapref)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.grid(True, linestyle=":", linewidth=0.5)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.show()

            if cuts:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
                ix0 = np.argmin(np.abs(x))
                iy0 = np.argmin(np.abs(y))

                ax1.plot(x, phase[iy0, :], color="darkred", lw=1.5)
                ax1.set_title("Hor. cut (y=0)")
                ax1.set_xlabel("x [mm]")
                ax1.set_ylabel("rad")
                ax1.grid(True, linestyle=":", linewidth=0.5)
                ax1.tick_params(direction="in", top=True, right=True)

                ax2.plot(y, phase[:, ix0], color="darkred", lw=1.5)
                ax2.set_title("Ver. cut (x=0)")
                ax2.set_xlabel("y [mm]")
                ax2.grid(True, linestyle=":", linewidth=0.5)
                ax2.tick_params(direction="in", top=True, right=True)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

# ---------------------------------------------------------------------------
# Power density
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Cross-spectral density
# ---------------------------------------------------------------------------

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
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
from barc4sr.processing.wavefront import integrate_wavefront_window
from barc4sr.processing.power import integrate_power_density_window

# ---------------------------------------------------------------------------
# Wavefront
# ---------------------------------------------------------------------------

def plot_wavefront(
    wfr: dict,
    cuts: bool = True,
    show_phase: bool = True,
    log_intensity: bool = False,
    observation_plane: float = None,
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
    observation_plane : float, optional
        Distance to observation plane in meters. If provided, axes are shown in mrad
        and intensity is converted to ph/s/mrad²/0.1%bw (default: None).
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
        xmin : float, optional
            Minimum x-axis limit (in mm or mrad depending on observation_plane).
        xmax : float, optional
            Maximum x-axis limit (in mm or mrad depending on observation_plane).
        ymin : float, optional
            Minimum y-axis limit (in mm or mrad depending on observation_plane).
        ymax : float, optional
            Maximum y-axis limit (in mm or mrad depending on observation_plane).
    """
    k = kwargs.get("k", 1)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    unwrap = kwargs.get("unwrap", True)
    cmap_name = kwargs.get("cmap", "jet")
    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    if cmap_name == "srw":
        cmap_intensity = srw_cmap
    elif cmap_name == "igor":
        cmap_intensity = igor_cmap
    else:
        cmap_intensity = cmap_name 

    start_plotting(k)

    x_m = wfr["axis"]["x"]
    y_m = wfr["axis"]["y"]
    x_mm = x_m * 1e3
    y_mm = y_m * 1e3

    dx_mm = x_m[1] - x_m[0]
    dy_mm = y_m[1] - y_m[0]

    if observation_plane is not None:
        x = 2 * np.arctan(x_m / 2 / observation_plane) * 1e3
        y = 2 * np.arctan(y_m / 2 / observation_plane) * 1e3
        
        dx_mrad_mean = np.mean(np.diff(x))
        dy_mrad_mean = np.mean(np.diff(y))
        
        intensity_factor = (dx_mm * 1e3 * dy_mm * 1e3) / (dx_mrad_mean * dy_mrad_mean)
        
        unit_label = "mrad"
        intensity_unit = r"ph/s/mrad$^2$/0.1%bw"
        
        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_mrad = [xmin if xmin is not None else x.min(), 
                          xmax if xmax is not None else x.max()]
            y_lim_mrad = [ymin if ymin is not None else y.min(), 
                          ymax if ymax is not None else y.max()]
            

            x_lim_m = [2 * observation_plane * np.tan(xl / 2e3) for xl in x_lim_mrad]
            y_lim_m = [2 * observation_plane * np.tan(yl / 2e3) for yl in y_lim_mrad]
            
            hor_slit = tuple(x_lim_m)
            ver_slit = tuple(y_lim_m)
        else:
            hor_slit = None
            ver_slit = None
    else:
        x = x_mm
        y = y_mm
        intensity_factor = 1.0
        unit_label = "mm"
        intensity_unit = r"ph/s/mm$^2$/0.1%bw"
        
        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_mm = [xmin if xmin is not None else x.min(), 
                        xmax if xmax is not None else x.max()]
            y_lim_mm = [ymin if ymin is not None else y.min(), 
                        ymax if ymax is not None else y.max()]
            
            hor_slit = tuple([xl / 1e3 for xl in x_lim_mm])
            ver_slit = tuple([yl / 1e3 for yl in y_lim_mm])
        else:
            hor_slit = None
            ver_slit = None

    X, Y = np.meshgrid(x, y)

    if xmin is not None or xmax is not None:
        x_range_min = xmin if xmin is not None else x.min()
        x_range_max = xmax if xmax is not None else x.max()
    else:
        x_range_min = x.min()
        x_range_max = x.max()
    
    if ymin is not None or ymax is not None:
        y_range_min = ymin if ymin is not None else y.min()
        y_range_max = ymax if ymax is not None else y.max()
    else:
        y_range_min = y.min()
        y_range_max = y.max()
    
    fctr = (x_range_max - x_range_min) / (y_range_max - y_range_min)

    for pol, data in wfr["intensity"].items():
        if hor_slit is not None and ver_slit is not None:
            flux_dict = integrate_wavefront_window(wfr, hor_slit, ver_slit)
            flux = flux_dict[pol]
        else:
            flux = np.sum(data * dx_mm * 1e3 * dy_mm * 1e3)

        data_converted = data * intensity_factor

        if log_intensity:
            data_masked = np.ma.masked_less_equal(data_converted, 0.0)
            positive = data_converted[data_converted > 0]

            if positive.size == 0:
                continue

            vmin_eff = vmin if (vmin is not None and vmin > 0) else positive.min()
            vmax_eff = vmax if (vmax is not None and vmax > vmin_eff) else data_converted.max()
            norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
            vmin_lin = None
            vmax_lin = None
        else:
            data_masked = data_converted
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
        ax.set_xlabel(f"x [{unit_label}]")
        ax.set_ylabel(f"y [{unit_label}]")
        
        if xmin is not None:
            ax.set_xlim(left=xmin)
        if xmax is not None:
            ax.set_xlim(right=xmax)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        
        ax.grid(True, linestyle=":", linewidth=0.5)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            hor = data_converted[iy0, :]
            ver = data_converted[:, ix0]

            if log_intensity:
                hor_plot = np.where(hor > 0, hor, np.nan)
                ver_plot = np.where(ver > 0, ver, np.nan)

                ax1.semilogy(x, hor_plot, color="darkred", lw=1.5)
                ax2.semilogy(y, ver_plot, color="darkred", lw=1.5)
            else:
                ax1.plot(x, hor, color="darkred", lw=1.5)
                ax2.plot(y, ver, color="darkred", lw=1.5)

            ax1.set_title("Hor. cut (y=0)")
            ax1.set_xlabel(f"x [{unit_label}]")
            ax1.set_ylabel(intensity_unit)
            ax1.grid(True, linestyle=":", linewidth=0.5)
            ax1.tick_params(direction="in", top=True, right=True)
            
            if xmin is not None:
                ax1.set_xlim(left=xmin)
            if xmax is not None:
                ax1.set_xlim(right=xmax)

            ax2.set_title("Ver. cut (x=0)")
            ax2.set_xlabel(f"y [{unit_label}]")
            ax2.grid(True, linestyle=":", linewidth=0.5)
            ax2.tick_params(direction="in", top=True, right=True)
            
            if ymin is not None:
                ax2.set_xlim(left=ymin)
            if ymax is not None:
                ax2.set_xlim(right=ymax)

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
            ax.set_xlabel(f"x [{unit_label}]")
            ax.set_ylabel(f"y [{unit_label}]")
            
            if xmin is not None:
                ax.set_xlim(left=xmin)
            if xmax is not None:
                ax.set_xlim(right=xmax)
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            if ymax is not None:
                ax.set_ylim(top=ymax)
            
            ax.grid(True, linestyle=":", linewidth=0.5)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.show()

            if cuts:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
                ix0 = np.argmin(np.abs(x))
                iy0 = np.argmin(np.abs(y))

                ax1.plot(x, phase[iy0, :], color="darkred", lw=1.5)
                ax1.set_title("Hor. cut (y=0)")
                ax1.set_xlabel(f"x [{unit_label}]")
                ax1.set_ylabel("rad")
                ax1.grid(True, linestyle=":", linewidth=0.5)
                ax1.tick_params(direction="in", top=True, right=True)
                
                if xmin is not None:
                    ax1.set_xlim(left=xmin)
                if xmax is not None:
                    ax1.set_xlim(right=xmax)

                ax2.plot(y, phase[:, ix0], color="darkred", lw=1.5)
                ax2.set_title("Ver. cut (x=0)")
                ax2.set_xlabel(f"y [{unit_label}]")
                ax2.grid(True, linestyle=":", linewidth=0.5)
                ax2.tick_params(direction="in", top=True, right=True)
                
                if ymin is not None:
                    ax2.set_xlim(left=ymin)
                if ymax is not None:
                    ax2.set_xlim(right=ymax)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

# ---------------------------------------------------------------------------
# Power density
# ---------------------------------------------------------------------------

def plot_power_density(
    pwr: dict,
    cuts: bool = True,
    observation_distance: float | None = None,
    **kwargs,
) -> None:
    """
    Plot power density maps from a power dictionary.

    For each polarisation in the dictionary, this plots:
        - If cuts=True: 2D power map + horizontal (y=0) and vertical (x=0) cuts.
        - If cuts=False: Only the 2D power map.

    Power density can be shown either in the native spatial units (mm) or, if
    observation_distance is provided, in angular units (mrad) using a small-angle
    mapping from the observation plane.

    Parameters
    ----------
    pwr : dict
        Dictionary returned by `write_power_density` or `read_power_density`.
    cuts : bool, optional
        Whether to include 1D cuts in the plot (default: True).
    observation_distance : float, optional
        Distance to the observation plane in meters. If provided, axes are shown
        in mrad and power density is converted to W/mrad² (default: None).
    **kwargs :
        k : float, optional
            Scaling factor for fonts and titles (default: 1).
        vmin : float, optional
            Minimum value for color scale (linear units).
        vmax : float, optional
            Maximum value for color scale (linear units).
        cmap : str, optional
            Colormap for power. Can be any Matplotlib cmap name (e.g. 'plasma'),
            or the special keywords:
                - 'srw'  : black → white
                - 'igor' : black → navy → darkred → red → orange → yellow
            Default is 'plasma'.
        xmin : float, optional
            Minimum x-axis limit (in mm or mrad depending on observation_distance).
        xmax : float, optional
            Maximum x-axis limit (in mm or mrad depending on observation_distance).
        ymin : float, optional
            Minimum y-axis limit (in mm or mrad depending on observation_distance).
        ymax : float, optional
            Maximum y-axis limit (in mm or mrad depending on observation_distance).
    """
    k = kwargs.get("k", 1)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    cmap_name = kwargs.get("cmap", "plasma")
    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    if cmap_name == "srw":
        cmap_power = srw_cmap
    elif cmap_name == "igor":
        cmap_power = igor_cmap
    else:
        cmap_power = cmap_name

    start_plotting(k)

    x_m = pwr["axis"]["x"]
    y_m = pwr["axis"]["y"]
    x_mm = x_m * 1e3
    y_mm = y_m * 1e3

    dx_mm = x_m[1] - x_m[0] 
    dy_mm = y_m[1] - y_m[0]


    if observation_distance is not None:

        x = 2 * np.arctan(x_m / (2 * observation_distance)) * 1e3  # mrad
        y = 2 * np.arctan(y_m / (2 * observation_distance)) * 1e3  # mrad

        dx_mrad_mean = np.mean(np.diff(x))
        dy_mrad_mean = np.mean(np.diff(y))

        power_factor = (dx_mm * 1e3 * dy_mm * 1e3) / (dx_mrad_mean * dy_mrad_mean)

        unit_label = "mrad"
        power_unit = r"W/mrad$^2$"

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_mrad = [
                xmin if xmin is not None else x.min(),
                xmax if xmax is not None else x.max(),
            ]
            y_lim_mrad = [
                ymin if ymin is not None else y.min(),
                ymax if ymax is not None else y.max(),
            ]

            x_lim_m = [2 * observation_distance * np.tan(xl / 2e3) for xl in x_lim_mrad]
            y_lim_m = [2 * observation_distance * np.tan(yl / 2e3) for yl in y_lim_mrad]

            hor_slit = tuple(x_lim_m)
            ver_slit = tuple(y_lim_m)
        else:
            hor_slit = None
            ver_slit = None
    else:
        x = x_mm
        y = y_mm
        power_factor = 1.0
        unit_label = "mm"
        power_unit = r"W/mm$^2$"

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_mm = [
                xmin if xmin is not None else x.min(),
                xmax if xmax is not None else x.max(),
            ]
            y_lim_mm = [
                ymin if ymin is not None else y.min(),
                ymax if ymax is not None else y.max(),
            ]

            hor_slit = tuple([xl / 1e3 for xl in x_lim_mm])
            ver_slit = tuple([yl / 1e3 for yl in y_lim_mm])
        else:
            hor_slit = None
            ver_slit = None

    X, Y = np.meshgrid(x, y)

    if xmin is not None or xmax is not None:
        x_range_min = xmin if xmin is not None else x.min()
        x_range_max = xmax if xmax is not None else x.max()
    else:
        x_range_min = x.min()
        x_range_max = x.max()

    if ymin is not None or ymax is not None:
        y_range_min = ymin if ymin is not None else y.min()
        y_range_max = ymax if ymax is not None else y.max()
    else:
        y_range_min = y.min()
        y_range_max = y.max()

    fctr = (x_range_max - x_range_min) / (y_range_max - y_range_min)

    if hor_slit is not None and ver_slit is not None:
        window_power = integrate_power_density_window(pwr, hor_slit, ver_slit)
    else:
        window_power = None

    for pol, pdata in pwr.items():
        if pol == "axis":
            continue

        data = pdata["map"]

        if window_power is not None:
            integrated = window_power[pol]
        else:
            integrated = pdata.get(
                "integrated",
                np.sum(data) * dx_mm * 1e3 * dy_mm * 1e3,
            )

        data_converted = data * power_factor
        peak_converted = data_converted.max()

        fig = plt.figure(figsize=(4.2 * fctr, 4))
        fig.suptitle(
            f"({pol}) | power: {integrated:.3e} W | peak: {peak_converted:.2f} {power_unit}",
            fontsize=16 * k,
            x=0.5,
        )

        ax = fig.add_subplot(111)
        im = ax.pcolormesh(
            X,
            Y,
            data_converted,
            shading="auto",
            cmap=cmap_power,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{unit_label}]")
        ax.set_ylabel(f"y [{unit_label}]")

        if xmin is not None:
            ax.set_xlim(left=xmin)
        if xmax is not None:
            ax.set_xlim(right=xmax)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)

        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()

        if cuts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            ix0 = np.argmin(np.abs(x))
            iy0 = np.argmin(np.abs(y))

            hor = data_converted[iy0, :]
            ver = data_converted[:, ix0]

            ax1.plot(x, hor, color="darkred", lw=1.5)
            ax1.set_title("Hor. cut (y=0)")
            ax1.set_xlabel(f"x [{unit_label}]")
            ax1.set_ylabel(f"power density [{power_unit}]")
            ax1.grid(True, linestyle=":", linewidth=0.5)
            ax1.tick_params(direction="in", top=True, right=True)
            if xmin is not None:
                ax1.set_xlim(left=xmin)
            if xmax is not None:
                ax1.set_xlim(right=xmax)

            ax2.plot(y, ver, color="darkred", lw=1.5)
            ax2.set_title("Ver. cut (x=0)")
            ax2.set_xlabel(f"y [{unit_label}]")
            ax2.grid(True, linestyle=":", linewidth=0.5)
            ax2.tick_params(direction="in", top=True, right=True)
            if ymin is not None:
                ax2.set_xlim(left=ymin)
            if ymax is not None:
                ax2.set_xlim(right=ymax)

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
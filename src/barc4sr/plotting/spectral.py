# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Plots for spectra, tuning curves, and beamline acceptance scans.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import start_plotting, scan_cmap

# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

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

    colors = scan_cmap(np.linspace(0, 1, len(scans)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"({polarisation}) | spectral flux", fontsize=16*k)

    indices_to_label = [0, len(scans)-1]

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

# ---------------------------------------------------------------------------
# Acceptance scan
# ---------------------------------------------------------------------------

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
# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Plots for 2D spatial maps: wavefronts, power density, and CSD.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, hsv_to_rgb
from skimage.restoration import unwrap_phase

from barc4sr.processing.power import integrate_power_density_window
from barc4sr.processing.wavefront import integrate_wavefront_window

from .style import igor_cmap, srw_cmap, start_plotting

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
        Distance to observation plane in meters. If provided, axes are shown in
        angular units and intensity is converted to ph/s/mrad²/0.1%bw
        (default: None).
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
            Minimum x-axis limit in base units:
                - meters if observation_plane is None
                - radians if observation_plane is not None
        xmax : float, optional
            Maximum x-axis limit in base units.
        ymin : float, optional
            Minimum y-axis limit in base units.
        ymax : float, optional
            Maximum y-axis limit in base units.
        threshold : float | None, optional
            Relative intensity threshold used to mask low-signal regions in the phase.
    """
    k = kwargs.get("k", 1)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    unwrap = kwargs.get("unwrap", True)
    threshold = kwargs.get("threshold", None)
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

    dx_m = x_m[1] - x_m[0]
    dy_m = y_m[1] - y_m[0]

    if observation_plane is not None:
        x_rad = 2 * np.arctan(x_m / (2 * observation_plane))
        y_rad = 2 * np.arctan(y_m / (2 * observation_plane))

        x_range_min_rad = xmin if xmin is not None else x_rad.min()
        x_range_max_rad = xmax if xmax is not None else x_rad.max()
        y_range_min_rad = ymin if ymin is not None else y_rad.min()
        y_range_max_rad = ymax if ymax is not None else y_rad.max()

        range_x_rad = x_range_max_rad - x_range_min_rad
        range_y_rad = y_range_max_rad - y_range_min_rad
        use_micro = max(range_x_rad, range_y_rad) < 0.7e-3

        if use_micro:
            axis_factor = 1e6
            unit_label = "µrad"
        else:
            axis_factor = 1e3
            unit_label = "mrad"

        x = x_rad * axis_factor
        y = y_rad * axis_factor

        x_lim_display = [
            x_range_min_rad * axis_factor,
            x_range_max_rad * axis_factor,
        ]
        y_lim_display = [
            y_range_min_rad * axis_factor,
            y_range_max_rad * axis_factor,
        ]

        dx_mrad_mean = np.mean(np.diff(x_rad * 1e3))
        dy_mrad_mean = np.mean(np.diff(y_rad * 1e3))

        intensity_factor = ((dx_m * 1e3) * (dy_m * 1e3)) / (dx_mrad_mean * dy_mrad_mean)
        intensity_unit = r"ph/s/mrad$^2$/0.1%bw"

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_m = [2 * observation_plane * np.tan(val / 2) for val in (x_range_min_rad, x_range_max_rad)]
            y_lim_m = [2 * observation_plane * np.tan(val / 2) for val in (y_range_min_rad, y_range_max_rad)]
            hor_slit = tuple(x_lim_m)
            ver_slit = tuple(y_lim_m)
        else:
            hor_slit = None
            ver_slit = None

    else:
        x_range_min_m = xmin if xmin is not None else x_m.min()
        x_range_max_m = xmax if xmax is not None else x_m.max()
        y_range_min_m = ymin if ymin is not None else y_m.min()
        y_range_max_m = ymax if ymax is not None else y_m.max()

        range_x_m = x_range_max_m - x_range_min_m
        range_y_m = y_range_max_m - y_range_min_m
        use_micro = max(range_x_m, range_y_m) < 0.7e-3

        if use_micro:
            axis_factor = 1e6
            unit_label = "µm"
        else:
            axis_factor = 1e3
            unit_label = "mm"

        x = x_m * axis_factor
        y = y_m * axis_factor

        x_lim_display = [
            x_range_min_m * axis_factor,
            x_range_max_m * axis_factor,
        ]
        y_lim_display = [
            y_range_min_m * axis_factor,
            y_range_max_m * axis_factor,
        ]

        intensity_factor = 1.0
        intensity_unit = r"ph/s/mm$^2$/0.1%bw"

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            hor_slit = (x_range_min_m, x_range_max_m)
            ver_slit = (y_range_min_m, y_range_max_m)
        else:
            hor_slit = None
            ver_slit = None

    X, Y = np.meshgrid(x, y)

    x_range_min = x_lim_display[0]
    x_range_max = x_lim_display[1]
    y_range_min = y_lim_display[0]
    y_range_max = y_lim_display[1]

    fctr = (x_range_max - x_range_min) / (y_range_max - y_range_min)

    meta = wfr.get("meta", {})
    residual_phase = bool(meta.get("residual_phase", False))

    for pol, data in wfr["intensity"].items():
        if hor_slit is not None and ver_slit is not None:
            flux_dict = integrate_wavefront_window(wfr, hor_slit, ver_slit)
            flux = flux_dict[pol]
        else:
            flux = np.sum(data * (dx_m * 1e3) * (dy_m * 1e3))

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

        if threshold is not None:
            mask = data_masked >= threshold * data_masked.max()
            data_masked[~mask] = threshold * data_masked.max()

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
            ax.set_xlim(left=x_lim_display[0])
        if xmax is not None:
            ax.set_xlim(right=x_lim_display[1])
        if ymin is not None:
            ax.set_ylim(bottom=y_lim_display[0])
        if ymax is not None:
            ax.set_ylim(top=y_lim_display[1])

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
                ax1.set_xlim(left=x_lim_display[0])
            if xmax is not None:
                ax1.set_xlim(right=x_lim_display[1])

            ax2.set_title("Ver. cut (x=0)")
            ax2.set_xlabel(f"y [{unit_label}]")
            ax2.grid(True, linestyle=":", linewidth=0.5)
            ax2.tick_params(direction="in", top=True, right=True)

            if ymin is not None:
                ax2.set_xlim(left=y_lim_display[0])
            if ymax is not None:
                ax2.set_xlim(right=y_lim_display[1])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        if show_phase:

            vmin = None
            vmax = None

            phase = wfr["phase"][pol].copy()
            if unwrap:
                phase = unwrap_phase(phase)
                phase -= phase[phase.shape[0] // 2, phase.shape[1] // 2]
                cmapref = "terrain"
            else:
                cmapref = "coolwarm"

            if threshold is not None:
                phase[~mask] = np.nan

            if unwrap:
                phase -= np.nanpercentile(phase, 0.1)

            # vmin = np.nanpercentile(phase, 0.25)
            # vmax = np.nanpercentile(phase, 99.75)

            # if vmax > 0:
            #     vmax *= 1.0005
            # else:
            #     vmax *= 0.9995

            # if vmin > 0:
            #     vmin *= 0.9995
            # else:
            #     vmin *= 1.0005

            Rx = wfr.get("Rx", None)
            Ry = wfr.get("Ry", None)

            fig = plt.figure(figsize=(4.2 * fctr, 4))

            if residual_phase:
                fig.suptitle(
                    f"({pol}) | residual phase - Rx = {Rx:.2f} m, Ry = {Ry:.2f} m",
                    fontsize=16 * k,
                    x=0.5,
                )
            else:
                fig.suptitle(
                    f"({pol}) | phase - Rx = {Rx:.2f} m, Ry = {Ry:.2f} m",
                    fontsize=16 * k,
                    x=0.5,
                )

            ax = fig.add_subplot(111)
            im = ax.pcolormesh(X, Y, phase, shading="auto", cmap=cmapref, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlabel(f"x [{unit_label}]")
            ax.set_ylabel(f"y [{unit_label}]")

            if xmin is not None:
                ax.set_xlim(left=x_lim_display[0])
            if xmax is not None:
                ax.set_xlim(right=x_lim_display[1])
            if ymin is not None:
                ax.set_ylim(bottom=y_lim_display[0])
            if ymax is not None:
                ax.set_ylim(top=y_lim_display[1])

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
                    ax1.set_xlim(left=x_lim_display[0])
                if xmax is not None:
                    ax1.set_xlim(right=x_lim_display[1])

                ax1.set_ylim(vmin, vmax)

                ax2.plot(y, phase[:, ix0], color="darkred", lw=1.5)
                ax2.set_title("Ver. cut (x=0)")
                ax2.set_xlabel(f"y [{unit_label}]")
                ax2.grid(True, linestyle=":", linewidth=0.5)
                ax2.tick_params(direction="in", top=True, right=True)

                if ymin is not None:
                    ax2.set_xlim(left=y_lim_display[0])
                if ymax is not None:
                    ax2.set_xlim(right=y_lim_display[1])

                ax2.set_ylim(vmin, vmax)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


def plot_complex_wavefront(
    wfr: dict,
    observation_plane: float = None,
    **kwargs,
) -> None:
    """
    Plot a complex wavefront using a phase-hue / intensity-brightness encoding.

    For each polarisation in the dictionary, this plots a single 2D complex-field
    image where phase is encoded as hue and intensity modulates the displayed
    brightness, with two colorbars: one for phase [-pi, pi] and one for
    normalized intensity [0, 1].

    The 2D complex rendering uses a HSV-style mapping:
        - hue        <- wrapped phase in [-pi, pi]
        - value      <- normalized intensity
        - saturation <- 1 in valid pixels, 0 in masked pixels

    Parameters
    ----------
    wfr : dict
        Dictionary returned by `write_wavefront` or `read_wavefront`.
    observation_plane : float, optional
        Distance to observation plane in meters. If provided, axes are shown in
        angular units and intensity is converted to ph/s/mrad²/0.1%bw
        (default: None).
    **kwargs :
        k : float, optional
            Scaling factor for fonts and titles (default: 1).
        xmin : float, optional
            Minimum x-axis limit in the displayed axis unit.
        xmax : float, optional
            Maximum x-axis limit in the displayed axis unit.
        ymin : float, optional
            Minimum y-axis limit in the displayed axis unit.
        ymax : float, optional
            Maximum y-axis limit in the displayed axis unit.
        threshold : float | None, optional
            Relative intensity threshold used to mask low-signal regions in the
            complex rendering.
    """
    k = kwargs.get("k", 1)
    threshold = kwargs.get("threshold", None)
    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    start_plotting(k)

    x_m = wfr["axis"]["x"]
    y_m = wfr["axis"]["y"]

    x_mm = x_m * 1e3
    y_mm = y_m * 1e3

    dx_mm = x_m[1] - x_m[0]
    dy_mm = y_m[1] - y_m[0]

    if observation_plane is not None:
        x_rad = 2 * np.arctan(x_m / 2 / observation_plane)
        y_rad = 2 * np.arctan(y_m / 2 / observation_plane)

        if xmin is not None or xmax is not None:
            x_range_min_rad = xmin if xmin is not None else x_rad.min()
            x_range_max_rad = xmax if xmax is not None else x_rad.max()
        else:
            x_range_min_rad = x_rad.min()
            x_range_max_rad = x_rad.max()

        if ymin is not None or ymax is not None:
            y_range_min_rad = ymin if ymin is not None else y_rad.min()
            y_range_max_rad = ymax if ymax is not None else y_rad.max()
        else:
            y_range_min_rad = y_rad.min()
            y_range_max_rad = y_rad.max()

        range_x_rad = x_range_max_rad - x_range_min_rad
        range_y_rad = y_range_max_rad - y_range_min_rad
        use_micro = max(range_x_rad, range_y_rad) < 0.7e-3

        if use_micro:
            axis_factor = 1e6
            unit_label = "µrad"
        else:
            axis_factor = 1e3
            unit_label = "mrad"

        x = x_rad * axis_factor
        y = y_rad * axis_factor

        dx_mrad_mean = np.mean(np.diff(x_rad * 1e3))
        dy_mrad_mean = np.mean(np.diff(y_rad * 1e3))

        intensity_factor = (dx_mm * 1e3 * dy_mm * 1e3) / (dx_mrad_mean * dy_mrad_mean)

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_display = [
                xmin if xmin is not None else x.min(),
                xmax if xmax is not None else x.max(),
            ]
            y_lim_display = [
                ymin if ymin is not None else y.min(),
                ymax if ymax is not None else y.max(),
            ]

            x_lim_rad = [xl / axis_factor for xl in x_lim_display]
            y_lim_rad = [yl / axis_factor for yl in y_lim_display]

            x_lim_m = [2 * observation_plane * np.tan(xl / 2) for xl in x_lim_rad]
            y_lim_m = [2 * observation_plane * np.tan(yl / 2) for yl in y_lim_rad]

            hor_slit = tuple(x_lim_m)
            ver_slit = tuple(y_lim_m)
        else:
            hor_slit = None
            ver_slit = None

    else:
        if xmin is not None or xmax is not None:
            x_range_min_m = xmin * 1e-3 if xmin is not None else x_m.min()
            x_range_max_m = xmax * 1e-3 if xmax is not None else x_m.max()
        else:
            x_range_min_m = x_m.min()
            x_range_max_m = x_m.max()

        if ymin is not None or ymax is not None:
            y_range_min_m = ymin * 1e-3 if ymin is not None else y_m.min()
            y_range_max_m = ymax * 1e-3 if ymax is not None else y_m.max()
        else:
            y_range_min_m = y_m.min()
            y_range_max_m = y_m.max()

        range_x_m = x_range_max_m - x_range_min_m
        range_y_m = y_range_max_m - y_range_min_m
        use_micro = max(range_x_m, range_y_m) < 0.7e-3

        if use_micro:
            axis_factor = 1e6
            unit_label = "µm"
        else:
            axis_factor = 1e3
            unit_label = "mm"

        x = x_m * axis_factor
        y = y_m * axis_factor

        intensity_factor = 1.0

        if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            x_lim_display = [
                xmin if xmin is not None else x.min(),
                xmax if xmax is not None else x.max(),
            ]
            y_lim_display = [
                ymin if ymin is not None else y.min(),
                ymax if ymax is not None else y.max(),
            ]

            hor_slit = tuple([xl / axis_factor for xl in x_lim_display])
            ver_slit = tuple([yl / axis_factor for yl in y_lim_display])
        else:
            hor_slit = None
            ver_slit = None

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

    _n = 256
    _hue = np.linspace(0.0, 1.0, _n, endpoint=False)
    _hsv_strip = np.ones((_n, 1, 3))
    _hsv_strip[:, 0, 0] = _hue
    _rgb_strip = hsv_to_rgb(_hsv_strip)[:, 0, :]
    phase_cmap = mcolors.LinearSegmentedColormap.from_list(
        "phase_cyclic", _rgb_strip, N=_n
    )

    intens_cmap = mcolors.LinearSegmentedColormap.from_list(
        "intensity_bw", ["black", "white"], N=256
    )

    for pol, intensity in wfr["intensity"].items():
        if hor_slit is not None and ver_slit is not None:
            flux_dict = integrate_wavefront_window(wfr, hor_slit, ver_slit)
            flux = flux_dict[pol]
        else:
            flux = np.sum(intensity * dx_mm * 1e3 * dy_mm * 1e3)

        intensity_converted = intensity * intensity_factor

        if threshold is not None:
            mask = intensity_converted >= threshold * intensity_converted.max()
        else:
            mask = np.ones_like(intensity_converted, dtype=bool)

        phase = wfr["phase"][pol].copy()
        phase_wrapped = np.angle(np.exp(1j * phase))
        hue = (phase_wrapped + np.pi) / (2 * np.pi)

        value = intensity_converted.astype(float)
        peak = np.amax(value)
        value = value / peak if peak > 0 else np.zeros_like(value)

        saturation = np.ones_like(value)
        saturation[~mask] = 0.0
        value[~mask] = 0.0

        hsv = np.dstack((hue, saturation, value))
        rgb = hsv_to_rgb(hsv)

        fig = plt.figure(figsize=(4.2 * fctr + 1.8, 4))
        fig.suptitle(
            f"({pol}) | flux: {flux:.2e} ph/s/0.1%bw",
            fontsize=16 * k,
            x=0.5,
        )

        ax = fig.add_axes([0.10, 0.12, 0.68, 0.76])

        ax.imshow(
            rgb,
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="equal",
        )

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

        ax_cb_phase = fig.add_axes([0.80, 0.12, 0.030, 0.76])
        sm_phase = plt.cm.ScalarMappable(
            cmap=phase_cmap,
            norm=mcolors.Normalize(vmin=-np.pi, vmax=np.pi),
        )
        sm_phase.set_array([])
        cb_phase = plt.colorbar(sm_phase, cax=ax_cb_phase)
        cb_phase.set_ticks([-np.pi, 0, np.pi])
        cb_phase.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

        ax_cb_int = fig.add_axes([0.9, 0.12, 0.030, 0.76])
        sm_int = plt.cm.ScalarMappable(
            cmap=intens_cmap,
            norm=mcolors.Normalize(vmin=0, vmax=1),
        )
        sm_int.set_array([])
        cb_int = plt.colorbar(sm_int, cax=ax_cb_int)
        cb_int.set_ticks([0, 1])
        cb_int.set_ticklabels(["0", "1"])

        plt.show()


def plot_caustic(
    caustic: dict,
    direction: str = "both",
    observation_plane: float | None = None,
    **kwargs,
) -> None:
    """
    Plot horizontal and/or vertical beam caustics from a caustic dictionary.

    Parameters
    ----------
    caustic : dict
        Dictionary returned by `write_caustic` or `read_caustic`, with keys:
            - 'axis': {'x', 'y', 'z'}
            - 'intensity': {'horizontal', 'vertical'} per polarisation
            - 'meta': {'kind', 'threshold', 'energy'}
    direction : str, optional
        Direction to plot:
            - 'horizontal', 'h', 'x'
            - 'vertical', 'v', 'y'
            - 'both', 'b'
        Default is 'both'.
    observation_plane : float | None, optional
        Reserved for API consistency with `plot_wavefront`. Not used here.
    **kwargs
        k : float, optional
            Scaling factor for fonts and titles (default: 1).
        cmap : str, optional
            Colormap for intensity. Can be any Matplotlib cmap name, or:
                - 'srw'  : black → white
                - 'igor' : black → navy → darkred → red → orange → yellow
            Default is 'jet'.
        vmin : float | None, optional
            Minimum color scale value (default: None).
        vmax : float | None, optional
            Maximum color scale value (default: None).
        log_intensity : bool, optional
            If True, use logarithmic color normalization (default: False).
        zmin : float | None, optional
            Minimum z-axis limit in meters.
        zmax : float | None, optional
            Maximum z-axis limit in meters.
        xmin : float | None, optional
            Minimum x-axis limit in meters for horizontal caustic.
        xmax : float | None, optional
            Maximum x-axis limit in meters for horizontal caustic.
        ymin : float | None, optional
            Minimum y-axis limit in meters for vertical caustic.
        ymax : float | None, optional
            Maximum y-axis limit in meters for vertical caustic.
        show_colorbar : bool, optional
            Whether to show the colorbar (default: True).
    """
    k = kwargs.get("k", 1)
    cmap_name = kwargs.get("cmap", "jet")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    log_intensity = kwargs.get("log_intensity", False)

    zmin = kwargs.get("zmin", None)
    zmax = kwargs.get("zmax", None)
    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    show_colorbar = kwargs.get("show_colorbar", True)

    if observation_plane is not None:
        print(">>>>> observation_plane is ignored in plot_caustic.")

    if cmap_name == "srw":
        cmap_intensity = srw_cmap
    elif cmap_name == "igor":
        cmap_intensity = igor_cmap
    else:
        cmap_intensity = cmap_name

    start_plotting(k)

    x_m = np.asarray(caustic["axis"]["x"], dtype=float)
    y_m = np.asarray(caustic["axis"]["y"], dtype=float)
    z_m = np.asarray(caustic["axis"]["z"], dtype=float)

    if x_m.size < 2 or y_m.size < 2 or z_m.size < 2:
        raise ValueError("Caustic axes must contain at least two points each.")

    x_range_m = x_m.max() - x_m.min()
    y_range_m = y_m.max() - y_m.min()
    z_range_m = z_m.max() - z_m.min()

    use_micro_xy = max(x_range_m, y_range_m) < 0.7e-3
    if use_micro_xy:
        xy_factor = 1e6
        xy_unit = "µm"
    else:
        xy_factor = 1e3
        xy_unit = "mm"

    use_micro_z = z_range_m < 0.7e-3
    if use_micro_z:
        z_factor = 1e6
        z_unit = "µm"
    else:
        z_factor = 1e3
        z_unit = "mm"

    x = x_m * xy_factor
    y = y_m * xy_factor
    z = z_m * z_factor

    energy = caustic.get("meta", {}).get("energy", None)

    d = direction.lower()
    if d in ["x", "h", "hor", "horizontal"]:
        directions = ["horizontal"]
    elif d in ["y", "v", "ver", "vertical"]:
        directions = ["vertical"]
    elif d in ["b", "both"]:
        directions = ["horizontal", "vertical"]
    else:
        raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'.")

    def _get_norm(data: np.ndarray):
        if log_intensity:
            positive = data[np.isfinite(data) & (data > 0)]
            if positive.size == 0:
                return None, None, None
            vmin_eff = vmin if (vmin is not None and vmin > 0) else positive.min()
            vmax_eff = vmax if (vmax is not None and vmax > vmin_eff) else positive.max()
            return LogNorm(vmin=vmin_eff, vmax=vmax_eff), None, None
        return None, vmin, vmax

    def _plot_one(pol: str, which: str) -> None:
        if which == "horizontal":
            data = np.asarray(caustic["intensity"]["horizontal"][pol], dtype=float)
            transverse = x
            transverse_label = f"x [{xy_unit}]"
            title = f"({pol}) | horizontal caustic"
            tmin = xmin * xy_factor if xmin is not None else None
            tmax = xmax * xy_factor if xmax is not None else None
        else:
            data = np.asarray(caustic["intensity"]["vertical"][pol], dtype=float)
            transverse = y
            transverse_label = f"y [{xy_unit}]"
            title = f"({pol}) | vertical caustic"
            tmin = ymin * xy_factor if ymin is not None else None
            tmax = ymax * xy_factor if ymax is not None else None

        Z, T = np.meshgrid(z, transverse, indexing="xy")

        norm, vmin_eff, vmax_eff = _get_norm(data)
        if log_intensity and norm is None:
            return

        fig, ax = plt.subplots(figsize=(10, 4.5))
        if energy is None:
            fig.suptitle(title, fontsize=16 * k)
        else:
            fig.suptitle(f"{title} - E = {energy:.2f} eV", fontsize=16 * k)

        im = ax.pcolormesh(
            Z,
            T,
            data.T,
            shading="auto",
            cmap=cmap_intensity,
            norm=norm,
            vmin=vmin_eff,
            vmax=vmax_eff,
        )

        ax.set_xlabel(f"z [{z_unit}]")
        ax.set_ylabel(transverse_label)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_aspect("auto")

        if zmin is not None:
            ax.set_xlim(left=zmin * z_factor)
        if zmax is not None:
            ax.set_xlim(right=zmax * z_factor)
        if tmin is not None:
            ax.set_ylim(bottom=tmin)
        if tmax is not None:
            ax.set_ylim(top=tmax)

        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def _plot_both(pol: str) -> None:
        data_h = np.asarray(caustic["intensity"]["horizontal"][pol], dtype=float)
        data_v = np.asarray(caustic["intensity"]["vertical"][pol], dtype=float)

        norm_h, vmin_h, vmax_h = _get_norm(data_h)
        norm_v, vmin_v, vmax_v = _get_norm(data_v)

        if log_intensity and (norm_h is None or norm_v is None):
            return

        Zh, Xh = np.meshgrid(z, x, indexing="xy")
        Zv, Yv = np.meshgrid(z, y, indexing="xy")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        if energy is None:
            fig.suptitle(f"({pol}) | beam caustic", fontsize=16 * k)
        else:
            fig.suptitle(f"({pol}) | beam caustic - E = {energy:.2f} eV", fontsize=16 * k)

        im0 = axes[0].pcolormesh(
            Zh,
            Xh,
            data_h.T,
            shading="auto",
            cmap=cmap_intensity,
            norm=norm_h,
            vmin=vmin_h,
            vmax=vmax_h,
        )
        axes[0].set_ylabel(f"x [{xy_unit}]")
        axes[0].set_title("Horizontal")
        axes[0].grid(True, linestyle=":", linewidth=0.5)
        axes[0].tick_params(direction="in", top=True, right=True)
        axes[0].set_aspect("auto")

        im1 = axes[1].pcolormesh(
            Zv,
            Yv,
            data_v.T,
            shading="auto",
            cmap=cmap_intensity,
            norm=norm_v,
            vmin=vmin_v,
            vmax=vmax_v,
        )
        axes[1].set_xlabel(f"z [{z_unit}]")
        axes[1].set_ylabel(f"y [{xy_unit}]")
        axes[1].set_title("Vertical")
        axes[1].grid(True, linestyle=":", linewidth=0.5)
        axes[1].tick_params(direction="in", top=True, right=True)
        axes[1].set_aspect("auto")

        if zmin is not None:
            axes[0].set_xlim(left=zmin * z_factor)
        if zmax is not None:
            axes[0].set_xlim(right=zmax * z_factor)

        if xmin is not None:
            axes[0].set_ylim(bottom=xmin * xy_factor)
        if xmax is not None:
            axes[0].set_ylim(top=xmax * xy_factor)

        if ymin is not None:
            axes[1].set_ylim(bottom=ymin * xy_factor)
        if ymax is not None:
            axes[1].set_ylim(top=ymax * xy_factor)

        if show_colorbar:
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    pols_h = set(caustic["intensity"]["horizontal"].keys())
    pols_v = set(caustic["intensity"]["vertical"].keys())
    if pols_h != pols_v:
        raise ValueError("Horizontal and vertical caustics do not share the same polarisations.")

    for pol in caustic["intensity"]["horizontal"].keys():
        if len(directions) == 2:
            _plot_both(pol)
        else:
            _plot_one(pol, directions[0])

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


# def plot_complex_wavefront_white(
#     wfr: dict,
#     observation_plane: float = None,
#     **kwargs,
# ) -> None:
#     """
#     Plot a complex wavefront using a phase-hue / alpha-intensity encoding
#     on a white background.

#     For each polarisation in the dictionary, this plots a single 2D complex-field
#     image where phase is encoded as hue and intensity modulates the alpha
#     (transparency), with a white background and a single colorbar for phase
#     [-pi, pi].

#     The 2D complex rendering uses an RGBA mapping:
#         - hue        <- wrapped phase in [-pi, pi]
#         - saturation <- 1 in valid pixels, 0 in masked pixels
#         - value      <- 1 (full brightness throughout)
#         - alpha      <- normalized intensity [0, 1]

#     Parameters
#     ----------
#     wfr : dict
#         Dictionary returned by `write_wavefront` or `read_wavefront`.
#     observation_plane : float, optional
#         Distance to observation plane in meters. If provided, axes are shown in
#         angular units and intensity is converted to ph/s/mrad²/0.1%bw
#         (default: None).
#     **kwargs :
#         k : float, optional
#             Scaling factor for fonts and titles (default: 1).
#         xmin : float, optional
#             Minimum x-axis limit in the displayed axis unit.
#         xmax : float, optional
#             Maximum x-axis limit in the displayed axis unit.
#         ymin : float, optional
#             Minimum y-axis limit in the displayed axis unit.
#         ymax : float, optional
#             Maximum y-axis limit in the displayed axis unit.
#         threshold : float | None, optional
#             Relative intensity threshold used to mask low-signal regions.
#     """
#     k = kwargs.get("k", 1)
#     threshold = kwargs.get("threshold", None)
#     xmin = kwargs.get("xmin", None)
#     xmax = kwargs.get("xmax", None)
#     ymin = kwargs.get("ymin", None)
#     ymax = kwargs.get("ymax", None)

#     start_plotting(k)

#     x_m = wfr["axis"]["x"]
#     y_m = wfr["axis"]["y"]

#     x_mm = x_m * 1e3
#     y_mm = y_m * 1e3

#     dx_mm = x_m[1] - x_m[0]
#     dy_mm = y_m[1] - y_m[0]

#     if observation_plane is not None:
#         x_rad = 2 * np.arctan(x_m / 2 / observation_plane)
#         y_rad = 2 * np.arctan(y_m / 2 / observation_plane)

#         if xmin is not None or xmax is not None:
#             x_range_min_rad = xmin if xmin is not None else x_rad.min()
#             x_range_max_rad = xmax if xmax is not None else x_rad.max()
#         else:
#             x_range_min_rad = x_rad.min()
#             x_range_max_rad = x_rad.max()

#         if ymin is not None or ymax is not None:
#             y_range_min_rad = ymin if ymin is not None else y_rad.min()
#             y_range_max_rad = ymax if ymax is not None else y_rad.max()
#         else:
#             y_range_min_rad = y_rad.min()
#             y_range_max_rad = y_rad.max()

#         range_x_rad = x_range_max_rad - x_range_min_rad
#         range_y_rad = y_range_max_rad - y_range_min_rad
#         use_micro = max(range_x_rad, range_y_rad) < 0.7e-3

#         if use_micro:
#             axis_factor = 1e6
#             unit_label = "µrad"
#         else:
#             axis_factor = 1e3
#             unit_label = "mrad"

#         x = x_rad * axis_factor
#         y = y_rad * axis_factor

#         dx_mrad_mean = np.mean(np.diff(x_rad * 1e3))
#         dy_mrad_mean = np.mean(np.diff(y_rad * 1e3))

#         intensity_factor = (dx_mm * 1e3 * dy_mm * 1e3) / (dx_mrad_mean * dy_mrad_mean)

#         if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
#             x_lim_display = [
#                 xmin if xmin is not None else x.min(),
#                 xmax if xmax is not None else x.max(),
#             ]
#             y_lim_display = [
#                 ymin if ymin is not None else y.min(),
#                 ymax if ymax is not None else y.max(),
#             ]
#             x_lim_rad = [xl / axis_factor for xl in x_lim_display]
#             y_lim_rad = [yl / axis_factor for yl in y_lim_display]
#             x_lim_m = [2 * observation_plane * np.tan(xl / 2) for xl in x_lim_rad]
#             y_lim_m = [2 * observation_plane * np.tan(yl / 2) for yl in y_lim_rad]
#             hor_slit = tuple(x_lim_m)
#             ver_slit = tuple(y_lim_m)
#         else:
#             hor_slit = None
#             ver_slit = None

#     else:
#         if xmin is not None or xmax is not None:
#             x_range_min_m = xmin * 1e-3 if xmin is not None else x_m.min()
#             x_range_max_m = xmax * 1e-3 if xmax is not None else x_m.max()
#         else:
#             x_range_min_m = x_m.min()
#             x_range_max_m = x_m.max()

#         if ymin is not None or ymax is not None:
#             y_range_min_m = ymin * 1e-3 if ymin is not None else y_m.min()
#             y_range_max_m = ymax * 1e-3 if ymax is not None else y_m.max()
#         else:
#             y_range_min_m = y_m.min()
#             y_range_max_m = y_m.max()

#         range_x_m = x_range_max_m - x_range_min_m
#         range_y_m = y_range_max_m - y_range_min_m
#         use_micro = max(range_x_m, range_y_m) < 0.7e-3

#         if use_micro:
#             axis_factor = 1e6
#             unit_label = "µm"
#         else:
#             axis_factor = 1e3
#             unit_label = "mm"

#         x = x_m * axis_factor
#         y = y_m * axis_factor

#         intensity_factor = 1.0

#         if xmin is not None or xmax is not None or ymin is not None or ymax is not None:
#             x_lim_display = [
#                 xmin if xmin is not None else x.min(),
#                 xmax if xmax is not None else x.max(),
#             ]
#             y_lim_display = [
#                 ymin if ymin is not None else y.min(),
#                 ymax if ymax is not None else y.max(),
#             ]
#             hor_slit = tuple([xl / axis_factor for xl in x_lim_display])
#             ver_slit = tuple([yl / axis_factor for yl in y_lim_display])
#         else:
#             hor_slit = None
#             ver_slit = None

#     if xmin is not None or xmax is not None:
#         x_range_min = xmin if xmin is not None else x.min()
#         x_range_max = xmax if xmax is not None else x.max()
#     else:
#         x_range_min = x.min()
#         x_range_max = x.max()

#     if ymin is not None or ymax is not None:
#         y_range_min = ymin if ymin is not None else y.min()
#         y_range_max = ymax if ymax is not None else y.max()
#     else:
#         y_range_min = y.min()
#         y_range_max = y.max()

#     fctr = (x_range_max - x_range_min) / (y_range_max - y_range_min)

#     _n = 256
#     _hue = np.linspace(0.0, 1.0, _n, endpoint=False)
#     _hsv_strip = np.ones((_n, 1, 3))
#     _hsv_strip[:, 0, 0] = _hue
#     _rgb_strip = hsv_to_rgb(_hsv_strip)[:, 0, :]
#     phase_cmap = mcolors.LinearSegmentedColormap.from_list(
#         "phase_cyclic", _rgb_strip, N=_n
#     )

#     for pol, intensity in wfr["intensity"].items():
#         if hor_slit is not None and ver_slit is not None:
#             flux_dict = integrate_wavefront_window(wfr, hor_slit, ver_slit)
#             flux = flux_dict[pol]
#         else:
#             flux = np.sum(intensity * dx_mm * 1e3 * dy_mm * 1e3)

#         intensity_converted = intensity * intensity_factor

#         if threshold is not None:
#             mask = intensity_converted >= threshold * intensity_converted.max()
#         else:
#             mask = np.ones_like(intensity_converted, dtype=bool)

#         phase = wfr["phase"][pol].copy()
#         phase_wrapped = np.angle(np.exp(1j * phase))
#         hue = (phase_wrapped + np.pi) / (2 * np.pi)

#         saturation = np.ones_like(hue)
#         saturation[~mask] = 0.0

#         value = np.ones_like(hue)

#         hsv = np.dstack((hue, saturation, value))
#         rgb = hsv_to_rgb(hsv)

#         alpha = intensity_converted.astype(float)
#         peak = np.amax(alpha)
#         alpha = alpha / peak if peak > 0 else np.zeros_like(alpha)
#         alpha[~mask] = 0.0

#         rgba = np.dstack((rgb, alpha))

#         fig = plt.figure(figsize=(4.2 * fctr + 1.8, 4))
#         fig.suptitle(
#             f"({pol}) | flux: {flux:.2e} ph/s/0.1%bw",
#             fontsize=16 * k,
#             x=0.5,
#         )

#         ax = fig.add_axes([0.10, 0.12, 0.68, 0.76])
#         ax.set_facecolor("white")

#         ax.imshow(
#             rgba,
#             origin="lower",
#             extent=[x.min(), x.max(), y.min(), y.max()],
#             aspect="equal",
#         )

#         ax.set_xlabel(f"x [{unit_label}]")
#         ax.set_ylabel(f"y [{unit_label}]")

#         if xmin is not None:
#             ax.set_xlim(left=xmin)
#         if xmax is not None:
#             ax.set_xlim(right=xmax)
#         if ymin is not None:
#             ax.set_ylim(bottom=ymin)
#         if ymax is not None:
#             ax.set_ylim(top=ymax)

#         ax.grid(True, linestyle=":", linewidth=0.5)

#         ax_cb_phase = fig.add_axes([0.80, 0.12, 0.030, 0.76])
#         sm_phase = plt.cm.ScalarMappable(
#             cmap=phase_cmap,
#             norm=mcolors.Normalize(vmin=-np.pi, vmax=np.pi),
#         )
#         sm_phase.set_array([])
#         cb_phase = plt.colorbar(sm_phase, cax=ax_cb_phase)
#         cb_phase.set_ticks([-np.pi, 0, np.pi])
#         cb_phase.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

#         plt.show()
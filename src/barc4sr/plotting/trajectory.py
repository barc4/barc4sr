# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Plotting of particle trajectories and magnetic fields.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import cm

from .style import start_plotting

# ---------------------------------------------------------------------------
# Electron trajectory
# ---------------------------------------------------------------------------

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
        eBeamTraj['eTraj'][axis] * 1e3,
        eBeamTraj['eTraj'][f'{axis}p'] * 1e3
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

def plot_chief_rays(
    result: dict,
    direction: str = "horizontal",
    *,
    k: float = 1.0,
    ray_length: float = 1.0,
) -> None:
    """
    Plot chief rays on top of the electron trajectory with a magnetic-field panel.

    Layout (matches plot_electron_trajectory):
    - Panel 1: Magnetic field (Bx or By depending on direction) + chief-ray points
    - Panel 2: Trajectory (x or y) with chief rays overlaid
    - Panel 3: Trajectory angle (x' or y') with chief-ray points

    """
    start_plotting(k)

    if "chief_rays" not in result:
        raise KeyError("Input dict must contain key 'chief_rays' as returned by trace_chief_rays().")
    if "magnetic_field" not in result:
        raise KeyError("Input dict must contain key 'magnetic_field'.")
    if "trajectory" not in result:
        raise KeyError("Input dict must contain key 'trajectory' (full trajectory).")

    df = result["chief_rays"]
    mf = result["magnetic_field"]
    traj = result["trajectory"]

    required_cols = {"X", "Y", "Z", "dX", "dY", "id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"'chief_rays' DataFrame is missing required columns: {missing}.")

    d = direction.lower()
    if d in ("x", "h", "hor", "horizontal"):
        axis = "X"
        slope_col_full = "Xp"
        axis_label = "x"
        B_label = "By [T]"
        title = "Horizontal chief rays on electron trajectory"
        field_component = "By"
        comp_index = 1
    elif d in ("y", "v", "ver", "vertical"):
        axis = "Y"
        slope_col_full = "Yp"
        axis_label = "y"
        B_label = "Bx [T]"
        title = "Vertical chief rays on electron trajectory"
        field_component = "Bx"
        comp_index = 0
    else:
        raise ValueError("Direction must be 'horizontal'/'x' or 'vertical'/'y'.")

    Z_full = np.asarray(traj["Z"], dtype=float)
    R_full = np.asarray(traj[axis], dtype=float)
    Rp_full = np.asarray(traj[slope_col_full], dtype=float)

    if not (Z_full.ndim == R_full.ndim == Rp_full.ndim == 1):
        raise ValueError("Trajectory arrays Z, X/Y, Xp/Yp must be 1D.")
    if not (Z_full.size == R_full.size == Rp_full.size):
        raise ValueError("Z, X/Y, Xp/Yp must have the same length.")

    Z_full_mm = Z_full * 1e3
    R_full_mm = R_full * 1e3
    Rp_full_mrad = Rp_full * 1e3

    df_sorted = df.sort_values("Z", kind="mergesort").reset_index(drop=True)
    Zr = df_sorted["Z"].to_numpy(dtype=float)
    Rr = df_sorted[axis].to_numpy(dtype=float)
    dRr = df_sorted[f"d{axis}"].to_numpy(dtype=float)
    ids = df_sorted["id"].astype(str).to_numpy()

    Zr_mm = Zr * 1e3
    Rr_mm = Rr * 1e3

    s = np.asarray(mf["s"], dtype=float)
    B = np.asarray(mf["B"], dtype=float)

    if s.ndim != 1:
        raise ValueError("'magnetic_field['s']' must be 1D.")

    if B.ndim == 1:
        if B.size != s.size:
            raise ValueError("If 'B' is 1D, it must have the same length as 's'.")
        B_comp = B
    elif B.ndim == 2:
        if B.shape[0] != s.size or B.shape[1] != 3:
            raise ValueError("'B' must have shape (N, 3) with N == len(s).")
        B_comp = B[:, comp_index]
    else:
        raise ValueError("'B' must be 1D or 2D with shape (N, 3).")

    s_mm = s * 1e3

    if s.size > 1:
        B_at_Zr = np.interp(Zr, s, B_comp)
    else:
        B_at_Zr = np.full_like(Zr, B_comp[0] if B_comp.size else 0.0)

    if Z_full.size > 1:
        Rp_at_Zr_mrad = np.interp(Zr, Z_full, Rp_full_mrad)
    else:
        Rp_at_Zr_mrad = np.full_like(Zr, Rp_full_mrad[0] if Rp_full_mrad.size else 0.0)

    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(8, 8),
        height_ratios=[1, 1, 1],
    )
    fig.suptitle(title, fontsize=16 * k)
    fig.subplots_adjust(hspace=0.3)

    colors = ["darkred", "olive", "steelblue"]
    labels = [B_label, f"{axis_label} [mm]", f"{axis_label}' [mrad]"]

    unique_ids = list(dict.fromkeys(ids))
    n_seg = max(len(unique_ids), 1)
    seg_colours = [cm.tab10(i / max(n_seg - 1, 1)) for i in range(n_seg)]
    id_to_color = {seg_id: seg_colours[i] for i, seg_id in enumerate(unique_ids)}

    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(s_mm, B_comp, color=colors[0], linewidth=1.5, label=labels[0])

    for seg_id in unique_ids:
        mask = (ids == seg_id)
        if not np.any(mask):
            continue
        color = id_to_color[seg_id]
        ax.plot(
            Zr_mm[mask],
            B_at_Zr[mask],
            linestyle="none",
            marker="o",
            markersize=4,
            color=color,
            alpha=0.9,
        )

    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    ax.yaxis.label.set_color("black")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(Z_full_mm, R_full_mm, color=colors[1], linewidth=1.5, label=labels[1])

    for seg_id in unique_ids:
        mask = (ids == seg_id)
        Z0 = Zr[mask]
        R0 = Rr[mask]
        m = dRr[mask]

        if Z0.size == 0:
            continue

        color = id_to_color[seg_id]

        Z_ray_start = Z0
        Z_ray_end = Z0 + ray_length
        R_ray_start = R0
        R_ray_end = R0 + m * (Z_ray_end - Z_ray_start)

        for zs, ze, rs, re in zip(Z_ray_start, Z_ray_end, R_ray_start, R_ray_end):
            ax.plot(
                [zs * 1e3, ze * 1e3],
                [rs * 1e3, re * 1e3],
                color=color,
                linewidth=0.9,
                alpha=0.8,
            )

    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    ax.yaxis.label.set_color("black")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[2]
    ax.set_facecolor("white")
    ax.plot(Z_full_mm, Rp_full_mrad, color=colors[2], linewidth=1.5, label=labels[2])

    for seg_id in unique_ids:
        mask = (ids == seg_id)
        if not np.any(mask):
            continue
        color = id_to_color[seg_id]
        ax.plot(
            Zr_mm[mask],
            Rp_at_Zr_mrad[mask],
            linestyle="none",
            marker="o",
            markersize=3,
            color=color,
            alpha=0.9,
        )

    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    ax.yaxis.label.set_color("black")
    ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("[mm]", color="black")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Magnetic field
# ---------------------------------------------------------------------------

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
    elif direction.lower() in ['y', 'v', 'ver', 'vertical']:
        B = eBeamTraj['eTraj']['Bx']
        graph_title = 'Horizontal magnetic field'
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
    labels = ['B [T]', r'$\int$B$\cdot$d$s$ [T$\cdot$m]', r'$\iint$B$\cdot$d$s$² [T$\cdot$m²]']

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
    Plot magnetic field components vs s, and (optionally) Twiss parameters.

    Parameters
    ----------
    data : dict
        Required:
          - 's' : array, positions [m], shape (N,)
          - 'B' : array, magnetic field vectors [T], shape (N, 3) for (Bx, By, Bz)
        Optional:
          - 'Twiss' : dict with:
              * 'beta_x' : array, [m], shape (N,)
              * 'beta_y' : array, [m], shape (N,)
              * 'eta_x' or 'eta' : array, [m], shape (N,)
    **kwargs
        k : float, optional
            Global scaling factor for fonts (default matches aux_plots).

    Raises
    ------
    KeyError
        If required keys are missing in `data`.
    ValueError
        If array shapes are inconsistent or values are non-finite.
    """
    k = kwargs.get('k', 1)
    try:
        start_plotting(k)
    except NameError:
        pass

    if 's' not in data or 'B' not in data:
        raise KeyError("Missing required keys: 's' and 'B' must be present.")

    s = np.asarray(data['s'], dtype=float)
    B = np.asarray(data['B'], dtype=float)

    if s.ndim != 1:
        raise ValueError("'s' must be 1D.")
    if B.ndim != 2 or B.shape[1] != 3 or B.shape[0] != s.size:
        raise ValueError("'B' must have shape (N, 3) with N == len(s).")
    if not (np.isfinite(s).all() and np.isfinite(B).all()):
        raise ValueError("'s' and 'B' must be finite.")

    color_Bx = 'darkred'
    color_By = 'olive'
    color_Bz = 'steelblue'
    color_beta_x = 'teal'
    color_beta_y = 'peru'
    color_eta    = 'slategray'

    def style_axis(ax):
        ax.set_facecolor('white')
        ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
        ax.tick_params(direction='in', top=True, right=True)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')

    tw = data.get('Twiss', None)
    has_twiss = tw is not None

    if has_twiss:
        for key in ('beta_x', 'beta_y'):
            if key not in tw:
                raise KeyError(f"Missing Twiss key: '{key}'.")
        eta_key = 'eta_x' if 'eta_x' in tw else ('eta' if 'eta' in tw else None)
        if eta_key is None:
            raise KeyError("Missing Twiss key: 'eta_x' (or 'eta').")

        beta_x = np.asarray(tw['beta_x'], dtype=float)
        beta_y = np.asarray(tw['beta_y'], dtype=float)
        eta    = np.asarray(tw[eta_key], dtype=float)
        for arr, name in ((beta_x, 'beta_x'), (beta_y, 'beta_y'), (eta, eta_key)):
            if arr.ndim != 1 or arr.size != s.size:
                raise ValueError(f"'{name}' must be 1D with len == len(s).")
            if not np.isfinite(arr).all():
                raise ValueError(f"'{name}' must be finite.")

        fig, (axB, axTw) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(9, 7),
            gridspec_kw={'height_ratios': [1, 1]}
        )
        fig.subplots_adjust(hspace=0.25)
        fig.suptitle("Magnetic field and Twiss parameters vs s",
                     fontsize=16 * k, x=0.5)

        style_axis(axB)
        labels = []
        for y, lbl, col in ((B[:, 0], 'Bx', color_Bx),
                            (B[:, 1], 'By', color_By),
                            (B[:, 2], 'Bz', color_Bz)):
            if np.any(y != 0):
                axB.plot(s, y, color=col, lw=1.5, label=lbl)
                axB.fill_between(s, y, 0.0, facecolor=col, alpha=0.5)
                labels.append(lbl)
        if labels:
            axB.legend(loc='upper right', frameon=True)
        axB.set_ylabel('B [T]')

        style_axis(axTw)
        axTw.plot(s, beta_x, color=color_beta_x, lw=1.5, label=r'$\beta_x$')
        axTw.plot(s, beta_y, color=color_beta_y, lw=1.5, label=r'$\beta_y$')
        axTw.set_ylabel(r'$\beta$ [m]', color='black')
        axTw.legend(loc='upper left', frameon=True)

        axTw2 = axTw.twinx()
        axTw2.tick_params(direction='in', top=True, right=True)
        axTw2.plot(s, eta, color=color_eta, lw=1.5, label=r'$\eta_x$')
        axTw2.set_ylabel(r'$\eta_x$ [m]', color=color_eta)

        axTw.set_xlabel('s [m]')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    else:
        fig, axB = plt.subplots(figsize=(9, 4.5))
        fig.suptitle("Magnetic field vs s", fontsize=16 * k, x=0.5)
        style_axis(axB)

        axB.plot(s, B[:, 0], color=color_Bx, lw=1.5, label='Bx')
        axB.fill_between(s, B[:, 0], 0.0, facecolor=color_Bx, alpha=0.5)
        axB.plot(s, B[:, 1], color=color_By, lw=1.5, label='By')
        axB.fill_between(s, B[:, 1], 0.0, facecolor=color_By, alpha=0.5)

        axB.set_ylabel('B [T]')
        axB.set_xlabel('s [m]')
        axB.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()
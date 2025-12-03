# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Ray tracing calculations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import barc4beams

from .trajectory import electron_trajectory


def trace_chief_rays(
    *,
    eTraj: Optional[Dict[str, Any]] = None,
    n_rays: Optional[int] = None,
    B_rel_threshold: float = 0.01,
    wavelength_A: float = 1.0,
    id_prefix: str = "seg",
    **traj_kwargs: Any,
) -> Dict[str, Any]:
    """
    Build a chief-ray beam from an electron trajectory.

    Emission points are defined by a |B| relative threshold and down-sampled
    *per magnetic segment*:

      - If n_rays is None, all emission samples in each segment are used.
      - If len(segment) <= n_rays, all samples in that segment are used.
      - Otherwise, the first and last indices of the segment are always kept,
        and n_rays-2 inner indices are chosen uniformly inside the segment.

    Parameters
    ----------
    eTraj : dict, optional
        Electron trajectory dictionary with key "eTraj" containing
        arrays "Z", "X", "Y", "Xp", "Yp" and optionally "Bx", "By", "Bz".
        If provided, `traj_kwargs` must be empty.
    n_rays : int or None, optional
        Number of rays *per magnetic segment*. See behaviour above.
    B_rel_threshold : float, default 0.01
        Relative threshold in [0, 1] applied to |B| to define emitting
        regions:
            mask = |B| >= B_rel_threshold * max(|B|)
        If <= 0 or max(|B|) <= 0, the full trajectory is treated as one
        emitting segment.
    wavelength_A : float, default 1.0
        Wavelength assigned to all rays [Å].
    id_prefix : str, default "seg"
        Prefix used to tag rays belonging to the same magnetic segment
        in the "id" column of the returned beam.
    **traj_kwargs :
        Keyword arguments forwarded to `electron_trajectory` if `eTraj`
        is not supplied.

    Returns
    -------
    dict
        {
          "magnetic_field": {"s": Z, "B": |B|(Z)},
          "chief_rays": beam_df,         # standard barc4beams beam
          "trajectory": {
              "Z": Z, "X": X, "Y": Y, "Xp": Xp, "Yp": Yp,
          },
        }
    """
    has_traj_dict = eTraj is not None
    has_traj_kwargs = bool(traj_kwargs)

    if has_traj_dict and has_traj_kwargs:
        raise ValueError("Make up your mind, user!!! "
                         "Provide either `eTraj` or trajectory kwargs, not both.")
    if not has_traj_dict and not has_traj_kwargs:
        raise ValueError("Provide either `eTraj` or trajectory kwargs for "
                         "`electron_trajectory()`.")

    if n_rays is not None:
        n_rays = int(n_rays)
        if n_rays <= 0:
            raise ValueError("n_rays must be a positive integer or None.")

    if eTraj is None:
        traj_dict = electron_trajectory(**traj_kwargs)
    else:
        traj_dict = eTraj

    if "eTraj" not in traj_dict:
        raise KeyError("Trajectory dictionary must contain key 'eTraj'.")

    data = traj_dict["eTraj"]

    Z  = np.asarray(data["Z"],  dtype=float)   # [m]
    X  = np.asarray(data["X"],  dtype=float)
    Y  = np.asarray(data["Y"],  dtype=float)
    Xp = np.asarray(data["Xp"], dtype=float)   # [rad]
    Yp = np.asarray(data["Yp"], dtype=float)   # [rad]

    if not (Z.ndim == X.ndim == Y.ndim == Xp.ndim == Yp.ndim == 1):
        raise ValueError("Trajectory arrays Z, X, Y, Xp, Yp must be 1D.")
    if not (Z.size == X.size == Y.size == Xp.size == Yp.size):
        raise ValueError("Z, X, Y, Xp, Yp must have the same length.")

    n_points = Z.size

    # optional B-field components
    Bx = np.asarray(data.get("Bx", np.zeros(n_points)), dtype=float)
    By = np.asarray(data.get("By", np.zeros(n_points)), dtype=float)
    Bz = np.asarray(data.get("Bz", np.zeros(n_points)), dtype=float)
    if Bx.size != n_points or By.size != n_points or Bz.size != n_points:
        raise ValueError("Bx, By, Bz (if present) must have the same length as Z.")

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_abs = np.abs(Bmag)
    B_max = float(np.nanmax(B_abs)) if B_abs.size > 0 else 0.0

    if B_max > 0.0 and B_rel_threshold > 0.0:
        thr = float(B_rel_threshold) * B_max
        mask_emit = B_abs >= thr
    else:
        mask_emit = np.ones_like(B_abs, dtype=bool)

    emit_indices = np.where(mask_emit)[0]
    if emit_indices.size == 0:
        emit_indices = np.arange(n_points, dtype=int)
        mask_emit = np.ones_like(mask_emit, dtype=bool)

    segments: list[np.ndarray] = []
    start = int(emit_indices[0])
    prev = int(emit_indices[0])
    for idx in emit_indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            segments.append(np.arange(start, prev + 1, dtype=int))
            start = idx
            prev = idx
    segments.append(np.arange(start, prev + 1, dtype=int))

    index_to_seg: Dict[int, int] = {}
    for seg_id, seg_inds in enumerate(segments):
        for idx in seg_inds:
            index_to_seg[int(idx)] = seg_id

    chosen_indices_list: list[int] = []
    for seg_inds in segments:
        seg_inds = np.asarray(seg_inds, dtype=int)
        seg_len = seg_inds.size

        if n_rays is None or seg_len <= n_rays:
            chosen = seg_inds
        else:
            if n_rays == 1:
                # only one ray: take segment centre
                centre = seg_inds[seg_len // 2]
                chosen = np.array([centre], dtype=int)
            elif n_rays == 2:
                # first and last only
                chosen = np.array([seg_inds[0], seg_inds[-1]], dtype=int)
            else:
                n_inner = n_rays - 2
                inner = seg_inds[1:-1]
                # uniformly spaced indices in the interior
                inner_idx = np.linspace(0, inner.size - 1, n_inner, dtype=int)
                chosen = np.concatenate(
                    ([seg_inds[0]], inner[inner_idx], [seg_inds[-1]])
                )

        chosen_indices_list.append(chosen)

    chosen_indices = np.concatenate(chosen_indices_list)
    chosen_indices = np.unique(chosen_indices)  # in case segments touch

    n_chief = chosen_indices.size
    if n_chief == 0:
        raise RuntimeError("No chief rays selected after thresholding and sampling.")

    X0 = X[chosen_indices]
    Y0 = Y[chosen_indices]
    Z0 = Z[chosen_indices]
    dX = Xp[chosen_indices]
    dY = Yp[chosen_indices]
    dZ = np.sqrt(np.clip(1.0 - dX**2 - dY**2, 0.0, None))

    wavelength_m = float(wavelength_A) * 1e-10
    energy_eV = 12_398.4193 / float(wavelength_A)

    E = np.full(n_chief, energy_eV)
    W = np.full(n_chief, wavelength_m)
    I = np.ones(n_chief)
    Is = np.ones(n_chief)
    Ip = np.ones(n_chief)
    lost = np.zeros(n_chief, dtype=np.uint8)

    ids = []
    for idx in chosen_indices:
        seg_id = index_to_seg.get(int(idx), 0)
        ids.append(f"{id_prefix}{seg_id}")
    ids = np.asarray(ids, dtype=object)

    beam_df = pd.DataFrame(
        {
            "energy": E,
            "X": X0,
            "Y": Y0,
            "Z": Z0,
            "dX": dX,
            "dY": dY,
            "dZ": dZ,
            "wavelength": W,
            "intensity": I,
            "intensity_s-pol": Is,
            "intensity_p-pol": Ip,
            "lost_ray_flag": lost,
            "id": ids,
        }
    )
    barc4beams.schema.validate_beam(beam_df)

    magnetic_field = {
        "s": Z.copy(),
        "B": np.column_stack([Bx, By, Bz]).astype(float),
    }

    trajectory = {
        "Z": Z.copy(),
        "X": X.copy(),
        "Y": Y.copy(),
        "Xp": Xp.copy(),
        "Yp": Yp.copy(),
    }

    return {
        "magnetic_field": magnetic_field,
        "chief_rays": beam_df,
        "trajectory": trajectory,
    }
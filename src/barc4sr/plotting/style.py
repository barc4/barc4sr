# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Common plotting helpers, styles, and colormaps.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, rcParamsDefault
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# General style
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

srw_cmap = LinearSegmentedColormap.from_list(
    "srw_bw",
    [(0.0, "black"), (1.0, "white")],
)

igor_colors = [
    (0.0, (000/255,  22/255,  65/255, 1)),
    (0.2, (000/255, 145/255, 232/255, 1)),
    (0.4, (128/255,  73/255, 116/255, 1)),
    (0.6, (255/255, 000/255, 000/255, 1)),
    (0.8, (255/255, 124/255,   2/255, 1)),
    (1.0, (255/255, 240/255,  48/255, 1)),
]

igor_cmap = LinearSegmentedColormap.from_list("igor", igor_colors)

scan_colors = [
    (0.00, ( 14/255,  14/255, 120/255, 1)),
    (0.17, ( 62/255, 117/255, 207/255, 1)),
    (0.30, ( 91/255, 190/255, 243/255, 1)),
    (0.43, (100/255, 200/255, 150/255, 1)),
    (0.59, (244/255, 213/255, 130/255, 1)),
    (0.71, (237/255, 158/255,  80/255, 1)),
    (0.85, (204/255,  90/255,  41/255, 1)),
    (1.00, (150/255,  20/255,  30/255, 1))
]

scan_cmap = LinearSegmentedColormap.from_list("scan", scan_colors)
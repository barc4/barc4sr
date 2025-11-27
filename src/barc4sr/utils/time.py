# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
time.py - Small helpers for measuring and printing elapsed wall-clock time.
"""

from time import time


def print_elapsed_time(start: float) -> None:
    """
    Print the elapsed time since the start of a computation.

    Parameters
    ----------
    start : float
        Start time in seconds, as returned by time().
    """
    delta_t = time() - start

    if delta_t < 1.0:
        print(f">> Total elapsed time: {delta_t * 1000.0:.2f} ms")
        return

    hours, rem = divmod(delta_t, 3600.0)
    minutes, seconds = divmod(rem, 60.0)

    if hours >= 1.0:
        print(
            f">> Total elapsed time: "
            f"{int(hours)} h {int(minutes)} min {seconds:.2f} s"
        )
    elif minutes >= 1.0:
        print(f">> Total elapsed time: {int(minutes)} min {seconds:.2f} s")
    else:
        print(f">> Total elapsed time: {seconds:.2f} s")
        
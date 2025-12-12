# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Electron trajectory.
"""

from __future__ import annotations

import time

from .shared import initialize_calculation
from barc4sr.io.rw import write_electron_trajectory

from barc4sr.utils.time import print_elapsed_time


def electron_trajectory(**kwargs) -> dict:
    """
    Calculates electron trajectory using SRW.

    Optional Args (kwargs):
        file_name (str): The name of the output file.
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        verbose (bool): If True, print log information
    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()
    
    bl, eBeam, magFldCnt, eTraj, source_type, verbose = initialize_calculation(
        kwargs,
        observation_point=100,
        hor_slit=1e-3,
        ver_slit=1e-3,
        hor_slit_cen=0,
        ver_slit_cen=0,
        calc_etraj=True
    )
    
    file_name = kwargs.get('file_name', None)
    
    function_txt = f"{source_type} electron trajectory using SRW:"
    if verbose: print(f"{function_txt} please wait...")
    
    if verbose: print('completed')
    if verbose: print_elapsed_time(t0)
    
    eTrajDict = write_electron_trajectory(file_name, eTraj, bl["ElectronEnergy"])
    
    return eTrajDict
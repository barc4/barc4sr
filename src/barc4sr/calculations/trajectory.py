# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Electron trajectory.
"""

from __future__ import annotations

from time import time

from barc4sr.backend.srw.interface import set_light_source
from barc4sr.io.rw import write_electron_trajectory
from barc4sr.syned import barc4sr_dictionary, syned_dictionary
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
    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)
    file_name = kwargs.get('file_name', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    if json_file is not None:
        bl = syned_dictionary(json_file, 100, 1e-3, 1e-3, 0, 0)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, 100, 1e-3, 1e-3, 0, 0)

    calc_etraj = True

    if bl['Class'] == 'bm':
        source_type = 'bending magnet'
    elif bl['Class'] == 'u':
        source_type = 'undulator'
    elif bl['Class'] == 'w':
        source_type = 'wiggler'
    elif bl['Class'] == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} electron trajectory using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    ebeam_initial_condition = kwargs.get('ebeam_initial_condition', 6*[0])

    eBeam, magFldCnt, eTraj = set_light_source(bl,
                                               calc_etraj,
                                               ebeam_initial_condition,
                                               verbose)

    if verbose: print('completed')
    if verbose: print_elapsed_time(t0)

    eTrajDict = write_electron_trajectory(file_name, eTraj)

    return eTrajDict
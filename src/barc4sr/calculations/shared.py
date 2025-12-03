# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Shared utilities.
"""

from __future__ import annotations

from barc4sr.backend.srw.interface import set_light_source
from barc4sr.syned.mapping import barc4sr_dictionary, syned_dictionary


def initialize_calculation(kwargs, observation_point=None, 
                           hor_slit=None, ver_slit=None,
                           hor_slit_cen=0, ver_slit_cen=0,
                           calc_etraj=False):
    """
    Common initialization for radiation/trajectory calculations.
    
    Args:
        kwargs: Keyword arguments containing json_file or light_source
        observation_point: Distance to observation point
        hor_slit: Horizontal slit size [m]
        ver_slit: Vertical slit size [m]
        hor_slit_cen: Horizontal slit center [m]
        ver_slit_cen: Vertical slit center [m]
        calc_etraj: Whether to calculate electron trajectory
    
    Returns:
        tuple: (bl, eBeam, magFldCnt, eTraj, source_type, verbose)
    
    Raises:
        ValueError: If neither or both json_file and light_source provided
    """
    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)
    
    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")
    
    if json_file is not None:
        bl = syned_dictionary(json_file, observation_point, 
                            hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    else:
        bl = barc4sr_dictionary(light_source, observation_point,
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    
    source_types = {
        'bm': 'bending magnet',
        'u': 'undulator',
        'w': 'wiggler',
        'arb': 'arbitrary magnetic field'
    }
    source_type = source_types.get(bl['Class'], 'unknown')
    
    ebeam_initial_condition = kwargs.get('ebeam_initial_condition', 6*[0])
    eBeam, magFldCnt, eTraj = set_light_source(bl, calc_etraj, 
                                               ebeam_initial_condition,
                                               verbose=verbose)
    
    return bl, eBeam, magFldCnt, eTraj, source_type, verbose
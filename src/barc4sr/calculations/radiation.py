# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Radiation calculations.
"""

from __future__ import annotations

import time

import numpy as np

from barc4sr.backend.srw.interface import (
    set_light_source,
    srwlibCalcElecFieldSR,
    srwlibCalcPowDenSR,
)
from barc4sr.io.rw import write_power_density, write_spectrum, write_wavefront
from barc4sr.syned.mapping import barc4sr_dictionary, syned_dictionary
from barc4sr.utils.time import print_elapsed_time

from .shared import initialize_calculation


def wavefront(photon_energy: float,
              observation_point: float,
              hor_slit: float, 
              hor_slit_n: int,
              ver_slit: float,
              ver_slit_n: int,
              **kwargs) -> dict:
    """
    Calculates emitted wavefront using SRW.

    Args:
        photon_energy (float): Wavefront energy
        observation_point (float): Distance to the observation point
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        file_name (str): The name of the output file.
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        radiation_polarisation (str): Polarisation component to be extracted. Default is 'T'.
            = LH   - Linear Horizontal (0); 
            = LV   - Linear Vertical   (1); 
            = L45  - Linear 45°        (2); 
            = L135 - Linear 135°       (3); 
            = CR   - Circular Right    (4); 
            = CL   - Circular Left     (5); 
            = T    - Total             (6);
        number_macro_electrons (int): Number of macro electrons. Default is 1.
        parallel (bool): Whether to use parallel computation. Default is False.
        verbose (bool): If True, print log information

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """
    t0 = time.time()

    bl, eBeam, magFldCnt, eTraj, source_type, verbose = initialize_calculation(
        kwargs,
        observation_point=observation_point,
        hor_slit=hor_slit,
        ver_slit=ver_slit,
        hor_slit_cen=kwargs.get('hor_slit_cen', 0),
        ver_slit_cen=kwargs.get('ver_slit_cen', 0),
        calc_etraj=False
    )
    
    file_name = kwargs.get('file_name', None)
    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')
    number_macro_electrons = kwargs.get('number_macro_electrons', 1)
    parallel = kwargs.get('parallel', False)
    
    function_txt = f"{source_type} wavefront for a fixed energy using SRW:"
    if verbose: print(f"{function_txt} please wait...")

    calc_txt = "> Performing monochromatic wavefront calculation (___CALC___ simulation)"

    if number_macro_electrons == 0:
        calc_txt = calc_txt.replace("___CALC___", "filament beam")
        if verbose: print(f'{calc_txt} ... ', end='') 

    elif number_macro_electrons == 1:
        calc_txt = calc_txt.replace("___CALC___", "simplified partially coherent")
        if verbose: print(f'{calc_txt} ... ', end='')

    else:
        calc_txt = calc_txt.replace("___CALC___", "partially coherent")
        if verbose: print(f'{calc_txt} ... ', end='')

    if number_macro_electrons < 2:
        wfr, dt = srwlibCalcElecFieldSR(bl, 
                                eBeam, 
                                magFldCnt,
                                photon_energy,
                                hor_slit_n,
                                ver_slit_n,
                                0) 
    else:
        # TODO reimplement ME calculation
        pass

    if verbose: print('completed')

    wfrDict = write_wavefront(file_name, wfr, radiation_polarisation,
                              number_macro_electrons, observation_point)
    
    if verbose: print(f"{function_txt} finished.")
    if verbose: print_elapsed_time(t0)

    return wfrDict

#***********************************************************************************
# power distribution
#***********************************************************************************

def power_density(observation_point: float,
                  hor_slit: float, 
                  hor_slit_n: int,
                  ver_slit: float,
                  ver_slit_n: int,
                  **kwargs):
    """
    Calculates power density using SRW.

    Args:
        observation_point (float): Distance to the observation point
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        file_name (str): The name of the output file.
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        radiation_polarisation (str): Polarisation component to be extracted. Default is 'T'.
            = LH   - Linear Horizontal (0); 
            = LV   - Linear Vertical   (1); 
            = L45  - Linear 45°        (2); 
            = L135 - Linear 135°       (3); 
            = CR   - Circular Right    (4); 
            = CL   - Circular Left     (5); 
            = T    - Total             (6);

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """

    t0 = time.time()
    
    bl, eBeam, magFldCnt, eTraj, source_type, verbose = initialize_calculation(
        kwargs,
        observation_point=observation_point,
        hor_slit=hor_slit,
        ver_slit=ver_slit,
        hor_slit_cen=kwargs.get('hor_slit_cen', 0),
        ver_slit_cen=kwargs.get('ver_slit_cen', 0),
        calc_etraj=False
    )
    
    file_name = kwargs.get('file_name', None)
    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')
    
    function_txt = f"{source_type} power density using SRW:"
    if verbose: print(f"{function_txt} please wait...")
    
    pwr = srwlibCalcPowDenSR(bl, eBeam, magFldCnt, hor_slit_n, ver_slit_n)
    
    pwrDict = write_power_density(file_name, pwr, radiation_polarisation)
    
    if verbose:
        for pol in pwrDict:
            if pol == "axis":
                continue
            
            integrated = pwrDict[pol]["integrated"]
            peak = pwrDict[pol]["peak"]
            
            print(f"Polarisation component {pol}")
            print(f">>> Total received power: {integrated:.3e} W")
            print(f">>> Peak power density: {peak:.2f} W/mm^2")
    
    if verbose: print(f"{function_txt} finished.")
    if verbose: print_elapsed_time(t0)
    
    return pwrDict

#***********************************************************************************
# spectrum
#***********************************************************************************

def spectrum():
    raise NotImplementedError('Ohhh ohhh we are half way there! Ohhh ohhh this function is not implemented yet!')



#***********************************************************************************
# spectral wavefront calculation - 3D datasets
#***********************************************************************************

def emitted_radiation():
    raise NotImplementedError('Ohhh ohhh we are half way there! Ohhh ohhh this function is not implemented yet!')


#***********************************************************************************
# tuning curves
#***********************************************************************************

def tuning_curves():
    raise NotImplementedError('Ohhh ohhh we are half way there! Ohhh ohhh this function is not implemented yet!')

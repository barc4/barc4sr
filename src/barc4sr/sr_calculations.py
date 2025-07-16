
"""
This module provides functions for interfacing SRW when calculating wavefronts, 
synchrotron radiation, power density, and spectra. 
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '17/JUN/2025'
__changed__ = '16/JUL/2025'

import os
import time

import numpy as np

from barc4sr.aux_energy import (
    generate_logarithmic_energy_array,
    get_undulator_emission_energy,
)
from barc4sr.aux_rw import (
    write_cmd,
    write_electron_trajectory,
    write_power_density,
    write_spectrum,
    write_wavefront,
)
from barc4sr.aux_syned import barc4sr_dictionary, syned_dictionary
from barc4sr.aux_time import print_elapsed_time
from barc4sr.aux_utils import (
    set_light_source,
    spectral_srwlibCalcElecFieldSR,
    spectral_srwlibCalcStokesUR,
    srwlibCalcElecFieldSR,
    srwlibCalcPowDenSR,
    wofrySlitCMD,
    wofrySourceCMD,
)

PI = np.pi

#***********************************************************************************
# electron trajectory
#***********************************************************************************

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
        bl = syned_dictionary(json_file, None, 10, 1e-3, 1e-3, 0, 0)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, 10, 1e-3, 1e-3, 0, 0)

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

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, True, bl['Class'],
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               verbose=verbose)

    if verbose: print('completed')
    if verbose: print_elapsed_time(t0)

    eTrajDict = write_electron_trajectory(file_name, eTraj)

    return eTrajDict

#***********************************************************************************
# wavefront
#***********************************************************************************

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

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)
    file_name = kwargs.get('file_name', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, observation_point, 
                                hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

    if bl['Class'] == 'bm':
        source_type = 'bending magnet'
    elif bl['Class'] == 'u':
        source_type = 'undulator'
    elif bl['Class'] == 'w':
        source_type = 'wiggler'
    elif bl['Class'] == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} wavefront for a fixed energy using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')
    number_macro_electrons = kwargs.get('number_macro_electrons', 1)

    parallel = kwargs.get('parallel', False)

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, False, bl['Class'],
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               verbose=verbose)
    
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
                                bl['Class']) 
    else:
        # TODO reimplement ME calculation
        pass

    if verbose: print('completed')

    wfrDict = write_wavefront(file_name, wfr, radiation_polarisation, number_macro_electrons)
    
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

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)
    file_name = kwargs.get('file_name', None)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, observation_point, 
                                hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
        
    if bl['Class'] == 'bm':
        source_type = 'bending magnet'
    elif bl['Class'] == 'u':
        source_type = 'undulator'
    elif bl['Class'] == 'w':
        source_type = 'wiggler'
    elif bl['Class'] == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} power density using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, False, bl['Class'],
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               verbose=verbose)
    
    pwr = srwlibCalcPowDenSR(bl, eBeam, magFldCnt, hor_slit_n, ver_slit_n)

    pwrDict = write_power_density(file_name, pwr, radiation_polarisation)

    if verbose:
        for pol in pwrDict:
            if pol == "axis":
                continue  # skip the axis entry

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

def spectrum(energy_min: float,
             energy_max: float,
             num_points: int,
             observation_point: float,
             hor_slit: float, 
             ver_slit: float,
             **kwargs) -> dict:
    """
    Calculates 1D spectrum using SRW.

    Args:
        energy_min (float): Minimum photon energy [eV].
        energy_max (float): Maximum photon energy [eV].
        num_points (int): Number of photon energy points.
        observation_point (float): Distance to the observation point
        hor_slit (float): Horizontal slit size [m].
        ver_slit (float): Vertical slit size [m].

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
        energy_sampling (int): Energy sampling method (0: linear, 1: logarithmic). Default is 0.
        number_macro_electrons (int): Number of macro electrons. Default is 1.
        parallel (bool): Whether to use parallel computation. Default is False.
        verbose (bool): If True, print log information

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """
    
    t0 = time.time()

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)
    file_name = kwargs.get('file_name', None)
    energy_sampling = kwargs.get('energy_sampling', 0)
    parallel = kwargs.get('parallel', False)

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, observation_point, 
                                hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
        
    if bl['Class'] == 'bm':
        source_type = 'bending magnet'
    elif bl['Class'] == 'u':
        source_type = 'undulator'
    elif bl['Class'] == 'w':
        source_type = 'wiggler'
    elif bl['Class'] == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} spectrum using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')
    number_macro_electrons = kwargs.get('number_macro_electrons', 0)

    parallel = kwargs.get('parallel', False)

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, False, bl['Class'],
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               verbose=verbose)
    
    # ---------------------------------------------------------
    # Energy sampling
    if bl['Class'] == 'u':
        resonant_energy = get_undulator_emission_energy(
                                bl['PeriodID'], 
                                np.sqrt(bl['Kv']**2 + bl['Kh']**2),
                                bl['ElectronEnergy'])
    else:
        resonant_energy = kwargs.get('resonant_energy', None)
    if resonant_energy is None:
        raise ValueError("Please, provide resonant_energy for the logarithmic energy sampling (see function docstring)")

    if energy_sampling == 0:
        energy_array = np.linspace(energy_min, energy_max, num_points)
    else:           
        stepsize = np.log(energy_max/resonant_energy)/num_points
        energy_array = generate_logarithmic_energy_array(energy_min,
                                                         energy_max,
                                                         resonant_energy,
                                                         stepsize,
                                                         verbose)
        
    # ---------------------------------------------------------
    # On-Axis Spectrum 
    if bl['slitH'] < 1e-6 or bl['slitV'] < 1e-6:
        if parallel:
            if verbose: print('> Performing on-axis spectrum in parallel ... ')
        else:
            if verbose: print('> Performing on-axis spectrum ... ', end='')

        if bl['Class'] == 'u' and number_macro_electrons > 0:
            bl['slitH'] = 1e-6
            bl['slitV'] = 1e-6
            spectrum = spectral_srwlibCalcStokesUR(bl,
                                                   eBeam,
                                                   magFldCnt,
                                                   energy_array,
                                                   resonant_energy,
                                                   1,
                                                   1,
                                                   parallel,
                                                   radiation_polarisation,
                                                   verbose)
        else:
            spectrum = spectral_srwlibCalcElecFieldSR(bl, 
                                    eBeam, 
                                    magFldCnt,
                                    energy_array,
                                    1,
                                    1,
                                    bl['Class'],
                                    parallel,
                                    radiation_polarisation,
                                    0,
                                    verbose) 
    elif number_macro_electrons == 1:
        if parallel:
            if verbose: print('> Performing (simplified) spectrum through slit in parallel ... ')
        else:
            if verbose: print('> Performing (simplified) spectrum through slit ... ', end='')

        if bl['Class'] == 'u':
            spectrum = spectral_srwlibCalcStokesUR(bl,
                                                   eBeam,
                                                   magFldCnt,
                                                   energy_array,
                                                   resonant_energy,
                                                   1,
                                                   1,
                                                   parallel,
                                                   radiation_polarisation,
                                                   verbose)

    else:
        if parallel:
            if verbose: print('> Performing (accurate) spectrum through slit in parallel ... ')
        else:
            if verbose: print('> Performing (accurate) spectrum through slit ... ', end='')

        if bl['Class'] == 'u':
            spectrum = spectral_srwlibCalcStokesUR(bl,
                                                   eBeam,
                                                   magFldCnt,
                                                   energy_array,
                                                   resonant_energy,
                                                   1,
                                                   1,
                                                   parallel,
                                                   radiation_polarisation,
                                                   verbose)

    if verbose: print('completed')

    spectrumDict = write_spectrum(file_name, spectrum)

    if verbose: print(f"{function_txt} finished.")
    if verbose: print_elapsed_time(t0)

    return spectrumDict

#***********************************************************************************
# coherent modes
#***********************************************************************************

def coherent_modes(photon_energy: float,
                   observation_point: float,
                   hor_slit: float, 
                   ver_slit: float,
                   **kwargs) -> dict:
    """
    Calculates the 1D coherent mode decomposition (CMD) for an undulator source using WOFRY.

    Args:
        photon_energy (float): Photon energy [eV].
        observation_point (float): Distance to the observation point [m].
        hor_slit (float): Horizontal slit size [m].
        ver_slit (float): Vertical slit size [m].

    Optional Args (kwargs):
        file_name (str): The name of the output file.
        json_file (str): Path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.UndulatorSource object.
        verbose (bool): If True, print log information. Default is True.

    Returns:
        dict: Dictionary containing:
            - 'energy': Photon energy [eV].
            - 'source': For each direction ('H', 'V'):
                - 'eigenvalues': Array of eigenvalues.
                - 'axis': Array of abscissas.
                - 'occupation': Normalised occupation array.
                - 'cumulated': Cumulative sum of occupation.
                - 'CF': Coherent fraction (first mode occupation).
                - 'CSD': Cross-spectral density matrix.
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

    hor_slit_cen = 0 # kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = 0 # kwargs.get('ver_slit_cen', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, observation_point, 
                                hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

    if bl['Class'] != 'u':
        raise ValueError("This function is only valid for ideal undulators (sr_classes.UndulatorSource)")

    source_type = 'undulator'
    function_txt = f"{source_type} 1D CMD for a fixed energy using WOFRY:"

    if verbose: print(f"{function_txt} please wait...")

    src_h_cmd = wofrySourceCMD(bl, photon_energy, 'H')
    src_v_cmd = wofrySourceCMD(bl, photon_energy, 'V')

    # slit_h_cmd = wofrySlitCMD(src_h_cmd, hor_slit, observation_point)

    if verbose: print('completed')

    cmdDict = write_cmd(file_name, {'src_h_cmd':src_h_cmd, 'src_v_cmd':src_v_cmd, 'energy': photon_energy})

    if verbose: print(f"{function_txt} finished.")
    if verbose: print_elapsed_time(t0)

    return cmdDict


#***********************************************************************************
# spectral wavefront calculation - 3D datasets
#***********************************************************************************

def emitted_radiation():
    raise NotImplementedError('Ohhh ohhh we are half way there! Ohhh ohhh living on a prayer! - this function is not implemented yet!')



#***********************************************************************************
# tuning curves
#***********************************************************************************

def tuning_curves():
    raise NotImplementedError('Ohhh ohhh we are half way there! Ohhh ohhh living on a prayer! - this function is not implemented yet!')


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
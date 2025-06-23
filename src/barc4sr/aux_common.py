
"""
This module provides common functions for interfacing SRW when calculating wavefronts, 
synchrotron radiation, power density, and spectra. 
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '17/JUN/2025'
__changed__ = '17/JUN/2025'

import time

from barc4sr.aux_rw import write_electron_trajectory, write_wavefront
from barc4sr.aux_syned import barc4sr_dictionary, syned_dictionary
from barc4sr.aux_time import print_elapsed_time
from barc4sr.aux_utils import set_light_source, srwlibCalcElecFieldSR_2D

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def electron_trajectory(file_name: str,
                        magnetic_structure_type: str, 
                        **kwargs) -> dict:
    """
    Calculate electron trajectory using SRW.

    Args:
        file_name (str): The name of the output file.
        magnetic_structure_type (str): type of mag. structure: 'u', 'bm' or 'arb'

    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class

    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if magnetic_structure_type == 'bm':
        source_type = 'bending magnet'
    elif magnetic_structure_type == 'u':
        source_type = 'undulator'
    elif magnetic_structure_type == 'w':
        source_type = 'wiggler'
    elif magnetic_structure_type == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} electron trajectory using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, 10, 1e-3, 1e-3, 0, 0)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, 10, 1e-3, 1e-3, 0, 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, True, magnetic_structure_type,
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

def wavefront(file_name: str, 
              photon_energy: float,
              hor_slit: float, 
              hor_slit_n: int,
              ver_slit: float,
              ver_slit_n: int,
              magnetic_structure_type: str, 
              **kwargs):
    
    """
    Calculate emitted wavefront using SRW.

    Args:
        file_name (str): The name of the output file.
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.
        magnetic_structure_type (str): type of mag. structure: 'u', 'bm' or 'arb'

    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        observation_point (float): Distance to the observation point. Default is 10 [m].
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
        number_macro_electrons (int): Number of macro electrons. Default is 0.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. 
                                    If not specified, it defaults to the number of available CPU cores.

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """
    t0 = time.time()

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if magnetic_structure_type == 'bm':
        source_type = 'bending magnet'
    elif magnetic_structure_type == 'u':
        source_type = 'undulator'
    elif magnetic_structure_type == 'w':
        source_type = 'wiggler'
    elif magnetic_structure_type == 'arb':
        source_type = 'arbitrary magnetic field'
        
    function_txt = f"{source_type} wavefront for a fixed energy using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")
    
    observation_point = kwargs.get('observation_point', 10.)

    hor_slit_cen = kwargs.get('hor_slit_cen', 0)
    ver_slit_cen = kwargs.get('ver_slit_cen', 0)

    radiation_polarisation = kwargs.get('radiation_polarisation', 'T')
    number_macro_electrons = kwargs.get('number_macro_electrons', -1)

    parallel = kwargs.get('parallel', False)
    num_cores = kwargs.get('num_cores', None)

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, None, observation_point, 
                              hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

    eBeam, magFldCnt, eTraj = set_light_source(bl, True, magnetic_structure_type,
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
        wfr = srwlibCalcElecFieldSR_2D(bl, 
                                eBeam, 
                                magFldCnt,
                                photon_energy,
                                h_slit_points=hor_slit_n,
                                v_slit_points=ver_slit_n,
                                id_type = magnetic_structure_type) 
    else:
        # TODO reimplement ME calculation
        pass

    if verbose: print('completed')

    wfrDict = write_wavefront(file_name, wfr, radiation_polarisation, number_macro_electrons)
    
    if verbose: print(f"{function_txt} finished.")
    if verbose: print_elapsed_time(t0)

    return wfrDict
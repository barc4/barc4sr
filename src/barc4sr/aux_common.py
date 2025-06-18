
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

from barc4sr.aux_rw import write_electron_trajectory
from barc4sr.aux_syned import barc4sr_dictionary, syned_dictionary
from barc4sr.aux_time import print_elapsed_time
from barc4sr.aux_utils import set_light_source

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def electron_trajectory(file_name: str, magnetic_structure_type: str, **kwargs) -> dict:
    """
    Calculate undulator electron trajectory using SRW.

    Args:
        file_name (str): The name of the output file.


    Optional Args (kwargs):
        json_file (optional): The path to the SYNED JSON configuration file.
        light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
        magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data. Default is None.


    Returns:
        Dict: A dictionary containing arrays of photon energy and flux.
    """

    t0 = time.time()

    verbose = kwargs.get('verbose', True)
    json_file = kwargs.get('json_file', None)
    light_source = kwargs.get('light_source', None)

    if magnetic_structure_type == 'bm':
        source_type = 'bending magnet'
        
    function_txt = f"{source_type} electron trajectory using SRW:"

    if verbose: print(f"{function_txt} please wait...")

    if json_file is None and light_source is None:
        raise ValueError("Please, provide either json_file or light_source (see function docstring)")
    if json_file is not None and light_source is not None:
        raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

    if json_file is not None:
        bl = syned_dictionary(json_file, magnetic_measurement, 10, 1e-3, 1e-3, 0, 0)
    if light_source is not None:
        bl = barc4sr_dictionary(light_source, magnetic_measurement, 10, 1e-3, 1e-3, 0, 0)

    eBeam, magFldCnt, eTraj = set_light_source(bl, True, magnetic_structure_type,
                                               ebeam_initial_position=ebeam_initial_position,
                                               magfield_central_position=magfield_central_position,
                                               magnetic_measurement=magnetic_measurement,
                                               verbose=verbose)

    if verbose: print('completed')
    if verbose: print_elapsed_time(t0)

    eTrajDict = write_electron_trajectory(file_name, eTraj)

    return eTrajDict

#***********************************************************************************
# wavefront
#***********************************************************************************
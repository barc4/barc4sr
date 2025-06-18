#!/bin/python

"""
This module provides functions for interfacing SRW when calculating wavefronts, 
synchrotron radiation, power density, and spectra. This module is based on the 
xoppy.sources module from https://github.com/oasys-kit/xoppylib
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '12/MAR/2024'
__changed__ = '17/JUN/2025'

import os
import barc4sr.aux_common as cm

magnetic_structure_type = 'bm'

#***********************************************************************************
# electron trajectory
#***********************************************************************************

def electron_trajectory(*args, **kwargs) -> dict:
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
    return cm.electron_trajectory(*args, magnetic_structure_type=magnetic_structure_type, **kwargs)

#***********************************************************************************
# Wavefront
#***********************************************************************************

def emitted_wavefront(*args, **kwargs) -> dict:
    """
    Calculate bending magnet emitted wavefront using SRW.

    Args:
        file_name (str): The name of the output file.
        json_file (str): The path to the SYNED JSON configuration file.
        hor_slit (float): Horizontal slit size [m].
        hor_slit_n (int): Number of horizontal slit points.
        ver_slit (float): Vertical slit size [m].
        ver_slit_n (int): Number of vertical slit points.

    Optional Args (kwargs):
        observation_point (float): Distance to the observation point. Default is 10 [m].
        hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
        ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
        radiation_polarisation (int): Polarisation component to be extracted. Default is 6.
            =0 -Linear Horizontal; 
            =1 -Linear Vertical; 
            =2 -Linear 45 degrees; 
            =3 -Linear 135 degrees; 
            =4 -Circular Right; 
            =5 -Circular Left; 
            =6 -Total
        number_macro_electrons (int): Number of macro electrons. Default is -1.
        parallel (bool): Whether to use parallel computation. Default is False.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                        it defaults to the number of available CPU cores.

    Returns:
        Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
    """

    return cm.emitted_wavefront(*args, magnetic_structure_type=magnetic_structure_type, **kwargs)

# def emitted_wavefront(file_name: str, 
#                       photon_energy: float,
#                       hor_slit: float, 
#                       hor_slit_n: int,
#                       ver_slit: float,
#                       ver_slit_n: int,
#                       **kwargs) -> Dict:
#     """
#     Calculate bending magnet emitted wavefront using SRW.

#     Args:
#         file_name (str): The name of the output file.
#         json_file (str): The path to the SYNED JSON configuration file.
#         hor_slit (float): Horizontal slit size [m].
#         hor_slit_n (int): Number of horizontal slit points.
#         ver_slit (float): Vertical slit size [m].
#         ver_slit_n (int): Number of vertical slit points.

#     Optional Args (kwargs):
#         observation_point (float): Distance to the observation point. Default is 10 [m].
#         hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
#         ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
#         radiation_polarisation (int): Polarisation component to be extracted. Default is 6.
#             =0 -Linear Horizontal; 
#             =1 -Linear Vertical; 
#             =2 -Linear 45 degrees; 
#             =3 -Linear 135 degrees; 
#             =4 -Circular Right; 
#             =5 -Circular Left; 
#             =6 -Total
#         magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
#             Overrides SYNED bending magnet data. Default is None.
#         number_macro_electrons (int): Number of macro electrons. Default is -1.
#         parallel (bool): Whether to use parallel computation. Default is False.
#         num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
#                                         it defaults to the number of available CPU cores.

#     Returns:
#         Dict: A dictionary intensity, phase, horizontal axis, and vertical axis.
#     """

#     t0 = time.time()

#     function_txt = "bending magnet spatial distribution for a given energy using SRW:"
#     calc_txt = "> Performing monochromatic wavefront calculation (___CALC___ partially-coherent simulation)"
#     print(f"{function_txt} please wait...")

#     json_file = kwargs.get('json_file', None)
#     light_source = kwargs.get('light_source', None)

#     if json_file is None and light_source is None:
#         raise ValueError("Please, provide either json_file or light_source (see function docstring)")
#     if json_file is not None and light_source is not None:
#         raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

#     observation_point = kwargs.get('observation_point', 10.)

#     hor_slit_cen = kwargs.get('hor_slit_cen', 0)
#     ver_slit_cen = kwargs.get('ver_slit_cen', 0)

#     radiation_polarisation = kwargs.get('radiation_polarisation', 6)

#     magnetic_measurement = kwargs.get('magnetic_measurement', None)

#     number_macro_electrons = kwargs.get('number_macro_electrons', -1)

#     parallel = kwargs.get('parallel', False)
#     num_cores = kwargs.get('num_cores', None)

#     if number_macro_electrons <= 0 :
#         calculation = 0
#     else:
#         calculation = 1

#     if json_file is not None:
#         bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
#                             hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
#     if light_source is not None:
#         bl = barc4sr_dictionary(light_source, magnetic_measurement, observation_point, 
#                             hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

   
#     eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, False, 'u',
#                                                magnetic_measurement=magnetic_measurement,
#                                                tabulated_undulator_mthd=tabulated_undulator_mthd)
    
#     # -----------------------------------------
#     # Spatially limited monochromatic wavefront
        
#     # simplified partially-coherent simulation    
#     if calculation == 0:
#         calc_txt = calc_txt.replace("___CALC___", "simplified")
#         print(f'{calc_txt} ... ', end='')

#         intensity, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
#                                                           eBeam, 
#                                                           magFldCnt,
#                                                           photon_energy,
#                                                           h_slit_points=hor_slit_n,
#                                                           v_slit_points=ver_slit_n,
#                                                           radiation_characteristic=1, 
#                                                           radiation_dependence=3,
#                                                           radiation_polarisation=radiation_polarisation,
#                                                           id_type = 'u',
#                                                           parallel=False)     
        
#         phase, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
#                                                       eBeam, 
#                                                       magFldCnt,
#                                                       photon_energy,
#                                                       h_slit_points=hor_slit_n,
#                                                       v_slit_points=ver_slit_n,
#                                                       radiation_characteristic=4, 
#                                                       radiation_dependence=3,
#                                                       radiation_polarisation=radiation_polarisation,
#                                                       id_type = 'u',
#                                                       parallel=False)     

#         # phase = unwrap_wft_phase(phase, h_axis, v_axis, observation_point, photon_energy)
#         print('completed')

#     # accurate partially-coherent simulation
#     if calculation == 1:
#         calc_txt = calc_txt.replace("___CALC___", "accurate")
#         if parallel:
#             print(f'{calc_txt} in parallel... ')
#         else:
#             print(f'{calc_txt} ... ', end='')

#         if file_name is None:
#             aux_file_name = 'emitted_radiation'

#         intensity, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
#                                                                      eBeam,
#                                                                      magFldCnt,
#                                                                      photon_energy,
#                                                                      hor_slit_n,
#                                                                      ver_slit_n,
#                                                                      radiation_polarisation=radiation_polarisation,
#                                                                      id_type = 'u',
#                                                                      number_macro_electrons=number_macro_electrons,
#                                                                      aux_file_name=aux_file_name,
#                                                                      parallel=False,
#                                                                      num_cores=num_cores,
#                                                                      srApprox=1) 
        
#         phase = np.zeros(intensity.shape)
#         print('completed')

#     wftDict = write_wavefront(file_name, intensity, phase, h_axis, v_axis)

#     print(f"{function_txt} finished.")
#     print_elapsed_time(t0)

#     return wftDict

#***********************************************************************************
# Bending magnet radiation
#***********************************************************************************
 
# def spectrum(file_name: str,
#              photon_energy_min: float,
#              photon_energy_max: float,
#              photon_energy_points: int, 
#              **kwargs) -> Dict:
#     """
#     Calculate 1D bending magnet spectrum using SRW.

#     Args:
#         file_name (str): The name of the output file.
#         photon_energy_min (float): Minimum photon energy [eV].
#         photon_energy_max (float): Maximum photon energy [eV].
#         photon_energy_points (int): Number of photon energy points.

#     Optional Args (kwargs):
#         json_file (optional): The path to the SYNED JSON configuration file.
#         light_source (optional): barc4sr.aux_utils.SynchrotronSource or inheriting class
#         energy_sampling (int): Energy sampling method (0: linear, 1: logarithmic). Default is 0.
#         observation_point (float): Distance to the observation point. Default is 10 [m].
#         hor_slit (float): Horizontal slit size [m]. Default is 1e-3 [m].
#         ver_slit (float): Vertical slit size [m]. Default is 1e-3 [m].
#         hor_slit_cen (float): Horizontal slit center position [m]. Default is 0.
#         ver_slit_cen (float): Vertical slit center position [m]. Default is 0.
#         radiation_polarisation (int): Polarisation component to be extracted. Default is 6.
#             =0 -Linear Horizontal; 
#             =1 -Linear Vertical; 
#             =2 -Linear 45 degrees; 
#             =3 -Linear 135 degrees; 
#             =4 -Circular Right; 
#             =5 -Circular Left; 
#             =6 -Total
#         ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
#         magfield_central_position (float): Longitudinal position of the magnet center [m]
#         magnetic_measurement (Optional[str]): Path to the file containing magnetic measurement data.
#             Overrides SYNED undulator data. Default is None.            
#         number_macro_electrons (int): Number of macro electrons. Default is 1000.
#         parallel (bool): Whether to use parallel computation. Default is False.
#         num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
#                                         it defaults to the number of available CPU cores.

#     Returns:
#         Dict: A dictionary containing arrays of photon energy and flux.
#     """

#     t0 = time.time()

#     function_txt = "Bending magnet spectrum calculation using SRW:"
#     calc_txt = "> Performing flux through finite aperture (___CALC___ partially-coherent simulation)"
#     print(f"{function_txt} please wait...")

#     json_file = kwargs.get('json_file', None)
#     light_source = kwargs.get('light_source', None)

#     if json_file is None and light_source is None:
#         raise ValueError("Please, provide either json_file or light_source (see function docstring)")
#     if json_file is not None and light_source is not None:
#         raise ValueError("Please, provide either json_file or light_source - not both (see function docstring)")

#     energy_sampling = kwargs.get('energy_sampling', 0)

#     observation_point = kwargs.get('observation_point', 10.)
    
#     hor_slit = kwargs.get('hor_slit', 1e-3)
#     ver_slit = kwargs.get('ver_slit', 1e-3)
#     hor_slit_cen = kwargs.get('hor_slit_cen', 0)
#     ver_slit_cen = kwargs.get('ver_slit_cen', 0)
    
#     radiation_polarisation = kwargs.get('radiation_polarisation', 6)
    
#     ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)

#     magfield_central_position = kwargs.get('magfield_central_position', 0)
#     magnetic_measurement = kwargs.get('magnetic_measurement', None)

#     number_macro_electrons = kwargs.get('number_macro_electrons', 1)

#     parallel = kwargs.get('parallel', False)
#     num_cores = kwargs.get('num_cores', None)

#     if hor_slit < 1e-6 and ver_slit < 1e-6:
#         calculation = 0
#         hor_slit = 0
#         ver_slit = 0
#     else:
#         if magnetic_measurement is None and number_macro_electrons == 1:
#             calculation = 1
#         if magnetic_measurement is not None or number_macro_electrons > 1:
#             calculation = 2

#     if json_file is not None:
#         bl = syned_dictionary(json_file, magnetic_measurement, observation_point, 
#                             hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)
#     if light_source is not None:
#         bl = barc4sr_dictionary(light_source, magnetic_measurement, observation_point, 
#                             hor_slit, ver_slit, hor_slit_cen, ver_slit_cen)

#     eBeam, magFldCnt, eTraj = set_light_source(file_name, bl, False, 'bm',
#                                                ebeam_initial_position=ebeam_initial_position,
#                                                magfield_central_position=magfield_central_position,
#                                                magnetic_measurement=magnetic_measurement
#                                                )

#     # ----------------------------------------------------------------------------------
#     # spectrum calculations
#     # ----------------------------------------------------------------------------------

#     if energy_sampling == 0: 
#         energy = np.linspace(photon_energy_min, photon_energy_max, photon_energy_points)
#     else:
#         stepsize = np.log(photon_energy_max/photon_energy_min)
#         energy = generate_logarithmic_energy_values(photon_energy_min,
#                                                     photon_energy_max,
#                                                     photon_energy_min,
#                                                     stepsize)
#     # ---------------------------------------------------------
#     # On-Axis Spectrum from Filament Electron Beam
#     if calculation == 0:
#         if parallel:
#             print('> Performing on-axis spectrum from filament electron beam in parallel ... ')
#         else:
#             print('> Performing on-axis spectrum from filament electron beam ... ', end='')

#         flux, h_axis, v_axis = srwlibCalcElecFieldSR(bl, 
#                                                      eBeam, 
#                                                      magFldCnt,
#                                                      energy,
#                                                      h_slit_points=1,
#                                                      v_slit_points=1,
#                                                      radiation_characteristic=0, 
#                                                      radiation_dependence=0,
#                                                      radiation_polarisation=radiation_polarisation,
#                                                      id_type='bm',
#                                                      parallel=parallel,
#                                                      num_cores=num_cores)
#         flux = flux.reshape((photon_energy_points))
#         print('completed')

#     # -----------------------------------------
#     # Flux through Finite Aperture

#     # # simplified partially-coherent simulation
#     # if calculation == 1:
#     #     calc_txt = calc_txt.replace("___CALC___", "simplified")
#     #     if parallel:
#     #         print(f'{calc_txt} in parallel... ')
#     #     else:
#     #         print(f'{calc_txt} ... ', end='')

#     #     # RC:2025JAN08 TODO: check the best implementation
#     #     print('completed')

#     # accurate partially-coherent simulation
#     if calculation == 2:
#         calc_txt = calc_txt.replace("___CALC___", "accurate")
#         if parallel:
#             print(f'{calc_txt} in parallel... ')
#         else:
#             print(f'{calc_txt} ... ', end='')

#         flux, h_axis, v_axis = srwlibsrwl_wfr_emit_prop_multi_e(bl, 
#                                                                 eBeam,
#                                                                 magFldCnt,
#                                                                 energy,
#                                                                 h_slit_points=1,
#                                                                 v_slit_points=1,
#                                                                 radiation_polarisation=radiation_polarisation,
#                                                                 id_type='bm',
#                                                                 number_macro_electrons=number_macro_electrons,
#                                                                 aux_file_name=file_name,
#                                                                 parallel=parallel,
#                                                                 num_cores=num_cores)       
#         print('completed')

#     spectrumSRdict = write_spectrum(file_name, flux, energy)

#     print(f"{function_txt} finished.")

#     print_elapsed_time(t0)

#     return spectrumSRdict


if __name__ == '__main__':

    file_name = os.path.basename(__file__)

    print(f"This is the barc4sr.{file_name} module!")
    print("This module provides functions for interfacing SRW when calculating wavefronts, synchrotron radiation, power density, and spectra.")
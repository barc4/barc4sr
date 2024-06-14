
"""
This module provides...
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '26/JAN/2024'
__changed__ = '21/MAY/2024'

from typing import Optional, Tuple, Union

import numpy as np
import xraylib
from xoppylib.scattering_functions.xoppy_calc_f1f2 import xoppy_calc_f1f2

#***********************************************************************************
# reflectivity curves
#***********************************************************************************

def nist_compound_list() -> list:
    """
    Retrieve and returns the list of NIST compounds.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
    return xraylib.GetCompoundDataNISTList()


def density_element(material:Union[str, int, float], verbose: bool = False) -> float:
    """
    Get the density of an element based on its atomic number or symbol.

    Args:
        material (str | int | float): The atomic symbol (str), atomic number (int), or atomic number (float).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the element.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
    if isinstance(material, str):
        Z = xraylib.SymbolToAtomicNumber(material)
    elif isinstance(material, int):
        Z = material
    elif isinstance(input, float):
        Z = int(material)
    else:
        raise Exception("Bad input.")
    density = xraylib.ElementDensity(Z)
    if verbose:
        print("Density for %s (Z=%d): %6.3f g/cm3"%(material, Z, density))
    return density


def density_nist(material: Union[str, int], verbose: bool = False) -> float:
    """
    Get the density of a NIST compound based on its name or index.

    Args:
        material (str | int): The compound name (str) or index (int).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the compound.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """

    if isinstance(material, str):
        Zarray = xraylib.GetCompoundDataNISTByName(material)
        density = Zarray["density"]
        if verbose:
            print("Density for %s: %6.3f g/cm3"%(material, density))
    elif isinstance(material, int):
        Zarray = xraylib.GetCompoundDataNISTByIndex(material)
        density = Zarray["density"]
        if verbose:
            print("Density for NIST compound with index %d: %6.3f " % (material, density))

    return density


def get_density(material: str, verbose: bool = False) -> float:
    """
    Determine the density of a material based on its descriptor.

    Args:
        material (str): The descriptor of the material (atomic symbol, compound name, etc.).
        verbose (bool): If True, prints the density. Defaults to False.

    Returns:
        float: The density of the material.

    Note:
        This function is taken from the xoppylib interface for xraylib.
    """
        
    if not isinstance(material, str):
        raise Exception("descriptor must be a string!")
    kind = -1
    if len(material) <=2:
        Z = xraylib.SymbolToAtomicNumber(material)
        if Z > 0:
            kind = 0
    elif material in nist_compound_list():
        kind = 2
    else:
        try:
            xraylib.CompoundParser(material)
            kind = 1
        except:
            pass

    if kind == 0:
        return density_element(material, verbose)
    elif kind == 1:
        raise Exception("cannot retrieve density for a compound (%s): it must be defined by user" % material)
    elif kind == 2:
        return density_nist(material, verbose)
    else:
        raise Exception("Unknown descriptor: %s" % material)
        

def reflectivity_curve(material: str, density: float, theta: float, ei: float, ef: float, ne: int,
                       e_axis: Optional[np.ndarray] = None) -> dict:
    """ 
    Calculate the reflectivity for a given material and conditions.

    Args:
        material (str): The material's name.
        density (float): The material's density in grams per cubic centimeter (g/cm^3).
        theta (float): The angle of incidence in milliradians (mrad).
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy steps.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis for point-wise calculation. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "reflectivity" (np.ndarray): The reflectivity values.
            - "energy" (np.ndarray): The corresponding energy values.
            - "theta" (float): The angle of incidence.
            - "material" (str): The material's name. 
    """

    if density == -1:
        density = get_density(material, True)

    if len(material) <=2:
        mat_flag = 0
    else:
        mat_flag = 1

    if e_axis is None:
        out_dict =  xoppy_calc_f1f2(
                descriptor   = material,
                density      = density,
                MAT_FLAG     = mat_flag,
                CALCULATE    = 9,
                GRID         = 1,
                GRIDSTART    = ei,
                GRIDEND      = ef,
                GRIDN        = ne,
                THETAGRID    = 0,
                ROUGH        = 0.0,
                THETA1       = theta,
                THETA2       = 5.0,
                THETAN       = 50,
                DUMP_TO_FILE = 0,
                FILE_NAME    = "%s.dat"%material,
                material_constants_library = xraylib,
            )
        
        energy_axis = out_dict["data"][0,:]
        reflectivity = out_dict["data"][-1,:]
    else:
        energy_axis = np.array([])
        reflectivity = np.array([])
        for E in e_axis:
            out_dict = xoppy_calc_f1f2(
                descriptor=material,
                density=density,
                MAT_FLAG=mat_flag,
                CALCULATE=9,
                GRID=2,
                GRIDSTART=E,
                GRIDEND=E,
                GRIDN=1,
                THETAGRID=0,
                ROUGH=0.0,
                THETA1=theta,
                THETA2=5.0,
                THETAN=50,
                DUMP_TO_FILE=0,
                FILE_NAME="%s.dat" % material,
                material_constants_library=xraylib,
            )
            energy_axis = np.append(energy_axis, out_dict["data"][0, :])
            reflectivity = np.append(reflectivity, out_dict["data"][-1, :])

    return {
        "reflectivity": reflectivity,
        "energy": energy_axis,
        "theta": theta,
        "material": material,
    }


def reflectivity_map(material: str, density: float, thetai: float, thetaf: float,
                     ntheta: int, ei: float, ef: float, ne: int,
                     e_axis: Optional[np.ndarray] = None) -> dict:
    """
    Compute a reflectivity map for a given material over a range of angles and energies.

    Args:
        material (str): The material's name.
        density (float): The material's density in g/cm^3.
        thetai (float): The initial angle of incidence in milliradians (mrad).
        thetaf (float): The final angle of incidence in milliradians (mrad).
        ntheta (int): The number of angles between thetai and thetaf.
        ei (float): The initial energy in electron volts (eV).
        ef (float): The final energy in electron volts (eV).
        ne (int): The number of energy points between ei and ef.
        e_axis (Optional[np.ndarray], optional): An array representing the energy axis. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "reflectivity_map" (np.ndarray): The reflectivity map.
            - "energy" (np.ndarray): The energy axis.
            - "theta" (np.ndarray): The angle axis.
            - "material" (str): The material's name. 
    """

    if density == -1:
        density = get_density(material)
        
    theta = np.linspace(thetai, thetaf, ntheta)

    if e_axis is None:
        reflectivity_map = np.zeros((ntheta, ne))
    else:
        reflectivity_map = np.zeros((ntheta, len(e_axis)))

    for k, th in enumerate(theta):
        result = reflectivity_curve(material, density, th, ei, ef, ne, e_axis)
        reflectivity_map[k, :] = result["reflectivity"]
    return {
        "reflectivity": reflectivity_map,
        "energy": result["energy"],
        "theta": theta,
        "material": material,
    }


def ml_reflectivity_curve():
    #TODO: interface for xoppy and DarpanX
    pass


def ml_reflectivity_map():
    pass


def crystal_reflectivity_curve():
    #TODO:  interface for xoppy and CrystalPy
    pass


def crystal__reflectivity_map():
    pass


#***********************************************************************************
# reflected and absorbed power
#***********************************************************************************

# def pow_ref_and_abs_from_spectrum(theta: float, matDict: Dict[str, Any], spectrumSRdict: dict) -> dict:
#     """
#     Calculate power reflection and absorption from a given spectrum for various materials.

#     Args:
#         theta (float): The angle of incidence in mrad.
#         matDict (Dict[str, Any]): A dictionary containing material information. It should have the following structure:
#             {
#                 "energy": array_like,
#                 "material_1": {
#                     "density": float,
#                     "flag": int,
#                     "reflectivity": array_like,
#                     "reflected_power": float,
#                     "absorbed_power": float
#                 },
#                 "material_2": {
#                     "density": float,
#                     "flag": int,
#                     "reflectivity": array_like,
#                     "reflected_power": float,
#                     "absorbed_power": float
#                 },
#                 ...
#             }
#         spectrumSRdict (dict): A dictionary containing the spectral power information. It should have the following structure:
#             {
#                 "spectrum": {
#                     "energy": array_like,
#                     "spectral_power": array_like
#                 }
#             }
#     Returns:
#         dict: A dictionary containing updated material information after calculating reflection and absorption powers.
#     """

#     matDict["energy"] = spectrumSRdict["spectrum"]["energy"]

#     k = 0
#     for key in matDict:
#         if key != "energy":
#             matDict[key]["angle"] = np.degrees(theta*1e-3)
#             matDict[key]["reflectivity"], ene = reflectivity_curve(key, matDict[key]["density"], 
#                                                                    theta, 0, 0, 1, spectrumSRdict["spectrum"]["energy"], mat_flag=matDict[key]["flag"])  

#             matDict[key]["reflected_power"] = integrate.trapezoid(np.multiply(matDict[key]["reflectivity"], 
#                                                                               spectrumSRdict["spectrum"]["spectral_power"]),
#                                                                   spectrumSRdict["spectrum"]["energy"])
#             matDict[key]["absorbed_power"] = integrate.trapezoid(np.multiply(1-matDict[key]["reflectivity"], 
#                                                                              spectrumSRdict["spectrum"]["spectral_power"]),
#                                                                  spectrumSRdict["spectrum"]["energy"])
#             print(f"> {key}")
#             mirr_reflec = matDict[key]["reflected_power"] 
#             mirr_absorb = matDict[key]["absorbed_power"]
#             print(f">> Reflected power:{mirr_reflec:.2f} W")
#             print(f">> Absorbed power :{mirr_absorb:.2f} W")
#             k+=1

#     return matDict


# def pow_ref_and_abs_from_undulator_radiation(theta: float, reflec_plane: str, matDict: Dict[str, Any], URdict: Dict[str, Any], PowDenSR: Dict[str, Any]=None) -> Dict[str, Any]:
#     """
#     Calculate the reflected and absorbed power from undulator radiation.

#     Args:
#         theta (float): Angle of incidence in milliradians.
#         reflec_plane (str): Plane of reflection ("h" for horizontal, "v" for vertical).
#         matDict (Dict[str, Any]): A dictionary containing material information with the following structure:
#             {
#                 "material_symbol": {
#                     "density": float,
#                     "flag": int,
#                 },
#             }     
#         URdict (Dict[str, Any]): Dictionary containing undulator radiation data.
#         PowDenSR (Dict[str, Any], optional): Dictionary containing integrated power density over the same window as URdict.

#     Returns:
#         Dict[str, Any]: Dictionary containing the calculated power values.
#             - "axis": Dictionary containing x and y axis values.
#                 - "x": X-axis values.
#                 - "y": Y-axis values.
#             - "projected_power": Dictionary containing projected power values.
#                 - "map": Projected power density map.
#                 - "CumPow": Cummulated projected power.
#                 - "PowDenSRmax": Maximum projected power density.
#             - "reflected_power": Dictionary containing reflected power values.
#                 - "map": Reflected power density map.
#                 - "CumPow": Cummulated reflected power.
#                 - "PowDenSRmax": Maximum reflected power density.
#             - "absorbed_power": Dictionary containing absorbed power values.
#                 - "map": Absorbed power density map.
#                 - "CumPow": Cummulated absorbed power.
#                 - "PowDenSRmax": Maximum absorbed power density.
#     """
    
#     powerDict = {
#         "axis": {
#             "x": None,
#             "y": None,
#         },    
#         "projected_power": {
#             "map": None,
#             "CumPow": None,
#             "PowDenSRmax": None
#         },
#         "reflected_power": {
#             "map": None,
#             "CumPow": None,
#             "PowDenSRmax": None
#         },
#         "absorbed_power": {
#             "map": None,
#             "CumPow": None,
#             "PowDenSRmax": None
#         },        
#     }
#     if reflec_plane == "h":
#         powerDict["axis"]["x"] = URdict["axis"]["x"]/np.sin(theta/1000)
#         powerDict["axis"]["y"] = URdict["axis"]["y"]
#     else:
#         powerDict["axis"]["x"] = URdict["axis"]["x"]
#         powerDict["axis"]["y"] = URdict["axis"]["y"]/np.sin(theta/1000)

#     dx = powerDict["axis"]["x"][1]-powerDict["axis"]["x"][0]
#     dy = powerDict["axis"]["y"][1]-powerDict["axis"]["y"][0]

#     for key in matDict:
#         reflectivity, ene = reflectivity_curve(key, matDict[key]["density"], theta, 0, 0, 1, 
#                                                URdict["spectrum"]["energy"], mat_flag=matDict[key]["flag"])  

#     powerDict["reflected_power"]["map"] = integrate.trapezoid(
#         np.multiply(reflectivity[:, np.newaxis, np.newaxis], URdict["spectral_power_3D"]), 
#         URdict["spectrum"]["energy"], axis=0
#         )*np.sin(theta/1000)
#     powerDict["reflected_power"]["CumPow"] = powerDict["reflected_power"]["map"].sum()*dx*dy
#     powerDict["reflected_power"]["PowDenSRmax"] = powerDict["reflected_power"]["map"].max()

#     if PowDenSR is None:
#         powerDict["projected_power"]["map"] = URdict["power_density"]["map"]*np.sin(theta/1000)

#         powerDict["absorbed_power"]["map"] = integrate.trapezoid(
#             np.multiply(1-reflectivity[:, np.newaxis, np.newaxis], URdict["spectral_power_3D"]), 
#             URdict["spectrum"]["energy"], axis=0
#             )*np.sin(theta/1000)
#     else:
#         projPowDenSR = trim_and_resample_power_density(PowDenSR,
#                                                        dx=URdict["axis"]["x"][-1]-URdict["axis"]["x"][0], 
#                                                        dy=URdict["axis"]["y"][-1]-URdict["axis"]["y"][0],
#                                                        X=URdict["axis"]["x"],
#                                                        Y=URdict["axis"]["y"])
#         powerDict["projected_power"]["map"] = projPowDenSR["power_density"]["map"]*np.sin(theta/1000)
#         powerDict["absorbed_power"]["map"] = projPowDenSR["power_density"]["map"]*np.sin(theta/1000) - powerDict["reflected_power"]["map"]
#     powerDict["absorbed_power"]["CumPow"] = powerDict["absorbed_power"]["map"].sum()*dx*dy
#     powerDict["absorbed_power"]["PowDenSRmax"] = powerDict["absorbed_power"]["map"].max()

#     powerDict["projected_power"]["CumPow"] = powerDict["projected_power"]["map"].sum()*dx*dy
#     powerDict["projected_power"]["PowDenSRmax"] = powerDict["projected_power"]["map"].max()

#     return powerDict


#***********************************************************************************
# transmitted and absorbed power
#***********************************************************************************
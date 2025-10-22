#!/bin/python

"""
This module provides interfacing functions for SYNED and barc4sr
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '26/JUN/2025'

import json

import numpy as np

#***********************************************************************************
# functions
#***********************************************************************************

def write_syned_file(json_file: str, light_source_name: str, ElectronBeamClass: object, 
                     MagneticStructureClass: object) -> None:
    """
    Writes a Python dictionary into a SYNED JSON configuration file.

    Parameters:
        json_file (str): The path to the JSON file where the dictionary will be written.
        light_source_name (str): The name of the light source.
        ElectronBeamClass (type): The class representing electron beam parameters.
        MagneticStructureClass (type): The class representing magnetic structure parameters.
    """

    data = {
        "CLASS_NAME": "LightSource",
        "name": light_source_name,
        "electron_beam": vars(ElectronBeamClass),
        "magnetic_structure": vars(MagneticStructureClass)
    }

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def read_syned_file(json_file: str) -> dict:
    """
    Reads a SYNED JSON configuration file and returns its contents as a dictionary.

    Parameters:
        json_file (str): The path to the SYNED JSON configuration file.

    Returns:
        dict: A dictionary containing the contents of the JSON file.
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def syned_dictionary(json_file: str, observation_point: float, 
                     hor_slit: float, ver_slit: float, hor_slit_cen: float, ver_slit_cen: float) -> dict:
    """
    Generate beamline parameters based on a SYNED JSON configuration file and additional input parameters.

    Args:
        json_file (str): Path to the SYNED JSON configuration file.
        observation_point (float): Distance to the observation point in meters.
        hor_slit (float): Horizontal slit size in meters.
        ver_slit (float): Vertical slit size in meters.
        hor_slit_cen (float): Horizontal slit center position in meters.
        ver_slit_cen (float): Vertical slit center position in meters.

    Returns:
        dict: A dictionary containing beamline parameters, including electron beam characteristics,
              magnetic structure details, and radiation observation settings.
    """

    data = read_syned_file(json_file)

    beamline = {}
    # accelerator
    beamline['ElectronEnergy'] = data["electron_beam"]["energy_in_GeV"]
    beamline['ElectronCurrent'] = data["electron_beam"]["current"]
    beamline['ElectronEnergySpread'] = data["electron_beam"]["energy_spread"]
    # electron beam
    beamline['ElectronBeamSizeH'] = np.sqrt(data["electron_beam"]["moment_xx"])
    beamline['ElectronBeamSizeV'] = np.sqrt(data["electron_beam"]["moment_yy"])
    beamline['ElectronBeamDivergenceH'] = np.sqrt(data["electron_beam"]["moment_xpxp"])
    beamline['ElectronBeamDivergenceV'] = np.sqrt(data["electron_beam"]["moment_ypyp"])
    # magnetic structure
    beamline['MagFieldCenter'] = 0
    # undulator        
    if data["magnetic_structure"]["CLASS_NAME"].startswith("U"):
        beamline['Class'] = 'u'
        beamline['NPeriods'] = data["magnetic_structure"]["number_of_periods"]
        beamline['PeriodID'] = data["magnetic_structure"]["period_length"]

        beamline['Kh'] = data["magnetic_structure"]["K_horizontal"]
        beamline['MagFieldPhaseH'] = data["magnetic_structure"]["B_horizontal_phase"]
        beamline['MagFieldSymmetryH'] = data["magnetic_structure"]["B_horizontal_symmetry"]

        beamline['Kv'] = data["magnetic_structure"]["K_vertical"]
        beamline['MagFieldPhaseV'] = data["magnetic_structure"]["B_vertical_phase"]
        beamline['MagFieldSymmetryV'] = data["magnetic_structure"]["B_vertical_symmetry"]
    # bending magnet        
    if data["magnetic_structure"]["CLASS_NAME"].startswith("B"):
        beamline['Class'] = 'bm'
        beamline['Bh'] = data["magnetic_structure"]["B_horizontal"]
        beamline['Bv'] = data["magnetic_structure"]["B_vertical"]
        beamline['B'] = data["magnetic_structure"]["magnetic_field"]
        beamline['R'] = data["magnetic_structure"]["radius"]
        beamline['Leff'] = data["magnetic_structure"]["length"]
        beamline['Ledge'] = data["magnetic_structure"]["length_edge"]
        beamline['ExtAng'] = data["magnetic_structure"]["extraction_angle"]
    # radiation observation
    beamline['distance'] = observation_point
    beamline['slitH'] = hor_slit
    beamline['slitV'] = ver_slit
    beamline['slitHcenter'] = hor_slit_cen
    beamline['slitVcenter'] = ver_slit_cen
  
    return beamline


def barc4sr_dictionary(light_source: object, observation_point: float,
                       hor_slit: float, ver_slit: float, hor_slit_cen: float, ver_slit_cen: float) -> dict:
    """
    Generate beamline parameters based on a SYNED JSON configuration file and additional input parameters.

    Args:
        light_source (SynchrotronSource): Instance of SynchrotronSource or any object that inherits from it.
        observation_point (float): Distance to the observation point in meters.
        hor_slit (float): Horizontal slit size in meters.
        ver_slit (float): Vertical slit size in meters.
        hor_slit_cen (float): Horizontal slit center position in meters.
        ver_slit_cen (float): Vertical slit center position in meters.

    Returns:
        dict: A dictionary containing beamline parameters, including electron beam characteristics,
              magnetic structure details, and radiation observation settings.
    """

    beamline = {}
    # accelerator
    beamline['ElectronEnergy'] = light_source.ElectronBeam.energy_in_GeV
    beamline['ElectronCurrent'] = light_source.ElectronBeam.current
    beamline['ElectronEnergySpread'] = light_source.ElectronBeam.energy_spread
    # electron beam
    beamline['ElectronBeamSizeH'] = np.sqrt(light_source.ElectronBeam.moment_xx)
    beamline['ElectronBeamSizeV'] = np.sqrt(light_source.ElectronBeam.moment_yy)
    beamline['ElectronBeamDivergenceH'] = np.sqrt(light_source.ElectronBeam.moment_xpxp)
    beamline['ElectronBeamDivergenceV'] = np.sqrt(light_source.ElectronBeam.moment_ypyp)
    # magnetic structure
    beamline['MagFieldCenter'] = light_source.MagneticStructure.center
    # undulator        
    if light_source.MagneticStructure.CLASS_NAME.startswith("U"):
        beamline['Class'] = 'u'
        beamline['NPeriods'] = light_source.MagneticStructure.number_of_periods
        beamline['PeriodID'] = light_source.MagneticStructure.period_length

        beamline['Kh'] = light_source.MagneticStructure.K_horizontal
        beamline['MagFieldPhaseH'] = light_source.MagneticStructure.B_horizontal_phase
        beamline['MagFieldSymmetryH'] = light_source.MagneticStructure.B_horizontal_symmetry

        beamline['Kv'] = light_source.MagneticStructure.K_vertical
        beamline['MagFieldPhaseV'] = light_source.MagneticStructure.B_vertical_phase
        beamline['MagFieldSymmetryV'] = light_source.MagneticStructure.B_vertical_symmetry
    # bending magnet        
    if light_source.MagneticStructure.CLASS_NAME.startswith("B"):
        beamline['Class'] = 'bm'
        beamline['Bh'] = None
        beamline['Bv'] = None
        beamline['B'] = light_source.MagneticStructure.magnetic_field
        beamline['R'] = light_source.MagneticStructure.radius
        beamline['Leff'] = light_source.MagneticStructure.field_length
        beamline['Ledge'] = light_source.MagneticStructure.edge_length
        beamline['ExtAng'] = light_source.MagneticStructure.extraction_angle
    # bending magnet        
    if light_source.MagneticStructure.CLASS_NAME.startswith("A"):
        beamline['Class'] = 'arb'
        beamline['MagFieldDict'] = light_source.MagneticStructure.magnetic_field

    # radiation observation
    beamline['distance'] = observation_point
    beamline['slitH'] = hor_slit
    beamline['slitV'] = ver_slit
    beamline['slitHcenter'] = hor_slit_cen
    beamline['slitVcenter'] = ver_slit_cen
  
    return beamline

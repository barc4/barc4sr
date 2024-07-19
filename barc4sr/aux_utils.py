#!/bin/python

""" 
This module provides SR classes, SRW interfaced functions, r/w SYNED compatible functions,
r/w functions for the electron trajectory and magnetic field as well as other auxiliary 
functions.
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '15/MAR/2024'
__changed__ = '19/JUL/2024'

import array
import copy
import json
import multiprocessing as mp
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import physical_constants
from skimage.restoration import unwrap_phase

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]
MASS = physical_constants["electron mass"][0]
PI = np.pi
RMS = np.sqrt(2)/2

#***********************************************************************************
# SYNED/SRW interface functions
#***********************************************************************************

class ElectronBeam(object):
    """
    Class for entering the electron beam parameters - this is based on SRWLPartBeam class
    """
    def __init__(self, energy: float = None, energy_spread: float = None, current: float = None,
                 number_of_bunches: int = 1, moment_xx: float = None, moment_xxp: float = None,
                 moment_xpxp: float = None, moment_yy: float = None, moment_yyp: float = None,
                 moment_ypyp: float = None) -> None:
        """
        Initializes an instance of the ElectronBeam class.

        Args:
            energy (float): Energy of the electron beam in GeV. 
            energy_spread (float): RMS energy spread of the electron beam. 
            current (float): Average current of the electron beam in Amperes. 
            number_of_bunches (int): Number of bunches in the electron beam. 
            moment_xx (float): Second order moment: <(x-<x>)^2>. 
            moment_xxp (float): Second order moment: <(x-<x>)(x'-<x'>)>. 
            moment_xpxp (float): Second order moment: <(x'-<x'>)^2>. 
            moment_yy (float): Second order moment: <(y-<y>)^2>. 
            moment_yyp (float): Second order moment: <(y-<y>)(y'-<y'>)>. 
            moment_ypyp (float): Second order moment: <(y'-<y'>)^2>. 
        """  
        self.CLASS_NAME = "ElectronBeam"
        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current
        self.number_of_bunches = number_of_bunches
        self.moment_xx = moment_xx
        self.moment_xxp = moment_xxp
        self.moment_xpxp = moment_xpxp
        self.moment_yy = moment_yy
        self.moment_yyp = moment_yyp
        self.moment_ypyp = moment_ypyp

    def from_twiss(self, energy: float, energy_spread: float, current: float, emittance: float,
                   coupling: float, emittance_x: float, beta_x: float, alpha_x: float, eta_x: float,
                   etap_x: float, emittance_y: float, beta_y: float, alpha_y: float, eta_y: float,
                   etap_y: float) -> None:
        """
        Sets up electron beam internal data from Twiss parameters.

        Args:
            energy (float): Energy of the electron beam in GeV.
            energy_spread (float): RMS energy spread of the electron beam.
            current (float): Average current of the electron beam in Amperes.
            emittance (float): Emittance of the electron beam.
            coupling (float): Coupling coefficient between horizontal and vertical emittances.
            emittance_x (float): Horizontal emittance in meters.
            beta_x (float): Horizontal beta-function in meters.
            alpha_x (float): Horizontal alpha-function in radians.
            eta_x (float): Horizontal dispersion function in meters.
            etap_x (float): Horizontal dispersion function derivative in radians.
            emittance_y (float): Vertical emittance in meters.
            beta_y (float): Vertical beta-function in meters.
            alpha_y (float): Vertical alpha-function in radians.
            eta_y (float): Vertical dispersion function in meters.
            etap_y (float): Vertical dispersion function derivative in radians.
        """
        if emittance_x is None:
            emittance_x = emittance*(1/(coupling+1))
        if emittance_y is None:
            emittance_y = emittance*(coupling/(coupling+1))
            
        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current
        sigE2 = energy_spread**2
        # <(x-<x>)^2>
        self.moment_xx = emittance_x*beta_x + sigE2*eta_x*eta_x 
        # <(x-<x>)(x'-<x'>)>          
        self.moment_xxp = -emittance_x*alpha_x + sigE2*eta_x*etap_x  
        # <(x'-<x'>)^2>      
        self.moment_xpxp = emittance_x*(1 + alpha_x*alpha_x)/beta_x + sigE2*etap_x*etap_x 
        #<(y-<y>)^2>    
        self.moment_yy = emittance_y*beta_y + sigE2*eta_y*eta_y 
        #<(y-<y>)(y'-<y'>)>          
        self.moment_yyp = -emittance_y*alpha_y + sigE2*eta_y*etap_y
        #<(y'-<y'>)^2>        
        self.moment_ypyp = emittance_y*(1 + alpha_y*alpha_y)/beta_y + sigE2*etap_y*etap_y     

        self.to_rms()

    def from_rms(self, energy: float, energy_spread: float, current: float, x: float, xp: float,
                 y: float, yp: float, xxp: float = 0, yyp: float = 0) -> None:
        """
        Sets up electron beam internal data from RMS values.

        Args:
            energy (float): Energy of the electron beam in GeV.
            energy_spread (float): RMS energy spread of the electron beam.
            current (float): Average current of the electron beam in Amperes.
            x (float): Horizontal RMS size in meters.
            xp (float): Horizontal RMS divergence in radians.
            y (float): Vertical RMS size in meters.
            yp (float): Vertical RMS divergence in radians.
            xxp (float): Cross-correlation term between x and xp in meters. Defaults to 0.
            yyp (float): Cross-correlation term between y and yp in meters. Defaults to 0.
        """

        self.energy_in_GeV = energy
        self.energy_spread = energy_spread
        self.current = current

        self.moment_xx = x*x        # <(x-<x>)^2>
        self.moment_xxp = xxp       # <(x-<x>)(x'-<x'>)>          
        self.moment_xpxp = xp*xp    # <(x'-<x'>)^2>      
        self.moment_yy = y*y        # <(y-<y>)^2>    
        self.moment_yyp = yyp       # <(y-<y>)(y'-<y'>)>          
        self.moment_ypyp = yp*yp    # <(y'-<y'>)^2>        

        self.to_rms()

    def propagate(self, dist: float) -> None:
        """
        Propagates electron beam statistical moments over a distance in free space.

        Args:
            dist (float): Distance the beam has to be propagated over in meters.
        """
        self.moment_xx  += (self.moment_xxp + self.moment_xpxp)*dist**2
        self.moment_xxp += (self.moment_xpxp)*dist
        self.moment_yy  += (self.moment_yyp + self.moment_ypyp)*dist**2
        self.moment_yyp += (self.moment_ypyp)*dist

        self.to_rms()

    def to_rms(self) -> None:
        """
        Computes the RMS sizes and divergences of the electron beam based on its 
        second-order statistical moments. The calculated values are stored back in the 
        object attributes (x, y, xp, yp).
        """
        self.x = np.sqrt(self.moment_xx)
        self.y = np.sqrt(self.moment_yy)
        self.xp = np.sqrt(self.moment_xpxp)
        self.yp = np.sqrt(self.moment_ypyp)

    def print_rms(self) -> None:
        """
        Prints electron beam rms sizes and divergences 
        """
        print(f"electron beam:\n\
              >> x/xp = {self.x*1e6:0.2f} um vs. {self.xp*1e6:0.2f} urad\n\
              >> y/yp = {self.y*1e6:0.2f} um vs. {self.yp*1e6:0.2f} urad")
        
    def get_attributes(self) -> None:
        """
        Prints all attribute of object
        """

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))

class MagneticStructure(object):
    """
    Class for entering the magnetic structure parameters, which can represent an undulator, wiggler, or a bending magnet.
    """
    def __init__(self, K_vertical: float = None, K_horizontal: float = None, 
                 B_horizontal: float = None, B_vertical: float = None, 
                 period_length: float = None, number_of_periods: int = None, 
                 radius: float = None, length: float = None, length_edge: float = 0, 
                 mag_structure: str = "u") -> None:
        """
        Initializes an instance of the MagneticStructure class.

            Args:
            K_horizontal (float): Horizontal magnetic parameter (K-value) of the undulator.
            K_vertical (float): Vertical magnetic parameter (K-value) of the undulator.
            B_horizontal (float): Horizontal magnetic field component.
            B_vertical (float): Vertical magnetic field component.
            period_length (float): Length of one period of the undulator in meters.
            number_of_periods (int): Number of periods of the undulator.
            radius (float): Bending magnet: radius of curvature of the central trajectory in meters (effective if > 0).
            length (float): Bending magnet: Effective length in meters (effective if > 0).
            length_edge (float): Bending magnet: "soft" edge length for field variation from 10% to 90% in meters; G/(1 + ((z-zc)/d)^2)^2 fringe field dependence is assumed.
            mag_structure (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm). Defaults to "u".
        """

        if not mag_structure.startswith(('u', 'w', 'bm')):
            raise ValueError("Not a valid magnetic structure")
        
        self.B_horizontal = B_horizontal
        self.B_vertical = B_vertical
        
        if mag_structure.startswith('u') or mag_structure.startswith('w'):
            self.K_vertical = K_vertical
            self.K_horizontal = K_horizontal
            self.period_length = period_length
            self.number_of_periods = number_of_periods
            if self.B_horizontal is not None or self.B_vertical is not None:
                self.set_magnetic_field(self.B_horizontal, self.B_vertical)

        if mag_structure.startswith('u'):
            self.CLASS_NAME = "Undulator"
        if mag_structure.startswith('w'):
            self.CLASS_NAME = "Wiggler"
        if mag_structure.startswith('bm'):
            self.CLASS_NAME = "BendingMagnet"
            self.magnetic_field = self.B_vertical
            self.radius = radius
            self.length = length
            self.length_edge = length_edge
    # -------------------------------------        
    # undulator auxiliary functions
    # -------------------------------------        
    def set_resonant_energy(self, energy: float, harmonic: int, eBeamEnergy: float, direction: str) -> None:
        """
        Sets the K-value based on the resonant energy and harmonic.

        Args:
            energy (float): Resonant energy in electron volts (eV).
            harmonic (int): Harmonic number.
            eBeamEnergy (float): Energy of the electron beam in GeV.
            direction (str): Direction of the undulator ('v' for vertical, 'h' for horizontal, 'b' for both).

        """
        if self.CLASS_NAME.startswith(('B', 'W')):
            raise ValueError("invalid operation for this synchrotron radiation source")
        else:
            wavelength = energy_wavelength(energy, 'eV')
            gamma = get_gamma(eBeamEnergy)
            K = np.sqrt(2)*np.sqrt(((2 * harmonic * wavelength * gamma ** 2)/self.period_length)-1)

            if "v" in direction:
                self.K_vertical = K
            elif "h" in direction:
                self.K_horizontal = K
            else:
                self.K_vertical = np.sqrt(K/2)
                self.K_horizontal = np.sqrt(K/2)

    def print_resonant_energy(self,  K: float, harmonic: int, eBeamEnergy: float) -> None:
        """
        Prints the resonant energy based on the provided K-value, harmonic number, and electron beam energy.

        Args:
            K (float): The K-value of the undulator.
            harmonic (int): The harmonic number.
            eBeamEnergy (float): Energy of the electron beam in GeV.
        """
        if self.CLASS_NAME.startswith(('B', 'W')):
            raise ValueError("invalid operation for this synchrotron radiation source")
        else:
            gamma = get_gamma(eBeamEnergy)
            wavelength = self.period_length/(2 * harmonic * gamma ** 2)*(1+(K**2)/2) 
            energy = energy_wavelength(wavelength, 'm')

            print(f">> resonant energy {energy:.2f} eV")
    # -------------------------------------        
    # undulator/wiggler auxiliary functions
    # -------------------------------------        
    def set_magnetic_field(self, B_horizontal: float=None, B_vertical: float=None) -> None:
        """
        Sets the K-value based on the magnetic field strength.

        Args:
            Bx (float): Magnetic field strength in the horizontal direction.
            By (float): Magnetic field strength in the vertical direction.
        """
        if self.CLASS_NAME.startswith('B'):
            raise ValueError("invalid operation for bending magnet")
        else:
            if B_horizontal is not None:
                self.Kx = CHARGE * B_horizontal * self.period_length / (2 * PI * MASS * LIGHT)
            if B_vertical is not None:
                self.Kx = CHARGE * B_vertical * self.period_length / (2 * PI * MASS * LIGHT)
    # -------------------------------------        
    # bending magnet auxiliary functions
    # -------------------------------------        
    def set_bending_magnet_radius(self, eBeamEnergy: float) -> None:
        """
        Sets the radius of curvature from the magnetic field for a bending magnet based 
        on the electron beam energy.

        Args:
            eBeamEnergy (float): Energy of the electron beam in GeV.
        """
        if self.CLASS_NAME.startswith('B') is False:
            raise ValueError("invalid operation for undulator or wiggler sources")
        else:
            gamma = get_gamma(eBeamEnergy)
            e_speed = LIGHT * np.sqrt(1-1/gamma)
            self.radius = gamma * MASS * e_speed /(CHARGE * self.magnetic_field)
            print(f">> bending radius {self.radius:.3f} m")

    def set_bending_magnet_field(self, eBeamEnergy: float) -> None:
        """
        Sets the magnetic field from the radius of curvature for a bending magnet based 
        on the electron beam energy.

        Args:
            eBeamEnergy (float): Energy of the electron beam in GeV.
        """
        if self.CLASS_NAME.startswith('B') is False:
            raise ValueError("invalid operation for undulator or wiggler sources")
        else:
            gamma = get_gamma(eBeamEnergy)
            e_speed = LIGHT * np.sqrt(1-1/gamma)
            self.magnetic_field = gamma * MASS * e_speed /(CHARGE * self.radius)
            print(f">> magnetic field {self.magnetic_field:.3f} T")

    def print_critical_energy(self, eBeamEnergy: float) -> None:
        """
        Prints the critical energy for a bending magnet based on the electron beam energy

        Args:
            eBeamEnergy (float): Energy of the electron beam in GeV.
        """
        if self.CLASS_NAME.startswith('B') is False:
            raise ValueError("invalid operation for undulator or wiggler sources")
        else:
            gamma = get_gamma(eBeamEnergy)
            energy = (3*PLANCK*self.magnetic_field*gamma**2)/(4*PI*MASS)
            print(f">> critical energy {energy:.2f} eV")
    # -------------------------------------        
    # auxiliary functions
    # ------------------------------------- 
    def get_attributes(self) -> None:
        """
        Prints all attribute of object
        """

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))

class SynchrotronSource(object):
    """
    Class representing a synchrotron radiation source, which combines an electron beam and a magnetic structure.
    """
    def __init__(self, electron_beam: ElectronBeam, magnetic_structure: MagneticStructure) -> None:
        """
        Initializes an instance of the SynchrotronSource class.

        Args:
            electron_beam (ElectronBeam): An instance of the ElectronBeam class representing the electron beam parameters.
            magnetic_structure (MagneticStructure): An instance of the MagneticStructure class representing the magnetic structure parameters.
        """
        self.ElectronBeam = electron_beam
        self.MagneticStructure = magnetic_structure
    
    def __getattr__(self, name):
        """
        Retrieves an attribute from either the ElectronBeam or MagneticStructure instances if it exists.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            The value of the attribute from either the ElectronBeam or MagneticStructure instance.

        """
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.ElectronBeam, name):
            return getattr(self.ElectronBeam, name)
        elif hasattr(self.MagneticStructure, name):
            return getattr(self.MagneticStructure, name)
        else:
            raise AttributeError(f"'SynchrotronSource' object has no attribute '{name}'")
        
        
    def write_syned_config(self, json_file: str, light_source_name: str=None):
        """
        Writes a SYNED JSON configuration file.

        Parameters:
            json_file (str): The path to the JSON file where the dictionary will be written.
            light_source_name (str): The name of the light source.
        """
        if light_source_name is None:
            light_source_name = json_file.split('/')[-1].replace('.json','')

        write_syned_file(json_file, light_source_name, self.ElectronBeam, self.MagneticStructure)

class UndulatorSource(object):
    """
    Class representing an undulator radiation source, which combines an electron beam and a magnetic structure.
    """
    def __init__(self, electron_beam: ElectronBeam, magnetic_structure: MagneticStructure) -> None:
        """
        Initializes an instance of the UndulatorSource class.

        Args:
            electron_beam (ElectronBeam): An instance of the ElectronBeam class representing the electron beam parameters.
            magnetic_structure (MagneticStructure): An instance of the MagneticStructure class representing the magnetic structure parameters.
        """
        self.ElectronBeam = electron_beam
        self.MagneticStructure = magnetic_structure
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.ElectronBeam, name):
            return getattr(self.ElectronBeam, name)
        elif hasattr(self.MagneticStructure, name):
            return getattr(self.MagneticStructure, name)
        else:
            raise AttributeError(f"'UndulatorSource' object has no attribute '{name}'")
        
    def get_beam_dimensions(self, mth):

        # Tanaka - https://doi.org/10.1107/S0909049509009479

        # Boaz    - https://doi.org/10.1063/1.5084711

        # Convolution
        pass

    def print_beam_dimensions(self):
        # prints electron beam size
        # prints undulator natural size at resonance
        # prints source size at resonance
        pass

    def get_attributes(self) -> None:
        """
        Prints all attribute of object
        """

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))
        
#***********************************************************************************
# SRW interface functions
#***********************************************************************************

def set_light_source(file_name: str,
                     bl: dict,
                     filament_beam: bool,
                     energy_spread: bool,
                     electron_trajectory: bool,
                     id_type: str,
                     **kwargs) -> Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]:
    """
    Set up the light source parameters including electron beam, magnetic structure, and electron trajectory.

    Args:
        file_name (str): The name of the output file.
        bl (dict): Beamline parameters dictionary containing essential information for setup.
        filament_beam (bool): Flag indicating whether to set the beam emittance to zero.
        energy_spread (bool): Flag indicating whether to set the beam energy spread to zero.
        electron_trajectory (bool): Whether to calculate and save electron trajectory.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]
        magfield_initial_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (str): Path to the file containing magnetic measurement data.
        tabulated_undulator_mthd (int): Method to tabulate the undulator field.

    Returns:
        Tuple[srwlib.SRWLPartBeam, srwlib.SRWLMagFldC, srwlib.SRWLPrtTrj]: A tuple containing the electron beam,
        magnetic structure, and electron trajectory.
    """    

    ebeam_initial_position = kwargs.get('ebeam_initial_position', 0)
    magfield_central_position = kwargs.get('magfield_central_position', 0)
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)

    # ----------------------------------------------------------------------------------
    # definition of the electron beam
    # ----------------------------------------------------------------------------------
    print('> Generating the electron beam ... ', end='')
    eBeam = set_electron_beam(bl,
                              filament_beam,
                              energy_spread,
                              id_type,
                              initial_position=ebeam_initial_position)
    print('completed')
    # ----------------------------------------------------------------------------------
    # definition of magnetic structure
    # ----------------------------------------------------------------------------------
    print('> Generating the magnetic structure ... ', end='')
    magFldCnt = set_magnetic_structure(bl, 
                                       id_type,
                                       magnetic_measurement = magnetic_measurement, 
                                       magfield_central_position = magfield_central_position,
                                       tabulated_undulator_mthd = tabulated_undulator_mthd)
    print('completed')
    # ----------------------------------------------------------------------------------
    # calculate electron trajectory
    # ----------------------------------------------------------------------------------
    print('> Electron trajectory calculation ... ', end='')
    if electron_trajectory:
        electron_trajectory_file_name = file_name+"_eTraj.dat"
        eTraj = srwlCalcPartTraj(eBeam, magFldCnt)
        eTraj.save_ascii(electron_trajectory_file_name)
        print(f">>>{electron_trajectory_file_name}<<< ", end='')
    else:
        eTraj = 0
    print('completed')
    return eBeam, magFldCnt, eTraj


def set_electron_beam(bl: dict, 
                      filament_beam: bool, 
                      energy_spread: bool, 
                      id_type: str, 
                      **kwargs) -> srwlib.SRWLPartBeam:
    """
    Set up the electron beam parameters.

    Parameters:
        bl (dict): Dictionary containing beamline parameters.
        filament_beam (bool): Flag indicating whether to set the beam emittance to zero.
        energy_spread (bool): Flag indicating whether to set the beam energy spread to zero.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        ebeam_initial_position (float): Electron beam initial average longitudinal position [m]

    Returns:
        srwlib.SRWLPartBeam: Electron beam object initialized with specified parameters.

    """
    initial_position = kwargs.get('initial_position', 0)

    eBeam = srwlib.SRWLPartBeam()
    eBeam.Iavg = bl['ElectronCurrent']  # average current [A]
    eBeam.partStatMom1.x = 0.  # initial transverse positions [m]
    eBeam.partStatMom1.y = 0.
    if id_type.startswith('u'):
        eBeam.partStatMom1.z = - bl['PeriodID'] * (bl['NPeriods'] + 4) / 2  # initial longitudinal positions
    else:
        eBeam.partStatMom1.z = initial_position
    eBeam.partStatMom1.xp = 0  # initial relative transverse divergence [rad]
    eBeam.partStatMom1.yp = 0
    eBeam.partStatMom1.gamma = get_gamma(bl['ElectronEnergy'])
    if filament_beam:
        sigX = 1e-25
        sigXp = 1e-25
        sigY = 1e-25
        sigYp = 1e-25
        if energy_spread:
            sigEperE = bl['ElectronEnergySpread']
        else:
            sigEperE = 1e-25    
    else:
        sigX = bl['ElectronBeamSizeH']  # horizontal RMS size of e-beam [m]
        sigXp = bl['ElectronBeamDivergenceH']  # horizontal RMS angular divergence [rad]
        sigY = bl['ElectronBeamSizeV']  # vertical RMS size of e-beam [m]
        sigYp = bl['ElectronBeamDivergenceV']  # vertical RMS angular divergence [rad]
        if energy_spread:
            sigEperE = bl['ElectronEnergySpread']
        else:
            sigEperE = 1e-25    

    # 2nd order stat. moments:
    eBeam.arStatMom2[0] = sigX * sigX  # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0  # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp * sigXp  # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY * sigY  # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0  # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp * sigYp  # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE * sigEperE  # <(E-<E>)^2>/<E>^2

    return eBeam


def set_magnetic_structure(bl: dict, 
                           id_type: str, 
                           **kwargs) -> srwlib.SRWLMagFldC:
    """
    Sets up the magnetic field container.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).

    Optional Args (kwargs):
        magfield_central_position (float): Longitudinal position of the magnet center [m]
        magnetic_measurement (str): Path to the tabulated magnetic field data.
        tabulated_undulator_mthd (int): Method to use for generating undulator field if magnetic_measurement is provided. Defaults to 0

    Returns:
        srwlib.SRWLMagFldC: Magnetic field container.

    """
    magnetic_measurement = kwargs.get('magnetic_measurement', None)
    magfield_central_position = kwargs.get('magfield_central_position', 0)

    if id_type.startswith('u'):
        tabulated_undulator_mthd = kwargs.get('tabulated_undulator_mthd', 0)
        if magnetic_measurement is None:    # ideal sinusoidal undulator magnetic structure
            und = srwlib.SRWLMagFldU()
            und.set_sin(_per=bl["PeriodID"],
                        _len=bl['PeriodID']*bl['NPeriods'], 
                        _bx=bl['Kh']*2*PI*MASS*LIGHT/(CHARGE*bl["PeriodID"]), 
                        _by=bl['Kv']*2*PI*MASS*LIGHT/(CHARGE*bl["PeriodID"]), 
                        _phx=bl['KhPhase'], 
                        _phy=bl['KvPhase'], 
                        _sx=bl['MagFieldSymmetryH'], 
                        _sy=bl['MagFieldSymmetryV'])

            magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[und],
                                            _arXc=srwlib.array('d', [0.0]),
                                            _arYc=srwlib.array('d', [0.0]),
                                            _arZc=srwlib.array('d', [magfield_central_position]))
            
        else:    # tabulated magnetic field
            magFldCnt = srwlib.srwl_uti_read_mag_fld_3d(magnetic_measurement, _scom='#')
            print(" tabulated magnetic field ... ", end="")
            if tabulated_undulator_mthd  != 0:   # similar to srwl_bl.set_und_per_from_tab()
                # TODO: parametrise
                """Setup periodic Magnetic Field from Tabulated one
                :param _rel_ac_thr: relative accuracy threshold
                :param _max_nh: max. number of harmonics to create
                :param _max_per: max. period length to consider
                """
                _rel_ac_thr=0.05
                _max_nh=50
                _max_per=0.1
                arHarm = []
                for i in range(_max_nh): 
                    arHarm.append(srwlib.SRWLMagFldH())
                magFldCntHarm = srwlib.SRWLMagFldC(srwlib.SRWLMagFldU(arHarm))
                srwlib.srwl.UtiUndFromMagFldTab(magFldCntHarm, magFldCnt, [_rel_ac_thr, _max_nh, _max_per])
                return magFldCntHarm
            
    if id_type.startswith('bm'):

        bm = srwlib.SRWLMagFldM()
        bm.G = bl["Bv"]
        bm.m = 1         # multipole order: 1 for dipole, 2 for quadrupole, 3 for sextupole, 4 for octupole
        bm.n_or_s = 'n'  # normal ('n') or skew ('s')
        bm.Leff = bl["Leff"]
        bm.Ledge = bl["Ledge"]
        bm.R = bl["R"]

        magFldCnt = srwlib.SRWLMagFldC(_arMagFld=[bm],
                                       _arXc=srwlib.array('d', [0.0]),
                                       _arYc=srwlib.array('d', [0.0]),
                                       _arZc=srwlib.array('d', [magfield_central_position]))

    return magFldCnt


def srwlCalcPartTraj(eBeam:srwlib.SRWLPartBeam,
                     magFldCnt: srwlib.SRWLMagFldC,
                     number_points: int = 50000, 
                     ctst: float = 0, 
                     ctfi: float = 0) -> srwlib.SRWLPrtTrj:
    """
    Calculate the trajectory of an electron through a magnetic field.

    Args:
        eBeam (srwlib.SRWLPartBeam): Particle beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container representing the magnetic field.
        number_points (int, optional): Number of points for trajectory calculation. Defaults to 50000.
        ctst (float, optional): Initial time (ct) for trajectory calculation. Defaults to 0.
        ctfi (float, optional): Final time (ct) for trajectory calculation. Defaults to 0.

    Returns:
        srwlib.SRWLPrtTrj: Object containing the calculated trajectory.
    """
    partTraj = srwlib.SRWLPrtTrj()
    partTraj.partInitCond = eBeam.partStatMom1
    partTraj.allocate(number_points, True)
    partTraj.ctStart = ctst
    partTraj.ctEnd = ctfi

    arPrecPar = [1] 
    srwlib.srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)

    return partTraj


def srwlibCalcElecFieldSR(bl: dict, 
                          eBeam: srwlib.SRWLPartBeam, 
                          magFldCnt: srwlib.SRWLMagFldC, 
                          energy_array: np.ndarray,
                          h_slit_points: int, 
                          v_slit_points: int, 
                          radiation_characteristic: int, 
                          radiation_dependence: int, 
                          radiation_polarisation: int,
                          id_type: str,
                          parallel: bool,
                          num_cores: int=None) -> np.ndarray:
    """
    Calculates the electric field for synchrotron radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_characteristic (int): Radiation characteristic:
               =0 -"Single-Electron" Intensity; 
               =1 -"Multi-Electron" Intensity; 
               =4 -"Single-Electron" Radiation Phase; 
               =5 -Re(E): Real part of Single-Electron Electric Field;
               =6 -Im(E): Imaginary part of Single-Electron Electric Field
        radiation_dependence (int): Radiation dependence (e.g., 1 for angular distribution).
               =0 -vs e (photon energy or time);
               =1 -vs x (horizontal position or angle);
               =2 -vs y (vertical position or angle);
               =3 -vs x&y (horizontal and vertical positions or angles);
               =4 -vs e&x (photon energy or time and horizontal position or angle);
               =5 -vs e&y (photon energy or time and vertical position or angle);
               =6 -vs e&x&y (photon energy or time, horizontal and vertical positions or angles);
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                            it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data, horizontal and vertical axes
    """
    
    arPrecPar = [0]*7
    if id_type.startswith('bm'):
        arPrecPar[0] = 2      # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    else:
        arPrecPar[0] = 1
    arPrecPar[1] = 0.01  
    arPrecPar[2] = 0     # longitudinal position to start integration (effective if < zEndInteg)
    arPrecPar[3] = 0     # longitudinal position to finish integration (effective if > zStartInteg)
    arPrecPar[4] = 50000 # Number of points for trajectory calculation
    arPrecPar[5] = 1     # Use "terminating terms"  or not (1 or 0 respectively)
    arPrecPar[6] = 0     # sampling factor for adjusting nx, ny (effective if > 0)

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        dE = np.diff(energy_array)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy_array[-1]>200*energy_array[0])

        if np.allclose(dE1, dE2) and wiggler_regime:
            chunk_size = 20
            n_slices = len(energy_array)

            chunks = [(energy_array[i:i + chunk_size],
                    bl, 
                    eBeam,
                    magFldCnt, 
                    arPrecPar, 
                    h_slit_points, 
                    v_slit_points, 
                    radiation_characteristic, 
                    radiation_dependence,
                    radiation_polarisation,
                    parallel) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(core_srwlibCalcElecFieldSR, chunks)
        else:
            dE = (energy_array[-1] - energy_array[0]) / num_cores
            energy_chunks = []

            for i in range(num_cores):
                bffr = copy.copy(energy_array)                
                bffr = np.delete(bffr, bffr < dE * (i) + energy_array[0])
                if i + 1 != num_cores:
                    bffr = np.delete(bffr, bffr >= dE * (i + 1) + energy_array[0])
                energy_chunks.append(bffr)

            results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcElecFieldSR)((
                                                                        list_pairs,
                                                                        bl,
                                                                        eBeam,
                                                                        magFldCnt,
                                                                        arPrecPar,
                                                                        h_slit_points,
                                                                        v_slit_points,
                                                                        radiation_characteristic,
                                                                        radiation_dependence,
                                                                        radiation_polarisation,
                                                                        parallel))
                                                for list_pairs in energy_chunks)
            
        for i, (intensity_chunck, h_chunck, v_chunck, e_chunck, t_chunck) in enumerate(results):
            if i == 0:
                intensity = intensity_chunck
                energy_array = np.asarray([e_chunck[0]])
                energy_chunks = np.asarray([len(e_chunck)])
                time_array = np.asarray([t_chunck])
            else:
                intensity = np.concatenate((intensity, intensity_chunck), axis=0)
                energy_array = np.concatenate((energy_array, np.asarray([e_chunck[0]])))
                energy_chunks = np.concatenate((energy_chunks, np.asarray([len(e_chunck)])))
                time_array = np.concatenate((time_array, np.asarray([t_chunck])))

        if not wiggler_regime:
            print(">>> ellapse time:")
            for ptime in range(len(time_array)):
                print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")

    else:
        results = core_srwlibCalcElecFieldSR((energy_array,
                                             bl, 
                                             eBeam,
                                             magFldCnt, 
                                             arPrecPar, 
                                             h_slit_points, 
                                             v_slit_points, 
                                             radiation_characteristic, 
                                             radiation_dependence,
                                             radiation_polarisation,
                                             parallel))
        intensity = results[0]

    if h_slit_points == 1 or v_slit_points == 1:
        x_axis = np.asarray([0])
        y_axis = np.asarray([0])
    else:
        x_axis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        y_axis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    return intensity, x_axis, y_axis


def core_srwlibCalcElecFieldSR(args: Tuple[np.ndarray, 
                                           dict, 
                                           srwlib.SRWLPartBeam, 
                                           srwlib.SRWLMagFldC, 
                                           List[float], 
                                           int, int, int, int, int, bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Core function to calculate electric field for synchrotron radiation.

    Args:
        args (Tuple): Tuple containing the following elements:
            energy_array (np.ndarray): Array of photon energies [eV].
            bl (dict): Dictionary containing beamline parameters.
            eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            arPrecPar (List[float]): Array of parameters for SR calculation.
            h_slit_points (int): Number of horizontal slit points.
            v_slit_points (int): Number of vertical slit points.
            rad_characteristic (int): Radiation characteristic (e.g., 0 for intensity).
            rad_dependence (int): Radiation dependence (e.g., 1 for angular distribution).
            radiation_polarisation (int): Polarisation component to be extracted.
            parallel (bool): Whether to use parallel computation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Tuple containing intensity data, 
                                                          horizontal axis, vertical axis, 
                                                          and computation time.
    """

    energy_array, bl, eBeam, magFldCnt, arPrecPar,  h_slit_points, v_slit_points, \
        rad_characteristic, rad_dependence, rad_polarisation, parallel = args
    
    tzero = time()

    _inPol = rad_polarisation
    _inIntType = rad_characteristic
    _inDepType = rad_dependence

    monochromatic = False
    if isinstance(energy_array, int) or isinstance(energy_array, float):
        monochromatic = True 

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
        _inDepType = 0
        intensity = np.zeros((energy_array.size))
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)
        _inDepType = 3
        if monochromatic:
            intensity =  np.zeros((vAxis.size, hAxis.size))
        else:
            intensity = np.zeros((energy_array.size, vAxis.size, hAxis.size))

    if parallel:    
        # this is rather convinient for step by step calculations and less memory intensive
        for ie in range(energy_array.size):
            try:
                mesh = srwlib.SRWLRadMesh(energy_array[ie], energy_array[ie], 1,
                                         hAxis[0], hAxis[-1], h_slit_points,
                                         vAxis[0], vAxis[-1], v_slit_points, 
                                         bl['distance'])

                wfr = srwlib.SRWLWfr()
                wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
                wfr.mesh = mesh
                wfr.partBeam = eBeam

                srwlib.srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)
                if _inIntType == 4:
                    arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                else:
                    arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)

                srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
                if _inDepType == 0:    # 0 -vs e (photon energy or time);
                    intensity[ie] = np.asarray(arI1, dtype="float64")
                else:
                    # data = np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                    data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) 
                    intensity[ie, :, :] = data
            except:
                 raise ValueError("Error running SRW.")
    else:
        try:
            if monochromatic:
                ei = ef = energy_array
                nf = 1
            else:
                ei = energy_array[0]
                ef = energy_array[-1]
                nf = len(energy_array)

            mesh = srwlib.SRWLRadMesh(ei, ef, nf,
                                      hAxis[0], hAxis[-1], h_slit_points,
                                      vAxis[0], vAxis[-1], v_slit_points, 
                                      bl['distance'])
            
            wfr = srwlib.SRWLWfr()
            wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
            wfr.mesh = mesh
            wfr.partBeam = eBeam

            # srwl_bl.calc_sr_se sets eTraj=0 despite having measured magnetic field
            srwlib.srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)

            if _inDepType == 0:    # 0 -vs e (photon energy or time);
                arI1 = array.array('f', [0]*wfr.mesh.ne)
                srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, wfr.mesh.eStart, 0, 0)
                intensity = np.asarray(arI1, dtype="float64")
            else:
                if monochromatic:
                    if _inIntType == 4:
                        arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                    else:
                        arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
                    srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, ei, 0, 0)
                    intensity = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx))
                else:
                    for ie in range(len(energy_array)):
                        if _inIntType == 4:
                            arI1 = array.array('d', [0]*wfr.mesh.nx*wfr.mesh.ny)
                        else:
                            arI1 = array.array('f', [0]*wfr.mesh.nx*wfr.mesh.ny)
                        srwlib.srwl.CalcIntFromElecField(arI1, wfr, _inPol, _inIntType, _inDepType, energy_array[ie], 0, 0)
                        data = np.asarray(arI1, dtype="float64").reshape((wfr.mesh.ny, wfr.mesh.nx)) #np.ndarray(buffer=arI1, shape=(wfr.mesh.ny, wfr.mesh.nx),dtype=arI1.typecode)
                        intensity[ie, :, :] = data

        except:
             raise ValueError("Error running SRW.")

    return intensity, hAxis, vAxis, energy_array, time()-tzero


def srwlibsrwl_wfr_emit_prop_multi_e(bl: dict,
                                     eBeam: srwlib.SRWLPartBeam, 
                                     magFldCnt: srwlib.SRWLMagFldC, 
                                     energy_array: np.ndarray,
                                     h_slit_points: int, 
                                     v_slit_points: int, 
                                     radiation_polarisation: int,
                                     id_type: str,
                                     number_macro_electrons: int, 
                                     aux_file_name: str,
                                     parallel: bool,
                                     num_cores: int=None,
                                     srApprox: int=0):
    """
    Interface function to compute multi-electron emission and propagation through a beamline using SRW.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        h_slit_points (int): Number of horizontal slit points.
        v_slit_points (int): Number of vertical slit points.
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        id_type (str): Type of magnetic structure, can be undulator (u), wiggler (w), or bending magnet (bm).
        number_macro_electrons (int): Total number of macro-electrons.
        aux_file_name (str): Auxiliary file name for saving intermediate data.
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.
        srApprox (int): Approximation to be used at multi-electron integration: 
                0- none (i.e. do standard M-C integration over 5D phase space volume of e-beam), 
                1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance
    Returns:
        np.ndarray: Array containing intensity data.
    """
    nMacroElecAvgPerProc = 10   # number of macro-electrons / wavefront to average on worker processes
    nMacroElecSavePer = 100     # intermediate data saving periodicity (in macro-electrons)
    if id_type.startswith('bm'):
        srCalcMeth = 2          # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    else:
        srCalcMeth = 1

    srApprox = 0
    srCalcPrec = 0.01           # SR calculation rel. accuracy

    if h_slit_points == 1 or v_slit_points == 1:
        hAxis = np.asarray([0])
        vAxis = np.asarray([0])
    else:
        hAxis = np.linspace(-bl['slitH']/2-bl['slitHcenter'], bl['slitH']/2-bl['slitHcenter'], h_slit_points)
        vAxis = np.linspace(-bl['slitV']/2-bl['slitVcenter'], bl['slitV']/2-bl['slitVcenter'], v_slit_points)

    if num_cores is None:
        num_cores = mp.cpu_count()

    if parallel:
        dE = np.diff(energy_array)    
        dE1 = np.min(dE)
        dE2 = np.max(dE)

        wiggler_regime = bool(energy_array[-1]>200*energy_array[0])

        if np.allclose(dE1, dE2) and wiggler_regime:
            chunk_size = 20
            n_slices = len(energy_array)

            chunks = [(energy_array[i:i + chunk_size],
                        bl,
                        eBeam, 
                        magFldCnt, 
                        h_slit_points, 
                        v_slit_points, 
                        number_macro_electrons, 
                        aux_file_name+'_'+str(i),
                        srCalcMeth,
                        srCalcPrec,
                        srApprox,
                        radiation_polarisation,
                        nMacroElecAvgPerProc,
                        nMacroElecSavePer) for i in range(0, n_slices, chunk_size)]
            
            with mp.Pool() as pool:
                results = pool.map(core_srwlibsrwl_wfr_emit_prop_multi_e, chunks)
        else:
            dE = (energy_array[-1] - energy_array[0]) / num_cores
            energy_chunks = []

            for i in range(num_cores):
                bffr = copy.copy(energy_array)                
                bffr = np.delete(bffr, bffr < dE * (i) + energy_array[0])
                if i + 1 != num_cores:
                    bffr = np.delete(bffr, bffr >= dE * (i + 1) + energy_array[0])
                energy_chunks.append(bffr)

            results = Parallel(n_jobs=num_cores)(delayed(core_srwlibsrwl_wfr_emit_prop_multi_e)((
                                                                        list_pairs,
                                                                        bl,
                                                                        eBeam, 
                                                                        magFldCnt, 
                                                                        h_slit_points, 
                                                                        v_slit_points, 
                                                                        number_macro_electrons, 
                                                                        aux_file_name+'_'+str(list_pairs[0]),
                                                                        srCalcMeth,
                                                                        srCalcPrec,
                                                                        srApprox,
                                                                        radiation_polarisation,
                                                                        nMacroElecAvgPerProc,
                                                                        nMacroElecSavePer))
                                                for list_pairs in energy_chunks)

        for i, (intensity_chunck, e_chunck, t_chunck) in enumerate(results):
            if i == 0:
                intensity = intensity_chunck
                energy_chunck = np.asarray([e_chunck[0]])
                energy_chunks = np.asarray([len(e_chunck)])
                time_array = np.asarray([t_chunck])
            else:
                intensity = np.concatenate((intensity, intensity_chunck), axis=0)
                energy_chunck = np.concatenate((energy_chunck, np.asarray([e_chunck[0]])))
                energy_chunks = np.concatenate((energy_chunks, np.asarray([len(e_chunck)])))
                time_array = np.concatenate((time_array, np.asarray([t_chunck])))

        if not wiggler_regime:
            print(">>> ellapse time:")
            for ptime in range(len(time_array)):
                print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_chunck[ptime]:.1f} eV).")
    else:
        results = core_srwlibsrwl_wfr_emit_prop_multi_e((energy_array,
                                                        bl,
                                                        eBeam, 
                                                        magFldCnt, 
                                                        h_slit_points, 
                                                        v_slit_points, 
                                                        number_macro_electrons, 
                                                        aux_file_name,
                                                        srCalcMeth,
                                                        srCalcPrec,
                                                        srApprox,
                                                        radiation_polarisation,
                                                        nMacroElecAvgPerProc,
                                                        nMacroElecSavePer))
        intensity = np.asarray(results[0], dtype="float64")

    return intensity, hAxis, vAxis


def core_srwlibsrwl_wfr_emit_prop_multi_e(args: Tuple[np.ndarray,
                                                      dict, 
                                                      srwlib.SRWLPartBeam, 
                                                      srwlib.SRWLMagFldC, 
                                                      int, int, int, str, int, float,
                                                      int, int, int, int]) -> Tuple[np.ndarray, float]:
    """
    Core function for computing multi-electron emission and propagation through a beamline using SRW.

    Args:
        args (tuple): Tuple containing arguments:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - h_slit_points (int): Number of horizontal slit points.
            - v_slit_points (int): Number of vertical slit points.
            - number_macro_electrons (int): Total number of macro-electrons.
            - aux_file_name (str): Auxiliary file name for saving intermediate data.
            - srCalcMeth (int): SR calculation method.
            - srCalcPrec (float): SR calculation relative accuracy.
            - srApprox (int): Approximation to be used at multi-electron integration: 
                    0- none (i.e. do standard M-C integration over 5D phase space volume of e-beam), 
                    1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance
            - radiation_polarisation (int): Polarisation component to be extracted.
            - nMacroElecAvgPerProc (int): Number of macro-electrons / wavefront to average on worker processes.
            - nMacroElecSavePer (int): Intermediate data saving periodicity (in macro-electrons).

    Returns:
        tuple: A tuple containing intensity data array and the elapsed time.
    """

    energy_array, bl, eBeam, magFldCnt, h_slit_points, v_slit_points, \
        number_macro_electrons, aux_file_name, srCalcMeth, srCalcPrec, srApprox, radiation_polarisation,\
        nMacroElecAvgPerProc, nMacroElecSavePer = args
    
    tzero = time()

    try:    
        
        if isinstance(energy_array, int) or isinstance(energy_array, float):
            monochromatic = True 
            ei = ef = energy_array
            nf = 1
        else:
            monochromatic = False
            ei = energy_array[0]
            ef = energy_array[-1]
            nf = len(energy_array)

        mesh = srwlib.SRWLRadMesh(ei, 
                                  ef, 
                                  nf,
                                  bl['slitHcenter'] - bl['slitH']/2,
                                  bl['slitHcenter'] + bl['slitH']/2, 
                                  h_slit_points,
                                  bl['slitVcenter'] - bl['slitV']/2, 
                                  bl['slitVcenter'] - bl['slitV']/2, 
                                  v_slit_points,
                                  bl['distance'])

        MacroElecFileName = aux_file_name + '_'+ str(int(number_macro_electrons / 1000)).replace('.', 'p') +'k_ME_intensity.dat'

        stk = srwlib.srwl_wfr_emit_prop_multi_e(_e_beam = eBeam, 
                                                _mag = magFldCnt,
                                                _mesh = mesh,
                                                _sr_meth = srCalcMeth,
                                                _sr_rel_prec = srCalcPrec,
                                                _n_part_tot = number_macro_electrons,
                                                _n_part_avg_proc=nMacroElecAvgPerProc, 
                                                _n_save_per=nMacroElecSavePer,
                                                _file_path=MacroElecFileName, 
                                                _sr_samp_fact=-1, 
                                                _opt_bl=None,
                                                _char=0,
                                                _me_approx=srApprox)
    
        os.system('rm %s'% MacroElecFileName)
        me_intensity = np.asarray(stk.to_int(_pol=radiation_polarisation), dtype='float64')

        if h_slit_points != 1 or v_slit_points != 1:
            k = 0
            if monochromatic:
                data = np.zeros((v_slit_points, h_slit_points))
                for iy in range(v_slit_points):
                    for ix in range(h_slit_points):
                        data[iy, ix] = me_intensity[k]
                        k+=1
            else:
                data = np.zeros((len(energy_array), v_slit_points, h_slit_points))
                for iy in range(v_slit_points):
                    for ix in range(h_slit_points):
                        for ie in range(len(energy_array)):
                            data[ie, iy, ix] = me_intensity[k]
                            k+=1
            me_intensity = data

    except:
         raise ValueError("Error running SRW.")

    return (me_intensity, energy_array, time()-tzero)


def srwlibCalcStokesUR(bl: dict, 
                       eBeam: srwlib.SRWLPartBeam, 
                       magFldCnt: srwlib.SRWLMagFldC, 
                       energy_array: np.ndarray, 
                       resonant_energy: float, 
                       radiation_polarisation: int,
                       parallel: bool,
                       num_cores: int=None) -> np.ndarray:
    """
    Calculates the Stokes parameters for undulator radiation.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        eBeam (srwlib.SRWLPartBeam): Electron beam properties.
        magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
        energy_array (np.ndarray): Array of photon energies [eV].
        resonant_energy (float): Resonant energy [eV].
        radiation_polarisation (int): Polarisation component to be extracted.
               =0 -Linear Horizontal; 
               =1 -Linear Vertical; 
               =2 -Linear 45 degrees; 
               =3 -Linear 135 degrees; 
               =4 -Circular Right; 
               =5 -Circular Left; 
               =6 -Total
        parallel (bool): Whether to use parallel computation.
        num_cores (int, optional): Number of CPU cores to use for parallel computation. If not specified, 
                                   it defaults to the number of available CPU cores.

    Returns:
        np.ndarray: Array containing intensity data.
    """
    
    if parallel:
        if num_cores is None:
            num_cores = mp.cpu_count()

        energy_chunks = np.array_split(list(energy_array), num_cores)

        results = Parallel(n_jobs=num_cores)(delayed(core_srwlibCalcStokesUR)((
                                                                    list_pairs,
                                                                    bl,
                                                                    eBeam,
                                                                    magFldCnt,
                                                                    resonant_energy,
                                                                    radiation_polarisation))
                                             for list_pairs in energy_chunks)
        energy_array = []
        time_array = []
        energy_chunks = []

        k = 0
        for calcs in results:
            energy_array.append(calcs[1][0])
            time_array.append(calcs[2])
            energy_chunks.append(len(calcs[0]))
            if k == 0:
                intensity = np.asarray(calcs[0], dtype="float64")
            else:
                intensity = np.concatenate((intensity, np.asarray(calcs[0], dtype="float64")), axis=0)
            k+=1
        print(">>> ellapse time:")

        for ptime in range(len(time_array)):
            print(f" Core {ptime+1}: {time_array[ptime]:.2f} s for {energy_chunks[ptime]} pts (E0 = {energy_array[ptime]:.1f} eV).")

    else:
        results = core_srwlibCalcStokesUR((energy_array,
                                          bl, 
                                          eBeam,
                                          magFldCnt, 
                                          resonant_energy,
                                          radiation_polarisation))
        
        intensity = np.asarray(results[0], dtype="float64")

    return intensity


def core_srwlibCalcStokesUR(args: Tuple[ np.ndarray, 
                                        dict, 
                                        srwlib.SRWLPartBeam, 
                                        srwlib.SRWLMagFldC, 
                                        float,
                                        int]) -> Tuple[np.ndarray, float]:
    """
    Core function to calculate Stokes parameters for undulator radiation.

    Args:
        args (tuple): Tuple containing arguments:
            - energy_array (np.ndarray): Array of photon energies [eV].
            - bl (dict): Dictionary containing beamline parameters.
            - eBeam (srwlib.SRWLPartBeam): Electron beam properties.
            - magFldCnt (srwlib.SRWLMagFldC): Magnetic field container.
            - resonant_energy (float): Resonant energy [eV].
            - radiation_polarisation (int): Polarisation component to be extracted.

    Returns:
        Tuple[np.ndarray, float]: Tuple containing intensity data and computation time.
    """

    energy_array, bl, eBeam, magFldCnt, resonant_energy, radiation_polarisation = args

    tzero = time()

    try:

        arPrecPar = [0]*5   # for spectral flux vs photon energy
        arPrecPar[0] = 1    # initial UR harmonic to take into account
        arPrecPar[1] = get_undulator_max_harmonic_number(resonant_energy, energy_array[-1]) #final UR harmonic to take into account
        arPrecPar[2] = 1.5  # longitudinal integration precision parameter
        arPrecPar[3] = 1.5  # azimuthal integration precision parameter
        arPrecPar[4] = 1    # calculate flux (1) or flux per unit surface (2)

        npts = len(energy_array)
        stk = srwlib.SRWLStokes() 
        stk.allocate(npts, 1, 1)     
        stk.mesh.zStart = bl['distance']
        stk.mesh.eStart = energy_array[0]
        stk.mesh.eFin =   energy_array[-1]
        stk.mesh.xStart = bl['slitHcenter'] - bl['slitH']/2
        stk.mesh.xFin =   bl['slitHcenter'] + bl['slitH']/2
        stk.mesh.yStart = bl['slitVcenter'] - bl['slitV']/2
        stk.mesh.yFin =   bl['slitVcenter'] + bl['slitV']/2
        und = magFldCnt.arMagFld[0]
        srwlib.srwl.CalcStokesUR(stk, eBeam, und, arPrecPar)
        # intensity = stk.arS[0:npts]
        intensity = stk.to_int(radiation_polarisation)
    except:
         raise ValueError("Error running SRW.")

    return intensity, energy_array, time()-tzero

#***********************************************************************************
# io/rw files
#***********************************************************************************

def write_syned_file(json_file: str, light_source_name: str, ElectronBeamClass: ElectronBeam, 
                     MagneticStructureClass: MagneticStructure) -> None:
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


def read_syned_file(json_file: str) -> Dict[str, Any]:
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


def read_electron_trajectory(file_path: str) -> Dict[str, List[Union[float, None]]]:
    """
    Reads SRW electron trajectory data from a .dat file.

    Args:
        file_path (str): The path to the .dat file containing electron trajectory data.

    Returns:
        dict: A dictionary where keys are the column names extracted from the header
            (ct, X, BetaX, Y, BetaY, Z, BetaZ, Bx, By, Bz),
            and values are lists containing the corresponding column data from the file.
    """
    data = []
    header = None
    with open(file_path, 'r') as file:
        header_line = next(file).strip()
        header = [col.split()[0] for col in header_line.split(',')]
        header[0] = header[0].replace("#","")
        for line in file:
            values = line.strip().split('\t')
            values = [float(value) if value != '' else None for value in values]
            data.append(values)
            
    eTrajDict = {}
    for i, key in enumerate(header):
        eTrajDict[key] = np.asarray([row[i] for row in data])

    return eTrajDict


def write_magnetic_field(mag_field_array: np.ndarray, file_path: Optional[str] = None) -> srwlib.SRWLMagFld3D:
    """
    Generate a 3D magnetic field object based on the input magnetic field array.

    Parameters:
        mag_field_array (np.ndarray): Array containing magnetic field data. Each row corresponds to a point in the 3D space,
                                      where the first column represents the position along the longitudinal axis, and subsequent 
                                      columns represent magnetic field components (e.g., Bx, By, Bz).
        file_path (str, optional): File path to save the generated magnetic field object. If None, the object won't be saved.

    Returns:
        SRWLMagFld3D: Generated 3D magnetic field object.

    """
    nfield, ncomponents = mag_field_array.shape

    field_axis = (mag_field_array[:, 0] - np.mean(mag_field_array[:, 0])) * 1e-3

    Bx = mag_field_array[:, 1]
    if ncomponents > 2:
        By = mag_field_array[:, 2]
    else:
        By = np.zeros(nfield)
    if ncomponents > 3:
        Bz = mag_field_array[:, 3]
    else:
        Bz = np.zeros(nfield)

    magFldCnt = srwlib.SRWLMagFld3D(Bx, By, Bz, 1, 1, nfield - 1, 0, 0, field_axis[-1]-field_axis[0], 1)

    if file_path is not None:
        print(f">>> saving {file_path}")
        magFldCnt.save_ascii(file_path)

    return magFldCnt


def read_magnetic_measurement(file_path: str) -> np.ndarray:
    """
    Read magnetic measurement data from a file.

    Parameters:
        file_path (str): The path to the file containing magnetic measurement data.

    Returns:
        np.ndarray: A NumPy array containing the magnetic measurement data.
    """

    data = []

    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                values = line.split( )
                data.append([float(value) for value in values])
                
    return np.asarray(data)


#***********************************************************************************
# auxiliary functions accelerator functions
#***********************************************************************************

def energy_wavelength(value: float, unity: str) -> float:
    """
    Converts energy to wavelength and vice versa.
    
    Parameters:
        value (float): The value of either energy or wavelength.
        unity (str): The unit of 'value'. Can be 'eV', 'meV', 'keV', 'm', 'nm', or 'A'. Case sensitive. 
        
    Returns:
        float: Converted value in meters if the input is energy, or in eV if the input is wavelength.
        
    Raises:
        ValueError: If an invalid unit is provided.
    """
    factor = 1.0
    
    # Determine the scaling factor based on the input unit
    if unity.endswith('eV') or unity.endswith('meV') or unity.endswith('keV'):
        prefix = unity[:-2]
        if prefix == "m":
            factor = 1e-3
        elif prefix == "k":
            factor = 1e3
    elif unity.endswith('m'):
        prefix = unity[:-1]
        if prefix == "n":
            factor = 1e-9
    elif unity.endswith('A'):
        factor = 1e-10
    else:
        raise ValueError("Invalid unit provided: {}".format(unity))

    return PLANCK * LIGHT / CHARGE / (value * factor)


def get_gamma(E: float) -> float:
    """
    Calculate the Lorentz factor (γ) based on the energy of electrons in GeV.

    Parameters:
        E (float): Energy of electrons in GeV.

    Returns:
        float: Lorentz factor (γ).
    """
    return E * 1e9 / (MASS * LIGHT ** 2) * CHARGE


def get_undulator_max_harmonic_number(resonant_energy: float, photonEnergyMax: float) -> int:
    """
    Calculate the maximum harmonic number for an undulator to be considered by srwlib.CalcStokesUR.

    Args:
        resonant_energy (float): The resonance energy of the undulator [eV].
        photonEnergyMax (float): The maximum photon energy of interest [eV].

    Returns:
        int: The maximum harmonic number.
    """
    srw_max_harmonic_number = int(photonEnergyMax / resonant_energy * 2.5)

    return srw_max_harmonic_number

#***********************************************************************************
# potpourri
#***********************************************************************************

def syned_dictionary(json_file: str, magnetic_measurement: Union[str, None], observation_point: float, 
                     hor_slit: float, ver_slit: float, hor_slit_cen: float, ver_slit_cen: float, **kwargs) -> dict:
    """
    Generate beamline parameters based on a SYNED JSON configuration file and additional input parameters.

    Args:
        json_file (str): Path to the SYNED JSON configuration file.
        magnetic_measurement (Union[str, None]): Path to the file containing magnetic measurement data.
            Overrides SYNED undulator data if provided.
        observation_point (float): Distance to the observation point in meters.
        hor_slit (float): Horizontal slit size in meters.
        ver_slit (float): Vertical slit size in meters.
        hor_slit_cen (float): Horizontal slit center position in meters.
        ver_slit_cen (float): Vertical slit center position in meters.
        **kwargs: Additional keyword arguments for undulator parameters:
            Kh (float): Horizontal undulator parameter K. If -1, it's taken from the SYNED file.
            Kh_phase (float): Initial phase of the horizontal magnetic field in radians.
            Kh_symmetry (int): Symmetry of the horizontal magnetic field vs longitudinal position.
                1 for symmetric (B ~ cos(2*π*n*z/per + φ)),
               -1 for anti-symmetric (B ~ sin(2*π*n*z/per + φ)).
            Kv (float): Vertical undulator parameter K. If -1, it's taken from the SYNED file.
            Kv_phase (float): Initial phase of the vertical magnetic field in radians.
            Kv_symmetry (int): Symmetry of the vertical magnetic field vs longitudinal position.
                1 for symmetric (B ~ cos(2*π*n*z/per + φ)),
               -1 for anti-symmetric (B ~ sin(2*π*n*z/per + φ)).

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
    beamline['magnetic_measurement'] = magnetic_measurement
    # undulator        
    if data["magnetic_structure"]["CLASS_NAME"].startswith("U"):
        Kh = kwargs.get('Kh', None)
        Kh_phase = kwargs.get('Kh_phase', 0)
        Kh_symmetry = kwargs.get('Kh_symmetry', 1)
        Kv = kwargs.get('Kv', None)
        Kv_phase = kwargs.get('Kv_phase', 0)
        Kv_symmetry = kwargs.get('Kv_symmetry', 1)
        if magnetic_measurement is None:
            if Kh == -1:
                Kh =  data["magnetic_structure"]["K_horizontal"]
            if Kv == -1:
                Kv =  data["magnetic_structure"]["K_vertical"]
        beamline['NPeriods'] = data["magnetic_structure"]["number_of_periods"]
        beamline['PeriodID'] = data["magnetic_structure"]["period_length"]
        beamline['Kh'] = Kh
        beamline['KhPhase'] = Kh_phase
        beamline['MagFieldSymmetryH'] = Kh_symmetry
        beamline['Kv'] = Kv
        beamline['KvPhase'] = Kv_phase
        beamline['MagFieldSymmetryV'] = Kv_symmetry
    # bending magnet        
    if data["magnetic_structure"]["CLASS_NAME"].startswith("B"):
        beamline['Bh'] = data["magnetic_structure"]["B_horizontal"]
        beamline['Bv'] = data["magnetic_structure"]["B_vertical"]
        beamline['R'] = data["magnetic_structure"]["radius"]
        beamline['Leff'] = data["magnetic_structure"]["length"]
        beamline['Ledge'] = data["magnetic_structure"]["length_edge"]
    # radiation observation
    beamline['distance'] = observation_point
    beamline['slitH'] = hor_slit
    beamline['slitV'] = ver_slit
    beamline['slitHcenter'] = hor_slit_cen
    beamline['slitVcenter'] = ver_slit_cen
  
    return beamline


def generate_logarithmic_energy_values(emin: float, emax: float, resonant_energy: float, stepsize: float) -> np.ndarray:
    """
    Generate logarithmically spaced energy values within a given energy range.

    Args:
        emin (float): Lower energy range.
        emax (float): Upper energy range.
        resonant_energy (float): Resonant energy.
        stepsize (float): Step size.

    Returns:
        np.ndarray: Array of energy values with logarithmic spacing.
    """

    # Calculate the number of steps for positive and negative energy values
    n_steps_pos = np.ceil(np.log(emax / resonant_energy) / stepsize)
    n_steps_neg = max(0, np.floor(np.log(emin / resonant_energy) / stepsize))

    # Calculate the total number of steps
    n_steps = int(n_steps_pos - n_steps_neg)
    print(f"generate_logarithmic_energy_values - number of steps: {n_steps}")

    # Generate the array of steps with logarithmic spacing
    steps = np.linspace(n_steps_neg, n_steps_pos, n_steps + 1)

    # Compute and return the array of energy values
    return resonant_energy * np.exp(steps * stepsize)

#***********************************************************************************
# time stamp
#***********************************************************************************

def print_elapsed_time(start0: float) -> None:
    """
    Prints the elapsed time since the start of computation.

    Args:
        start0 (float): The start time of computation (in seconds since the epoch).
    """

    deltaT = time() - start0
    if deltaT < 1:
        print(f'>> Total elapsed time: {deltaT * 1000:.2f} ms')
    else:
        hours, rem = divmod(deltaT, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours >= 1:
            print(f'>> Total elapsed time: {int(hours)} h {int(minutes)} min {seconds:.2f} s')
        elif minutes >= 1:
            print(f'>> Total elapsed time: {int(minutes)} min {seconds:.2f} s')
        else:
            print(f'>> Total elapsed time: {seconds:.2f} s')

#***********************************************************************************
# auxiliary functions
#***********************************************************************************

def central_value(arr: np.ndarray) -> float:
    """
    Calculate the central value of a 2D numpy array.
    
    If the number of rows and columns are both odd, return the central element.
    If one dimension is odd and the other is even, return the average of the two central elements.
    If both dimensions are even, return the average of the four central elements.

    Parameters:
    arr (np.ndarray): A 2D numpy array.

    Returns:
    float: The central value or the average of the central values.
    """
    rows, cols = arr.shape
    
    if rows % 2 == 1 and cols % 2 == 1:
        return arr[rows // 2, cols // 2]
    elif rows % 2 == 1 and cols % 2 == 0:
        return np.mean(arr[rows // 2, cols // 2 - 1:cols // 2 + 1])
    elif rows % 2 == 0 and cols % 2 == 1:
        return np.mean(arr[rows // 2 - 1:rows // 2 + 1, cols // 2])
    elif rows % 2 == 0 and cols % 2 == 0:
        return np.mean(arr[rows // 2 - 1:rows // 2 + 1, cols // 2 - 1:cols // 2 + 1])


def unwrap_wft_phase(phase: np.array, x_axis: np.array, y_axis: np.array, 
                     observation_point: float, photon_energy: float) -> np.array:
    """
    Unwraps the wavefront phase by correcting for the quadratic phase term.

    This function corrects the wavefront phase by computing and subtracting 
    the quadratic phase term (QPT), then unwrapping the phase, and finally adding 
    the QPT back. The central values of the phase and QPT are adjusted to ensure 
    proper unwrapping.

    Parameters:
    phase (np.array): The 2D array representing the wavefront phase.
    x_axis (np.array): The 1D array representing the x-axis coordinates.
    y_axis (np.array): The 1D array representing the y-axis coordinates.
    observation_point (float): The distance to the observation point.
    photon_energy (float): The energy of the photons in electron volts (eV).

    Returns:
    np.array: The unwrapped wavefront phase.
    """
    
    # calculation of the quadratic phase term (QPT)
    X, Y = np.meshgrid(x_axis, y_axis)
    k = 2 * np.pi / energy_wavelength(photon_energy, 'eV')
    qpt = np.mod(k * (X**2 + Y**2) / (2 * observation_point), 2 * np.pi)
    qpt -= central_value(qpt)

    # Centering the phase and QPT
    phase -= central_value(phase)
    phase = np.mod(phase, 2 * np.pi) - qpt
    
    # Unwrapping the phase
    phase = unwrap_phase(phase)
    
    # Adding back the QPT
    phase += k * (X**2 + Y**2) / observation_point

    return phase
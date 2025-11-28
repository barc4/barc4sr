# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Wofry backend interface: low-level helpers to build and run Wofry objects.
"""

from __future__ import annotations

import numpy as np
from barc4sr.core.energy import energy_wavelength
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle
from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofryimpl.propagator.util.tally import TallyCoherentModes
from wofryimpl.propagator.util.undulator_coherent_mode_decomposition_1d import (
    UndulatorCoherentModeDecomposition1D,
)

from wofry.propagator.propagator import (
    PropagationElements,
    PropagationManager,
    PropagationParameters,
)


def wofrySourceCMD(bl: dict, energy: float, scan_direction: str, **kwargs) -> UndulatorCoherentModeDecomposition1D:
    """
    Calculates the 1D coherent mode decomposition source using WOFRY.

    Args:
        bl (dict): Dictionary containing beamline parameters.
        energy (float): Photon energy [eV].
        scan_direction (str): Direction of the scan, either 'H' (horizontal) or 'V' (vertical).

    Optional Args (kwargs):
        magnification_forward (float): Forward magnification factor. Defaults to 100.
        magnification_backward (float): Backward magnification factor. Defaults to 1 / magnification_forward.
        distance_to_screen (float): Distance from source to screen [m]. Defaults to magnification_forward.
        nsigma (float): Number of standard deviations defining the extent of the spatial domain. Defaults to 10.
        number_of_points (int): Number of points for spatial sampling. If None, it is computed automatically.

    Returns:
        UndulatorCoherentModeDecomposition1D: Object containing the coherent mode decomposition result.
    """

    magnification_forward = kwargs.get('magnification_forward', 100)
    magnification_backward = kwargs.get('magnification_backward', 1/magnification_forward)
    distance_to_screen = kwargs.get('magnification_forward', 100)
    nsigma = kwargs.get('nsigma', 10)
    number_of_points = kwargs.get('number_of_points', None)

    wavelength = energy_wavelength(energy, 'eV')

    sigma_up = 0.69*np.sqrt(wavelength/(bl['NPeriods']*bl['PeriodID']))
    sigma = np.sqrt(sigma_up**2 + bl[f'ElectronBeamDivergence{scan_direction.upper()}']**2)

    abscissas_interval = (np.tan(sigma / 2) * distance_to_screen * 2)*nsigma*magnification_backward

    if number_of_points is None:
        number_of_points = int(1.25*nsigma*(sigma**2)* distance_to_screen/wavelength)

    coherent_mode_decomposition = UndulatorCoherentModeDecomposition1D(
        electron_energy=bl['ElectronEnergy'],
        electron_current=bl['ElectronCurrent'],
        undulator_period=bl['PeriodID'],
        undulator_nperiods=bl['NPeriods'],
        K=np.sqrt(bl['Kv']**2 + bl['Kh']**2),
        photon_energy=energy,
        abscissas_interval=abscissas_interval,
        number_of_points=number_of_points,
        distance_to_screen=distance_to_screen,
        magnification_x_forward=magnification_forward,
        magnification_x_backward=magnification_backward,
        scan_direction=scan_direction.upper(),
        sigmaxx=bl[f'ElectronBeamSize{scan_direction.upper()}'],
        sigmaxpxp=bl[f'ElectronBeamDivergence{scan_direction.upper()}'],
        useGSMapproximation=False,
        e_energy_dispersion_flag=1,
        e_energy_dispersion_sigma_relative=bl['ElectronEnergySpread'],
        e_energy_dispersion_interval_in_sigma_units=6,
        e_energy_dispersion_points=11)
    
    cmd = coherent_mode_decomposition.calculate()

    return coherent_mode_decomposition


def wofrySlitCMD(src_cmd, window, observation_point, nmodes=-1):

    ff_dist = 100
    angular_window = 2 * np.arctan(window / (2 * observation_point))
    ff_window = np.tan(angular_window/2)*ff_dist*2

    if nmodes == -1:
        nmodes = len(src_cmd.eigenvalues)-1

    tally = TallyCoherentModes()

    for current_mode_index in range(nmodes):

        input_wavefront = src_cmd.get_eigenvector_wavefront(current_mode_index).duplicate()

        optical_element = WOSlit1D(boundary_shape=Rectangle(-ff_window/2, ff_window/2,
                                                            -ff_window/2, ff_window/2))
        propagation_elements = PropagationElements()
        beamline_element = BeamlineElement(optical_element=optical_element,
                                           coordinates=ElementCoordinates(p=ff_dist, q=0, 
                                                                          angle_radial=np.radians(0), 
                                                                          angle_azimuthal=np.radians(0)))
        propagation_elements.add_beamline_element(beamline_element)

        propagation_parameters = PropagationParameters(wavefront=input_wavefront,
                                                       propagation_elements=propagation_elements)
        propagation_parameters.set_additional_parameters('magnification_x', 100)

        propagator = PropagationManager.Instance()
        try:
            propagator.add_propagator(FresnelZoom1D())
        except:
            pass
        tally.append(propagator.do_propagation(propagation_parameters=propagation_parameters,
                                               handler_name='FRESNEL_ZOOM_1D'))

    return tally
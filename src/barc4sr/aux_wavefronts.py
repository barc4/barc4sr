
"""
This module provides auxiliary for wavefronts
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

import numpy as np
from barc4sr.aux_energy import energy_wavelength
from skimage.restoration import unwrap_phase


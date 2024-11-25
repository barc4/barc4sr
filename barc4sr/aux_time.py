
"""
This module provides auxiliary functions for time keeping
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '25/NOV/2024'
__changed__ = '25/NOV/2024'

from time import time

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
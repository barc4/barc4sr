#!/bin/python

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'MIT'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__date__ = '16/JAN/2024'

import os
from time import time

import numpy as np


def logsparced(emin: float, emax:float, eres:float, stepsize:float) -> float:
    """_summary_

    Args:
        emin (float): lower energy range
        emax (float): upper energy range
        eres (float): _description_
        stepsize (float): _description_

    Returns:
        float: _description_
    """

    nStepsPos = np.ceil(np.log(emax/eres)/stepsize)

    if emin < eres:
        nStepsNeg = np.floor(np.log(emin/eres)/stepsize)
    else:
        nStepsNeg = 0

    nSteps = int(nStepsPos-nStepsNeg)
    print(f"number of steps: {nSteps}")

    steps = np.linspace(nStepsNeg, nStepsPos, nSteps+1)
    # print(f"steps grid {steps}\n")

    return eres*np.exp(steps*stepsize)

def main():
    code = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/f_xoppy_undulator_radiation.py"
    mthd = 222
    screen = [4.2e-3, 151, 4.2e-3, 151, 20.0]    # dx, nx, dy, ny, dz
    
    #***********************************************************************************
    # E0 = 50 eV
    #***********************************************************************************
    
    K = 6.548668
    E0 = 50
    
    Emin = 55
    Emax = 60000
    stepSize = 0.0025
    
    Eres = E0
    
    eArray = logsparced(Emin, Emax, Eres, stepSize)
    
    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL-UP_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil-up_hu64E0_%deV"%E0

    cmd = (
        "python %s -fn %s -f  %s -ei %f -ef %f -ne %d -dx %f -nx %d -dy %f -ny %d -dz %f -m %d -Kv %f"
        % (code, fname, fjson, Emin, Emax, len(eArray), screen[0], screen[1], screen[2], screen[3], screen[4], mthd, K)
    )
    print(cmd)
    os.system(cmd)
    
    
if __name__ == "__main__":

    start0 = time()
    main()
    print('End of computation')
    deltaT = time() - start0
    hours, minutes = divmod(deltaT, 3600)
    minutes, seconds = divmod(minutes, 60)
    print('> Total elapsed time: %d h %d min %.2d s' % (int(hours), int(minutes), seconds))
    
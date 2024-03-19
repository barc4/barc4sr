#!/bin/python

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'MIT'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__date__ = '16/JAN/2024'

import argparse
import os
from copy import copy
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
    
    p = argparse.ArgumentParser(description='undulator spectrum calculation')
    p.add_argument('-n', dest='n', metavar='INT', default=0, help='calculation number')
    args = p.parse_args()
    
    num_calc = int(args.n)
    num_chunks = 8
        
    code = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/f_xoppy_spectrum.py"
    
    mthd = 2
    screen = [4.2e-3, 4.2e-3, 20.0]    # dx, dy, dz
    
    #***********************************************************************************
    # E0 = 70 eV
    #***********************************************************************************
    
    K = 5.482769
    E0 = 70
    
    Emin = 55
    Emax = 60000
    stepSize = 0.0025
    
    Eres = E0
    
    eArray = logsparced(Emin, Emax, Eres, stepSize)
    
    dE = (Emax - Emin) / num_chunks
    energy_chunks = []
    
    for i in range(num_chunks):
        bffr = copy(eArray)
        bffr = np.delete(bffr, bffr <= dE * (i))
        if i + 1 != num_chunks:
            bffr = np.delete(bffr, bffr > dE * (i + 1))
        energy_chunks.append(bffr)

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil_hu64_E0_%deV_"%E0

    k = 0
    for chunks in energy_chunks:
        if k == num_calc:
            cmd = (
                "python %s -fn %s -f %s -ei %f -ef %f -ne %d -dx %f -dy %f -dz %f -m %d -Kv %f"
                % (code, fname+"%s"%chr(97+k), fjson, chunks[0], chunks[-1], len(chunks), screen[0], screen[1], screen[2], mthd, K)
            )
            print(cmd)
            os.system(cmd)
        k+=1
        
    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL-UP_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil-up_hu64_E0_%deV_"%E0

    k = 0
    for chunks in energy_chunks:
        if k == num_calc:
            cmd = (
                "python %s -fn %s -f %s -ei %f -ef %f -ne %d -dx %f -dy %f -dz %f -m %d -Kv %f"
                % (code, fname+"%s"%chr(97+k), fjson, chunks[0], chunks[-1], len(chunks), screen[0], screen[1], screen[2], mthd, K)
            )
            print(cmd)
            os.system(cmd)
        k+=1
        
    #***********************************************************************************
    # E0 = 50 eV
    #***********************************************************************************
        
    K = 6.548668
    E0 = 50

    Emin = 35
    Emax = 60000
    stepSize = 0.0025
    
    Eres = E0
    
    eArray = logsparced(Emin, Emax, Eres, stepSize)
    
    dE = (Emax - Emin) / num_chunks
    energy_chunks = []
    
    for i in range(num_chunks):
        bffr = copy(eArray)
        bffr = np.delete(bffr, bffr <= dE * (i))
        if i + 1 != num_chunks:
            bffr = np.delete(bffr, bffr > dE * (i + 1))
        energy_chunks.append(bffr)

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil_hu64_E0_%deV_"%E0

    k = 0
    for chunks in energy_chunks:
        if k == num_calc:
            cmd = (
                "python %s -fn %s -f %s -ei %f -ef %f -ne %d -dx %f -dy %f -dz %f -m %d -Kv %f"
                % (code, fname+"%s"%chr(97+k), fjson, chunks[0], chunks[-1], len(chunks), screen[0], screen[1], screen[2], mthd, K)
            )
            print(cmd)
            os.system(cmd)
        k+=1
        
    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL-UP_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil-up_hu64_E0_%deV_"%E0

    k = 0
    for chunks in energy_chunks:
        if k == num_calc:
            cmd = (
                "python %s -fn %s -f %s -ei %f -ef %f -ne %d -dx %f -dy %f -dz %f -m %d -Kv %f"
                % (code, fname+"%s"%chr(97+k), fjson, chunks[0], chunks[-1], len(chunks), screen[0], screen[1], screen[2], mthd, K)
            )
            print(cmd)
            os.system(cmd)
        k+=1


if __name__ == "__main__":
    start0 = time()
    main()
    print('End of computation')
    deltaT = time() - start0
    hours, minutes = divmod(deltaT, 3600)
    minutes, seconds = divmod(minutes, 60)
    print('> Total elapsed time: %d h %d min %.2d s' % (int(hours), int(minutes), seconds))
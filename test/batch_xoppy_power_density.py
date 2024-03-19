#!/bin/python

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'MIT'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__date__ = '16/JAN/2024'

import os
from time import time


def main():
    K = 5.482769
    E0 = 70
    mthd = 2
    screen = [4.2e-3, 1501, 4.2e-3, 1501, 20.0]    # dx, nx, dy, ny, dz
    
    code = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/f_xoppy_power_density.py"

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil_hu64_E0_%deV"%E0

    cmd = (
        "python %s -fn %s -f %s -dx %f -nx %d -dy %f -ny %d -dz %f -m %d -Kv %f"
        % (code, fname, fjson, screen[0], screen[1], screen[2], screen[3], screen[4], mthd, K)
    )
    print(cmd)
    os.system(cmd)

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL-UP_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil-up_hu64_E0_%deV"%E0

    cmd = (
        "python %s -fn %s -f %s -dx %f -nx %d -dy %f -ny %d -dz %f -m %d -Kv %f"
        % (code, fname, fjson, screen[0], screen[1], screen[2], screen[3], screen[4], mthd, K)
    )
    print(cmd)
    os.system(cmd)

    K = 6.548668
    E0 = 50

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil_hu64_E0_%deV"%E0

    cmd = (
        "python %s -fn %s -f %s -dx %f -nx %d -dy %f -ny %d -dz %f -m %d -Kv %f"
        % (code, fname, fjson, screen[0], screen[1], screen[2], screen[3], screen[4], mthd, K)
    )
    print(cmd)
    os.system(cmd)

    fjson = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/oasys_SOLEIL-UP_HU64.json"
    fname = "/Users/celestre/Work/simulations/power_calculations_SOLEIL/Hermes/_results/soleil-up_hu64_E0_%deV"%E0

    cmd = (
        "python %s -fn %s -f %s -dx %f -nx %d -dy %f -ny %d -dz %f -m %d -Kv %f"
        % (code, fname, fjson, screen[0], screen[1], screen[2], screen[3], screen[4], mthd, K)
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
    
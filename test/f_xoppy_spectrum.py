#!/bin/python

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'MIT'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__date__ = '07/JAN/2024'

import argparse
import json
import pickle

import numpy as np
from xoppylib.sources.xoppy_undulators import xoppy_calc_undulator_spectrum

if __name__ == '__main__':

    # ********************************************************
    # arguments
    
    p = argparse.ArgumentParser(description='undulator spectrum calculation')
    p.add_argument('-fn', dest='filename', metavar='STRING', default=None, help='file name for the pickle file')
    p.add_argument('-f', dest='jsonfile', metavar='STRING', default=None, help='SYNED LightSource json file path all the way to .json')
    p.add_argument('-ei', dest='ei', metavar='FLOAT', default=None, help='start beam energy in [eV]')
    p.add_argument('-ef', dest='ef', metavar='FLOAT', default=None, help='finish beam energy in [eV]')
    p.add_argument('-ne', dest='ne', metavar='INT', default=None, help='energy steps')
    p.add_argument('-dx', dest='dx', metavar='FLOAT', default=None, help='horizontal slit opening [m]')
    p.add_argument('-dxc', dest='dxc', metavar='FLOAT', default=0, help='horizontal slit centre [m]')
    p.add_argument('-dy', dest='dy', metavar='FLOAT', default=None, help='vertical slit opening [m]')
    p.add_argument('-dyc', dest='dyc', metavar='FLOAT', default=0, help='vertical slit centre [m]')
    p.add_argument('-dz', dest='dz', metavar='FLOAT', default=20, help='slit position for spectrum calculation [m]')
    p.add_argument('-Kh', dest='Kh', metavar='FLOAT', default=-1, help='horizontal undulator parameter')
    p.add_argument('-Kv', dest='Kv', metavar='FLOAT', default=-1, help='vertical undulator parameter')
    p.add_argument('-Kphase', dest='Kphase', metavar='FLOAT', default=0, help='phase between Kh and Kv')
    p.add_argument('-m', dest='m', metavar='INT', default=2, help='preferred calculation method')

    args = p.parse_args()
    jsonfile = args.jsonfile
    filename = args.filename

    f = open(jsonfile)
    data = json.load(f)
    f.close()

    if float(args.Kv) == -1:
        Kv = data["magnetic_structure"]["K_vertical"]
    else:
        Kv = float(args.Kv)
    if float(args.Kh) == -1:
        Kh = data["magnetic_structure"]["K_horizontal"]
    else:
        Kh = float(args.Kh) 
    
    energy, flux, spectral_power, cumulated_power = xoppy_calc_undulator_spectrum(
        ELECTRONENERGY=data["electron_beam"]["energy_in_GeV"],
        ELECTRONENERGYSPREAD=data["electron_beam"]["energy_spread"],
        ELECTRONCURRENT=data["electron_beam"]["current"],
        ELECTRONBEAMSIZEH=np.sqrt(data["electron_beam"]["moment_xx"]),
        ELECTRONBEAMSIZEV=np.sqrt(data["electron_beam"]["moment_yy"]),
        ELECTRONBEAMDIVERGENCEH=np.sqrt(data["electron_beam"]["moment_xpxp"]),
        ELECTRONBEAMDIVERGENCEV=np.sqrt(data["electron_beam"]["moment_ypyp"]),
        PERIODID=data["magnetic_structure"]["period_length"],
        NPERIODS=data["magnetic_structure"]["number_of_periods"],
        KV=Kv,
        KH=Kh,
        KPHASE=float(args.Kphase),
        DISTANCE=float(args.dz),
        GAPH=float(args.dx),
        GAPV=float(args.dy),
        GAPH_CENTER=float(args.dxc),
        GAPV_CENTER=float(args.dyc),
        PHOTONENERGYMIN=float(args.ei),
        PHOTONENERGYMAX=float(args.ef),
        PHOTONENERGYPOINTS=int(args.ne),
        METHOD=int(args.m),
        USEEMITTANCES=1
    )

    file = open('%s_spectrum.pickle'%filename, 'wb')
    pickle.dump([energy, flux, spectral_power, cumulated_power], file)
    file.close()
#!/bin/python

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'MIT'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__date__ = '16/JAN/2024'

import argparse
import json

import numpy as np
from xoppylib.sources.xoppy_undulators import xoppy_calc_undulator_radiation

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
    p.add_argument('-nx', dest='nx', metavar='INT', default=51, help='horizontal points')
    p.add_argument('-dy', dest='dy', metavar='FLOAT', default=None, help='vertical slit opening [m]')
    p.add_argument('-ny', dest='ny', metavar='INT', default=51, help='vertical points')
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
    
    h5_parameters = dict()
    h5_parameters["ELECTRONENERGY"] = data["electron_beam"]["energy_in_GeV"]
    h5_parameters["ELECTRONENERGYSPREAD"] = data["electron_beam"]["energy_spread"]
    h5_parameters["ELECTRONCURRENT"] = data["electron_beam"]["current"]
    h5_parameters["ELECTRONBEAMSIZEH"] = np.sqrt(data["electron_beam"]["moment_xx"])
    h5_parameters["ELECTRONBEAMSIZEV"] = np.sqrt(data["electron_beam"]["moment_yy"])
    h5_parameters["ELECTRONBEAMDIVERGENCEH"] = np.sqrt(data["electron_beam"]["moment_xpxp"])
    h5_parameters["ELECTRONBEAMDIVERGENCEV"] = np.sqrt(data["electron_beam"]["moment_ypyp"])
    h5_parameters["PERIODID"] = data["magnetic_structure"]["period_length"]
    h5_parameters["NPERIODS"] = data["magnetic_structure"]["number_of_periods"]
    if float(args.Kv) == -1:
        h5_parameters["KV"] = data["magnetic_structure"]["K_vertical"]
    else:
        h5_parameters["KV"] = float(args.Kv)
    if float(args.Kh) == -1:
        h5_parameters["KH"] = data["magnetic_structure"]["K_horizontal"]
    else:
        h5_parameters["KH"] = float(args.Kh)    
    h5_parameters["KPHASE"] = float(args.Kphase)
    h5_parameters["DISTANCE"] = float(args.dz)
    h5_parameters["SETRESONANCE"] = 0
    h5_parameters["HARMONICNUMBER"] = 1
    h5_parameters["GAPH"] = float(args.dx)
    h5_parameters["GAPV"] = float(args.dy)
    h5_parameters["HSLITPOINTS"] = int(args.nx)
    h5_parameters["VSLITPOINTS"] = int(args.ny)
    h5_parameters["METHOD"] = int(args.m)
    h5_parameters["PHOTONENERGYMIN"] = float(args.ei)
    h5_parameters["PHOTONENERGYMAX"] = float(args.ef)
    h5_parameters["PHOTONENERGYPOINTS"] = int(args.ne)
    h5_parameters["USEEMITTANCES"] = 1

    energy, horizontal, vertical, flux3D, code = xoppy_calc_undulator_radiation(
        ELECTRONENERGY=h5_parameters["ELECTRONENERGY"],
        ELECTRONENERGYSPREAD=h5_parameters["ELECTRONENERGYSPREAD"],
        ELECTRONCURRENT=h5_parameters["ELECTRONCURRENT"],
        ELECTRONBEAMSIZEH=h5_parameters["ELECTRONBEAMSIZEH"],
        ELECTRONBEAMSIZEV=h5_parameters["ELECTRONBEAMSIZEV"],
        ELECTRONBEAMDIVERGENCEH=h5_parameters["ELECTRONBEAMDIVERGENCEH"],
        ELECTRONBEAMDIVERGENCEV=h5_parameters["ELECTRONBEAMDIVERGENCEV"],
        PERIODID=h5_parameters["PERIODID"],
        NPERIODS=h5_parameters["NPERIODS"],
        KV=h5_parameters["KV"],
        KH=h5_parameters["KH"],
        KPHASE=h5_parameters["KPHASE"],
        DISTANCE=h5_parameters["DISTANCE"],
        SETRESONANCE=h5_parameters["SETRESONANCE"],
        HARMONICNUMBER=h5_parameters["HARMONICNUMBER"],
        GAPH=h5_parameters["GAPH"],
        GAPV=h5_parameters["GAPV"],
        HSLITPOINTS=h5_parameters["HSLITPOINTS"],
        VSLITPOINTS=h5_parameters["VSLITPOINTS"],
        METHOD=h5_parameters["METHOD"],
        PHOTONENERGYMIN=h5_parameters["PHOTONENERGYMIN"],
        PHOTONENERGYMAX=h5_parameters["PHOTONENERGYMAX"],
        PHOTONENERGYPOINTS=h5_parameters["PHOTONENERGYPOINTS"],
        USEEMITTANCES=h5_parameters["USEEMITTANCES"],
        h5_file="%s_undulator_radiation.h5"%filename,
        h5_entry_name="XOPPY_RADIATION",
        h5_initialize=True,
        h5_parameters=h5_parameters,
    )
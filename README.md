# barc4sr
**BARC** library for **S**ynchrotron **R**adiation

This library was created for facilitating the use of [SRW](https://github.com/ochubar/SRW) for a few routine calculations 
such as:

- undulator emission spectra - on axis or through a slit;
- power (density) through a slit;
- undulator radiation spectral and spatial distribution;

All calculations take either an ideal magnetic field or a tabulated measurement. In the 
case of a tabulated measurement, a Monte-Carlo sampling of the electron-beam phase space 
is necessary for a few calculations and recommended for others. 

This module is inspired by [xoppy](https://github.com/oasys-kit/xoppylib), but but with 
the "multi-electron" calculations and parallelisation of a few routines. 

## installation

bar4sr is on PyPi! So it can be installed as ```pip install barc4sr``` _hooray_!!! Otherwise,
clone the project, fix the (many bugs) and help improve it...

## TODO:

Ideally, I want to add the same functionalities to bending-magnets and wigglers through SRW.
I am also considering interfacing [SPECTRA](https://spectrax.org/spectra/index.html), but only if there is the need for that.

## Examples:
Check the examples! You can learn a lot from them.

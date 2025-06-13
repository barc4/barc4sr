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


---

 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/barc4/barc4sr">barc library for synchrotron radiation</a> by <span property="cc:attributionName">Rafael Celestre, Synchrotron SOLEIL</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p> 

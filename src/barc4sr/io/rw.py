# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
Read/Write helpers for barc4sr results in HDF5 format.
"""

from __future__ import annotations

import os
import pickle
from array import array
from copy import copy, deepcopy

import h5py as h5
import numpy as np
import scipy.integrate as integrate
from scipy.constants import physical_constants

from barc4sr.core.energy import get_gamma

try:
    import srwpy.srwlib as srwlib
    USE_SRWLIB = True
except:
    import oasys_srw.srwlib as srwlib
    USE_SRWLIB = True
if USE_SRWLIB is False:
     raise AttributeError("SRW is not available")

CHARGE = physical_constants["atomic unit of charge"][0]

# ---------------------------------------------------------------------------
# electron trajectory
# ---------------------------------------------------------------------------

def write_electron_trajectory(file_name:str, eTraj: srwlib.SRWLPrtTrj, energy: float) -> dict:
    """
    Save electron trajectory data to an HDF5 file and return a structured
    dictionary containing the trajectory, magnetic field, and metadata.

    Parameters
    ----------
    file_name : str or None
        Base file path for saving the trajectory data. The data is saved
        in a file with the suffix ``"_eTraj.h5"``. If None, no file is written.
    eTraj : SRWLPrtTrj
        SRW particle trajectory object containing arrays:
            arX, arXp, arY, arYp, arZ, arZp, arBx, arBy, arBz

    energy : float
        Electron beam energy in GeV used to generate this trajectory.

    Returns
    -------
    dict
        Dictionary with the following structure:

        {
            "eTraj": {
                "ct", "X", "Xp", "Y", "Yp", "Z", "Zp"
            },
            "mag_field": {
                "s": array_like (N,),
                "B": array_like (N, 3)
            },
            "meta": {
                "energy", "gamma", "n_points",
                "ct_start", "ct_end"
            }
        }
    """

    n_points = int(eTraj.np)

    ct = np.linspace(float(eTraj.ctStart), float(eTraj.ctEnd), n_points)

    eTraj_dict = {
        "eTraj": {
            "ct": ct,
            "X": np.asarray(eTraj.arX, dtype=float),
            "Xp": np.asarray(eTraj.arXp, dtype=float),
            "Y": np.asarray(eTraj.arY, dtype=float),
            "Yp": np.asarray(eTraj.arYp, dtype=float),
            "Z": np.asarray(eTraj.arZ, dtype=float),
            "Zp": np.asarray(eTraj.arZp, dtype=float),
        }
    }

    s = np.asarray(eTraj.arZ, dtype=float)
    B = np.column_stack(
        [
            np.asarray(eTraj.arBx, dtype=float),
            np.asarray(eTraj.arBy, dtype=float),
            np.asarray(eTraj.arBz, dtype=float),
        ]
    )

    eTraj_dict["mag_field"] = {"s": s, "B": B}

    eTraj_dict["meta"] = {
        "energy_GeV": float(energy),
        "gamma": float(get_gamma(energy)),
        "n_points": n_points,
        "ct_start": float(eTraj.ctStart),
        "ct_end": float(eTraj.ctEnd),
    }

    if file_name is not None:
        with h5.File(f"{file_name}_eTraj.h5", "w") as f:
            f.attrs["barc4sr_calc"] = "electron_trajectory"
            f.attrs["barc4sr_version"] = "1.0"

            g_t = f.create_group("eTraj")
            for key, arr in eTraj_dict["eTraj"].items():
                g_t.create_dataset(key, data=arr)

            g_B = f.create_group("mag_field")
            g_B.create_dataset("s", data=s)
            g_B.create_dataset("B", data=B)

            g_m = f.create_group("meta")
            for key, val in eTraj_dict["meta"].items():
                g_m.attrs[key] = val

    return eTraj_dict
    

def read_electron_trajectory(file_path: str) -> dict:
    """
    Read an electron trajectory from an HDF5 file written by
    ``write_electron_trajectory``.

    The function reconstructs the full structured dictionary:

        {
            "eTraj": { ... },
            "mag_field": { "s", "B" },
            "meta": { ... }
        }

    Parameters
    ----------
    file_path : str
        Path to the ``*_eTraj.h5`` file.

    Returns
    -------
    dict
        A dictionary with the following structure:

        - ``"eTraj"`` :
            {"ct", "X", "Xp", "Y", "Yp", "Z", "Zp"} as NumPy arrays.

        - ``"mag_field"`` :
            {"s", "B"} where:
                - "s" is an array of shape (N,)
                - "B" is an array of shape (N, 3),
            compatible with ``check_magnetic_field_dictionary``.

        - ``"meta"`` :
            Metadata stored as attributes:
                {"energy_GeV", "gamma", "n_points", "ct_start", "ct_end"}.

    Raises
    ------
    ValueError
        If the file does not contain the expected HDF5 groups.
    """
    result: dict[str, dict] = {
        "eTraj": {},
        "mag_field": {},
        "meta": {},
    }

    with h5.File(file_path, "r") as f:
        calc = f.attrs.get("barc4sr_calc", None)
        if calc not in (None, "electron_trajectory"):
            raise ValueError(
                f"Unexpected barc4sr_calc={calc!r} in file {file_path}."
            )

        if "eTraj" not in f:
            raise ValueError(
                f"Invalid file structure: missing 'eTraj' group."
            )
        g_t = f["eTraj"]
        for key in g_t.keys():
            result["eTraj"][key] = g_t[key][:].astype(float)

        if "mag_field" not in f:
            raise ValueError(
                f"Invalid file structure: missing 'mag_field' group."
            )
        g_B = f["mag_field"]
        result["mag_field"]["s"] = g_B["s"][:].astype(float)
        result["mag_field"]["B"] = g_B["B"][:].astype(float)

        if "meta" not in f:
            raise ValueError(
                f"Invalid file structure: missing 'meta' group."
            )
        g_m = f["meta"]
        for key, value in g_m.attrs.items():
            result["meta"][key] = float(value)

    return result


def read_electron_trajectory_dat(file_path: str) -> dict:
    """
    Read SRW electron trajectory data from a .dat file (SRW native text format)
    and convert it into the standard barc4sr electron trajectory structure:

        {
            "eTraj": {
                "ct", "X", "Xp", "Y", "Yp", "Z", "Zp"
            },
            "mag_field": {
                "s": Z-array,
                "B": (N, 3) array with columns [Bx, By, Bz]
            },
            "meta": {
                "energy_GeV": NaN,
                "gamma": NaN,
                "n_points": N,
                "ct_start": ct[0],
                "ct_end": ct[-1]
            }
        }

    Parameters
    ----------
    file_path : str
        Path to the SRW .dat trajectory file.

    Returns
    -------
    dict
        Electron trajectory dictionary in the new barc4sr format.
    """
    data_rows = []
    with open(file_path, "r") as f:
        header_line = next(f).strip()
        header = [col.split()[0] for col in header_line.split(",")]
        header[0] = header[0].lstrip("#")

        for line in f:
            values = line.strip().split("\t")
            data_rows.append([float(v) if v != "" else np.nan for v in values])

    data = np.asarray(data_rows)
    n_points = data.shape[0]

    def col(name):
        try:
            idx = header.index(name)
            return data[:, idx]
        except ValueError:
            return np.full(n_points, np.nan)

    eTraj_block = {
        "ct": col("ct"),
        "X": col("X"),
        "Xp": col("Xp"),
        "Y": col("Y"),
        "Yp": col("Yp"),
        "Z": col("Z"),
        "Zp": col("Zp"),
    }

    s = eTraj_block["Z"]
    B = np.column_stack([col("Bx"), col("By"), col("Bz")])

    mag_field_block = {"s": s, "B": B}

    meta_block = {
        "energy_GeV": np.nan,
        "gamma": np.nan,
        "n_points": n_points,
        "ct_start": eTraj_block["ct"][0] if n_points > 0 else np.nan,
        "ct_end": eTraj_block["ct"][-1] if n_points > 0 else np.nan,
    }

    return {
        "eTraj": eTraj_block,
        "mag_field": mag_field_block,
        "meta": meta_block,
    }

# ---------------------------------------------------------------------------
# Wavefront
# ---------------------------------------------------------------------------
   
def write_wavefront(
    file_name: str,
    wfr: srwlib.SRWLWfr,
    selected_polarisations: list,
    number_macro_electrons: int,
    propagation_distance: float | None = None,
) -> dict:
    """
    Write wavefront data (intensity, phase, and wavefront object) to an HDF5
    file and return the corresponding dictionary.

    Parameters
    ----------
    file_name : str
        Base file path for saving the wavefront data. The data is stored
        in a file named ``"<file_name>_undulator_wfr.h5"``.
    wfr : SRWLWfr
        SRW wavefront object containing the simulated electric field.
    selected_polarisations : list or str
        Polarisations to export. Can be a single string or a list of
        strings. Accepted values: 'LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T'.
    number_macro_electrons : int
        Number of macro electrons used in the simulation.
    propagation_distance : float or None, optional
        Propagation distance used for curvature estimation. If None,
        Rx and Ry are taken from the wavefront object itself.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'wfr': the SRW wavefront object.
            - 'axis': {'x', 'y'} in meters.
            - 'energy': photon energy [eV].
            - 'Rx', 'Ry': curvature radii [m].
            - 'intensity': {pol: 2D array}.
            - 'phase': {pol: 2D array}.
    """

    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings or a string.")

    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    wfr_qpt = deepcopy(wfr)
    wfrDict: dict[str, object] = {"wfr": wfr}

    wfrDict["axis"] = {
        "x": np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx),
        "y": np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny),
    }

    all_polarisations = ["LH", "LV", "L45", "L135", "CR", "CL", "T"]
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return write_wavefront(file_name, wfr, ["T"], number_macro_electrons)

    if propagation_distance is None:
        Rx, Ry = wfr.Rx, wfr.Ry
    else:
        Rx, Ry = propagation_distance, propagation_distance

    wfrDict["energy"] = wfr.mesh.eStart
    wfrDict["intensity"] = {}
    wfrDict["phase"] = {}
    wfrDict["Rx"], wfrDict["Ry"] = Rx, Ry

    _inIntType = int(number_macro_electrons)
    _inDepType = 3

    quadratic_phase_term = srwlib.SRWLOptL(_Fx=Rx, _Fy=Ry)
    pp_spherical_wave =  [0, 0, 1.0, 1, 0, 1., 1., 1., 1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    OE = [quadratic_phase_term]
    PP = [pp_spherical_wave]
    
    optBL = srwlib.SRWLOptC(OE, PP)
    srwlib.srwl.PropagElecField(wfr_qpt, optBL)

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index

        arInt = array("f", [0]*wfr_qpt.mesh.nx*wfr_qpt.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arInt, wfr_qpt, _inPol, _inIntType, _inDepType,
                                         wfr_qpt.mesh.eStart, 0, 0)
        intensity = np.asarray(arInt, dtype="float64").reshape((wfr_qpt.mesh.ny, wfr_qpt.mesh.nx))
        wfrDict["intensity"][polarisation] = intensity

        arPh = array("d", [0] * wfr_qpt.mesh.nx * wfr_qpt.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(arPh, wfr_qpt, _inPol, 4, _inDepType, 
                                         wfr_qpt.mesh.eStart, 0, 0)
        phase = np.asarray(arPh, dtype="float64").reshape((wfr_qpt.mesh.ny, wfr_qpt.mesh.nx))
        wfrDict["phase"][polarisation] = phase

    if file_name is not None:
        with h5.File(f"{file_name}_wfr.h5", "w") as f:
            f.attrs["barc4sr_calc"] = "wavefront"
            f.attrs["barc4sr_version"] = "1.0"

            g_axis = f.create_group("axis")
            g_axis.create_dataset("x", data=wfrDict["axis"]["x"])
            g_axis.create_dataset("y", data=wfrDict["axis"]["y"])

            g_meta = f.create_group("meta")
            g_meta.attrs["energy"] = float(wfrDict["energy"])
            g_meta.attrs["Rx"] = float(Rx)
            g_meta.attrs["Ry"] = float(Ry)
            g_meta.attrs["n_macro_electrons"] = int(number_macro_electrons)
            g_meta.attrs["polarisations"] = ",".join(selected_polarisations)

            g_int = f.create_group("intensity")
            for pol, img in wfrDict["intensity"].items():
                g_int.create_dataset(pol, data=img)

            g_phase = f.create_group("phase")
            for pol, img in wfrDict["phase"].items():
                g_phase.create_dataset(pol, data=img)

            wfr_pickled = pickle.dumps(wfr)
            f.create_dataset("wfr", data=np.void(wfr_pickled))

    return wfrDict

def read_wavefront(file_name: str) -> dict:
    """
    Read wavefront data from an HDF5 file written by ``write_wavefront``
    and reconstruct the full wavefront dictionary.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file containing wavefront data
        (``*_undulator_wfr.h5``).

    Returns
    -------
    dict
        Dictionary with keys:
            - 'wfr': SRW wavefront object (if present).
            - 'axis': {'x', 'y'} in meters.
            - 'energy': photon energy [eV] (if stored).
            - 'Rx', 'Ry': curvature radii [m] (if stored).
            - 'intensity': {pol: 2D numpy arrays}.
            - 'phase': {pol: 2D numpy arrays}.
    """
    if not (file_name.endswith("h5") or file_name.endswith("hdf5")):
        raise ValueError("Only HDF5 format supported for this function.")

    with h5.File(file_name, "r") as f:
        calc = f.attrs.get("barc4sr_calc", None)
        if calc not in (None, "wavefront"):
            raise ValueError(f"Unexpected barc4sr_calc={calc!r} in file {file_name}.")

        if "axis" not in f:
            raise ValueError("Invalid file structure: missing 'axis' group.")
        g_axis = f["axis"]
        x = g_axis["x"][()]
        y = g_axis["y"][()]

        if "intensity" not in f:
            raise ValueError("Invalid file structure: missing 'intensity' group.")
        g_int = f["intensity"]
        intensity = {pol: g_int[pol][()] for pol in g_int.keys()}

        if "phase" not in f:
            raise ValueError("Invalid file structure: missing 'phase' group.")
        g_phase = f["phase"]
        phase = {pol: g_phase[pol][()] for pol in g_phase.keys()}

        wfr = None
        if "wfr" in f:
            wfr = pickle.loads(f["wfr"][()].tobytes())

    Rx = getattr(wfr, "Rx", None) if wfr is not None else None
    Ry = getattr(wfr, "Ry", None) if wfr is not None else None
    energy = getattr(wfr.mesh, "eStart", None) if wfr is not None else None

    return {
        "wfr": wfr,
        "axis": {"x": x, "y": y},
        "energy": energy,
        "Rx": Rx,
        "Ry": Ry,
        "intensity": intensity,
        "phase": phase,
    }

# ---------------------------------------------------------------------------
# Power density
# ---------------------------------------------------------------------------

def write_power_density(
    file_name: str,
    stks: srwlib.SRWLStokes,
    selected_polarisations: list,
) -> dict:
    """
    Write power density data to an HDF5 file and return a power dictionary.

    Parameters
    ----------
    file_name : str
        Base file path for saving the power density data. The data is stored
        in a file named ``"<file_name>_power_density.h5"``.
    stks : SRWLStokes
        SRW Stokes object containing the simulated power density.
    selected_polarisations : list or str
        Polarisations to export. Can be a single string or a list of
        strings. Accepted values: 'LH', 'LV', 'L45', 'L135', 'CR', 'CL', 'T'.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'axis': {'x', 'y'} in meters.
            - for each polarisation:
                - 'map': 2D power density map.
                - 'integrated': total power [W].
                - 'peak': peak power density [W/m^2].
    """
    if isinstance(selected_polarisations, str):
        selected_polarisations = [selected_polarisations]
    elif not isinstance(selected_polarisations, list):
        raise ValueError("Input should be a list of strings or a string.")

    for i, s in enumerate(selected_polarisations):
        if not s.isupper():
            selected_polarisations[i] = s.upper()

    pwrDict: dict[str, object] = {}

    pwrDict["axis"] = {
        "x": np.linspace(stks.mesh.xStart, stks.mesh.xFin, stks.mesh.nx),
        "y": np.linspace(stks.mesh.yStart, stks.mesh.yFin, stks.mesh.ny),
    }

    all_polarisations = ["LH", "LV", "L45", "L135", "CR", "CL", "T"]
    pol_map = {pol: i for i, pol in enumerate(all_polarisations)}

    selected_indices = [pol_map[pol] for pol in selected_polarisations if pol in pol_map]

    if not selected_indices:
        print(">>>>> No valid polarisation found - defaulting to 'T'")
        return write_power_density(file_name, stks, selected_polarisations=["T"])

    dx = pwrDict["axis"]["x"][1] - pwrDict["axis"]["x"][0]
    dy = pwrDict["axis"]["y"][1] - pwrDict["axis"]["y"][0]

    for polarisation, index in zip(selected_polarisations, selected_indices):
        _inPol = index
        pow_map = np.reshape(stks.to_int(_inPol), (stks.mesh.ny, stks.mesh.nx))
        cum_pow = pow_map.sum() * dx * dy
        pwrDict[polarisation] = {
            "map": pow_map,
            "integrated": cum_pow,
            "peak": pow_map.max(),
        }

    if file_name is not None:
        with h5.File(f"{file_name}_power_density.h5", "w") as f:
            f.attrs["barc4sr_calc"] = "power_density"
            f.attrs["barc4sr_version"] = "1.0"

            f_axis = f.create_group("axis")
            f_axis.create_dataset("x", data=pwrDict["axis"]["x"])
            f_axis.create_dataset("y", data=pwrDict["axis"]["y"])

            for pol in selected_polarisations:
                pol_group = f.create_group(pol)
                pol_group.create_dataset("map", data=pwrDict[pol]["map"])
                pol_group.attrs["integrated"] = pwrDict[pol]["integrated"]
                pol_group.attrs["peak"] = pwrDict[pol]["peak"]

    return pwrDict

def read_power_density(file_name: str) -> dict:
    """
    Read power density data from an HDF5 file written by
    ``write_power_density`` and reconstruct the power dictionary.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file containing power density data.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'axis': {'x', 'y'} in meters.
            - <polarisation>: for each polarisation:
                - 'map': 2D numpy array of power density.
                - 'integrated': total power [W].
                - 'peak': peak power density [W/m^2].
    """
    if not (file_name.endswith("h5") or file_name.endswith("hdf5")):
        raise ValueError("Only HDF5 format supported for this function.")

    with h5.File(file_name, "r") as f:
        calc = f.attrs.get("barc4sr_calc", None)
        if calc not in (None, "power_density"):
            raise ValueError(f"Unexpected barc4sr_calc={calc!r} in file {file_name}.")

        if "axis" not in f:
            raise ValueError("Invalid file structure: missing 'axis' group.")
        g_axis = f["axis"]
        x = g_axis["x"][()]
        y = g_axis["y"][()]

        pwrDict: dict[str, object] = {"axis": {"x": x, "y": y}}

        for key in f.keys():
            if key == "axis":
                continue
            pol_group = f[key]
            pwrDict[key] = {
                "map": pol_group["map"][()],
                "integrated": float(pol_group.attrs["integrated"]),
                "peak": float(pol_group.attrs["peak"]),
            }

    return pwrDict

# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def write_spectrum(file_name: str, spectrum: dict) -> dict:
    """
    Save computed spectrum data to an HDF5 file and return a processed
    spectrum dictionary.

    This function processes the input `spectrum` dictionary to compute:
        - Flux [ph/s/0.1%bw]
        - Spectral power [W/eV]
        - Cumulated power [W] (integrated from minimum energy up to each point)
        - Integrated power [W] (total power over the entire spectrum)

    The data is stored in an HDF5 file under a root group ``"spectrum"``,
    with one subgroup per polarisation containing the computed arrays.

    Parameters
    ----------
    file_name : str
        Base file path for saving the spectrum data. The data is stored
        in a file named ``"<file_name>_spectrum.h5"``.
    spectrum : dict
        Dictionary containing the simulated spectrum with keys:
            - 'energy': 1D array of photon energies [eV].
            - 'axis': {'x', 'y'} window sizes [m] or [rad].
            - one key per polarisation, each a 1D flux array.

    Returns
    -------
    dict
        Processed spectrum dictionary with:
            - 'energy'
            - 'window': {'dx', 'dy'}
            - for each polarisation:
                - 'flux'
                - 'spectral_power'
                - 'cumulated_power'
                - 'integrated_power'
    """
    spectrumDict: dict[str, object] = {
        "energy": spectrum["energy"],
        "window": {
            "dx": spectrum["axis"]["x"],
            "dy": spectrum["axis"]["y"],
        },
    }

    for polarisation, data in spectrum.items():
        if polarisation in ["energy", "axis"]:
            continue

        flux = data.reshape(len(spectrum["energy"]))
        spectral_power = flux * CHARGE * 1e3
        cumulated_power = integrate.cumulative_trapezoid(
            spectral_power, spectrum["energy"], initial=0
        )
        integrated_power = cumulated_power[-1]

        spectrumDict[polarisation] = {
            "flux": flux,
            "spectral_power": spectral_power,
            "cumulated_power": cumulated_power,
            "integrated_power": integrated_power,
        }

    if file_name is not None:
        with h5.File(f"{file_name}_spectrum.h5", "w") as f:
            f.attrs["barc4sr_calc"] = "spectrum"
            f.attrs["barc4sr_version"] = "1.0"

            spec_group = f.create_group("spectrum")

            spec_group.create_dataset("energy", data=spectrumDict["energy"])

            window_group = spec_group.create_group("window")
            window_group.create_dataset("dx", data=spectrumDict["window"]["dx"])
            window_group.create_dataset("dy", data=spectrumDict["window"]["dy"])

            for pol, pol_data in spectrumDict.items():
                if pol in ["energy", "window"]:
                    continue

                pol_group = spec_group.create_group(pol)
                for key, array in pol_data.items():
                    pol_group.create_dataset(key, data=array)

    return spectrumDict

def read_spectrum(file_name: str) -> dict:
    """
    Read a spectrum HDF5 file written by ``write_spectrum`` and return
    the spectrum dictionary.

    Parameters
    ----------
    file_name : str
        Base path (without ``"_spectrum.h5"`` suffix).

    Returns
    -------
    dict
        Spectrum dictionary with keys:
            - 'energy'
            - 'window': {'dx', 'dy'}
            - one key per polarisation, each a dict with
              'flux', 'spectral_power', 'cumulated_power', 'integrated_power'.
    """
    spectrumDict: dict[str, object] = {}
    with h5.File(f"{file_name}_spectrum.h5", "r") as f:
        calc = f.attrs.get("barc4sr_calc", None)
        if calc not in (None, "spectrum"):
            raise ValueError(f"Unexpected barc4sr_calc={calc!r} in file {file_name}.")

        if "spectrum" not in f:
            raise ValueError("Invalid file structure: missing 'spectrum' group.")

        spec_group = f["spectrum"]

        spectrumDict["energy"] = spec_group["energy"][:]

        window_group = spec_group["window"]
        spectrumDict["window"] = {
            "dx": window_group["dx"][:],
            "dy": window_group["dy"][:],
        }

        for pol in spec_group:
            if pol in ["energy", "window"]:
                continue

            pol_group = spec_group[pol]
            spectrumDict[pol] = {}
            for key in pol_group:
                spectrumDict[pol][key] = pol_group[key][:]

    return spectrumDict

# ---------------------------------------------------------------------------
# Coherent mode decomposition
# ---------------------------------------------------------------------------

def write_cmd(file_name: str, cmd: dict) -> dict:
    """
    Save coherent mode decomposition (CMD) data to an HDF5 file and return
    a processed CMD dictionary.

    Parameters
    ----------
    file_name : str
        Base file path for saving the CMD data. The data is stored
        in a file named ``"<file_name>_cmd.h5"``.
    cmd : dict
        Dictionary containing the CMD results with keys:
            - 'energy': photon energy [eV].
            - 'src_h_cmd', 'src_v_cmd': CMD objects with attributes
              'eigenvalues', 'abscissas', 'CSD'.

    Returns
    -------
    dict
        Processed CMD dictionary:

        {
            'energy': float,
            'source': {
                'H': {
                    'eigenvalues', 'axis', 'occupation',
                    'cumulated', 'CF', 'CSD'
                },
                'V': {...}
            }
        }
    """
    cmdDict = {"energy": cmd["energy"], "source": {}}

    for direction in ["h", "v"]:
        eigenvalues = cmd[f"src_{direction}_cmd"].eigenvalues
        axis = cmd[f"src_{direction}_cmd"].abscissas
        occupation = eigenvalues / eigenvalues.sum()
        cumulated = np.cumsum(occupation)
        CF = occupation[0]
        CSD = np.abs(cmd[f"src_{direction}_cmd"].CSD)

        cmdDict["source"].update(
            {
                direction.upper(): {
                    "eigenvalues": eigenvalues,
                    "axis": axis,
                    "occupation": occupation,
                    "cumulated": cumulated,
                    "CF": CF,
                    "CSD": CSD,
                }
            }
        )

    if file_name is not None:
        with h5.File(f"{file_name}_cmd.h5", "w") as f:
            f.attrs["barc4sr_calc"] = "cmd"
            f.attrs["barc4sr_version"] = "1.0"

            f.attrs["energy"] = float(cmdDict["energy"])

            for direction in ["H", "V"]:
                src = cmdDict["source"][direction]
                dir_group = f.create_group(direction)
                dir_group.create_dataset("eigenvalues", data=src["eigenvalues"])
                dir_group.create_dataset("axis", data=src["axis"])
                dir_group.create_dataset("occupation", data=src["occupation"])
                dir_group.create_dataset("cumulated", data=src["cumulated"])
                dir_group.attrs["CF"] = float(src["CF"])
                dir_group.create_dataset("CSD", data=src["CSD"])

    return cmdDict


def read_cmd(file_name: str) -> dict:
    """
    Read CMD data from an HDF5 file written by ``write_cmd`` and return
    the CMD dictionary.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file containing CMD data (``*_cmd.h5``).

    Returns
    -------
    dict
        CMD dictionary:

        {
            'energy': float,
            'source': {
                'H': {...},
                'V': {...}
            }
        }
    """
    cmdDict = {"source": {}}

    with h5.File(file_name, "r") as f:
        calc = f.attrs.get("barc4sr_calc", None)
        if calc not in (None, "cmd"):
            raise ValueError(f"Unexpected barc4sr_calc={calc!r} in file {file_name}.")

        cmdDict["energy"] = float(f.attrs["energy"])

        for direction in ["H", "V"]:
            dir_group = f[direction]
            cmdDict["source"][direction] = {
                "eigenvalues": dir_group["eigenvalues"][:],
                "axis": dir_group["axis"][:],
                "occupation": dir_group["occupation"][:],
                "cumulated": dir_group["cumulated"][:],
                "CF": float(dir_group.attrs["CF"]),
                "CSD": dir_group["CSD"][:],
            }

    return cmdDict
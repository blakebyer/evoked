import pyabf
from pathlib import Path
from epspkit.base import RecordingData
from pandera.typing import DataFrame
import pandera as pa
import pandas as pd
import numpy as np

"""
    Tabular data should look like:
    id  time  voltage  intensity  sweepNumber
    <str>  <float>  <float>  <int>  <int>  
"""

"""
    Axon binary format is more complicated because intensity metadata is not included.
"""
"""
    For ABF each slice is generally a separate file, so that carries over to the other formats. If you had 11 stimuli, 3 repeats per intensity, and 625 samples, each file will be 20k rows.
"""
## load NWB?
## load other file acquisition types

@pa.check_types
def load_abf(filename, intensities: list[int], id_value: str, repnum: int) -> DataFrame[RecordingData]:
    abf = pyabf.ABF(filename)
    n_intensities = len(intensities)
    n_sweeps = len(abf.sweepList)
    expected = n_intensities * repnum
    if n_sweeps != expected:
        raise ValueError(f"{filename}: expected {expected} sweeps, got {n_sweeps}")
    data = []
    for sweep_i, sweepNumber in enumerate(abf.sweepList):
        abf.setSweep(sweepNumber)

        intensity_index = sweep_i // repnum
        stim_intensity = intensities[intensity_index]

        data.append(
            pd.DataFrame({
                "id": id_value,
                "time": np.asarray(abf.sweepX,dtype=np.float64),             
                "voltage": np.asarray(abf.sweepY,dtype=np.float64),     
                "intensity": stim_intensity, 
                "sweepNumber": sweepNumber,      
            })
        )
    return pd.concat(data, ignore_index=True)

@pa.check_types
def load_csv(filename) -> DataFrame[RecordingData]:
    return pd.read_csv(filename, index_col=False)

@pa.check_types
def load_tsv(filename) -> DataFrame[RecordingData]:
    return pd.read_csv(filename, sep='\t', index_col=False)

@pa.check_types
def load_bulk(
    filenames: list[str],
    intensities: list[int] | None = None,
    id_values: list[str] | None = None,
    repnum: int | None = None,
) -> DataFrame[RecordingData]:

    recordings = []
    seen_ids = set()
    abf_i = 0

    for file in filenames:
        file = Path(file)
        suffix = file.suffix.lower()

        if suffix == ".abf":
            if any(var is None for var in (intensities, repnum, id_values)):
                raise ValueError("ABF loading requires intensities, repnum, and id_values.")

            if abf_i >= len(id_values):
                raise ValueError("Not enough id_values provided for ABF files.")

            df = load_abf(
                file,
                intensities=intensities,
                id_value=id_values[abf_i],
                repnum=repnum
            )
            abf_i += 1

        elif suffix == ".csv":
            df = load_csv(file)
        elif suffix == ".tsv":
            df = load_tsv(file)
        else:
            raise ValueError(
                f"Error reading {file.name}. "
                "Suffix must be one of: .nwb, .abf, .csv, or .tsv"
            )

        file_ids = set(df["id"].unique())
        duplicated_ids = seen_ids.intersection(file_ids)

        if duplicated_ids: # fix this, maybe multiple slices per animal
            raise ValueError(
                f"Duplicate id(s) found across files: {sorted(duplicated_ids)}"
            )

        seen_ids.update(file_ids)
        recordings.append(df)

    if not recordings:
        raise ValueError("No files were loaded.")

    return pd.concat(recordings, ignore_index=True)

def save_results(filename):
    return
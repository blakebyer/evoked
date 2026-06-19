import pyabf
from pathlib import Path
import dataclasses
from evoked.base import RecordingData, RecordingResult
from pandera.typing import DataFrame
import pandera as pa
import pandas as pd
import numpy as np
import warnings
import json

"""
    Tabular data should look like:
    id  time  voltage  intensity  sweepNumber
    <str>  <float>  <float>  <int>  <int>  
"""
# TODO:
# Implement other file types like NWB, or make a neo loader

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

        try:
            if suffix == ".abf":
                if any(var is None for var in (intensities, repnum)):
                    raise ValueError("ABF loading requires intensities and repnum.")

                if id_values is None:
                    id_value = file.stem
                else:
                    if abf_i >= len(id_values):
                        raise ValueError("Not enough id_values provided for ABF files.")
                    id_value = id_values[abf_i]

                df = load_abf(
                    file,
                    intensities=intensities,
                    id_value=id_value,
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
                    "Suffix must be one of: .abf, .csv, or .tsv"
                )

            base_id = df["id"].iloc[0] # if we've seen an animal id before, append _1, _2, ... to the same animal ids
            matches = sum(1 for sid in seen_ids if sid == base_id or sid.startswith(f"{base_id}_"))
            
            if matches:
                df["id"] = f"{base_id}_{matches}"

            seen_ids.add(df["id"].iloc[0])
            recordings.append(df)
        except ValueError as e:
            # catch sweep errors and fail gracefully
            if "sweeps" in str(e):
                warnings.warn(
                    f"\nSkipping {file.name}: {str(e)}. "
                    f"This file will be omitted from quantification."
                )
                continue 
            else:
                raise e

    if not recordings:
        raise ValueError("No files were loaded.")

    return pd.concat(recordings, ignore_index=True)

# def save_results_xlsx(recording_result: RecordingResult, filename):
#     for 
#     return

def save_results_json(recording_result: RecordingResult, filepath: str):
    """Export recording result to JSON"""
    # Convert dataclasses to dicts, then handle dataframes and arrays on the fly
    data = dataclasses.asdict(recording_result)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=lambda x: x.to_dict(orient="records") if isinstance(x, pd.DataFrame) else (x.tolist() if isinstance(x, np.ndarray) else str(x)))
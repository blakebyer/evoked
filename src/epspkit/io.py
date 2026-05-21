import pyabf
from pathlib import Path
from epspkit.base import RecordingData
from pandera.typing import DataFrame
import pandera as pa
from pymatreader import read_mat
import pandas as pd

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

@pa.check_types
def load_abf(filename, intensities: list[int], repnum: int) -> DataFrame[RecordingData]:
    return None

@pa.check_types
def load_mat(filename, table_name: str) -> DataFrame[RecordingData]:
    data = read_mat(filename)
    return pd.DataFrame(data[table_name])

@pa.check_types
def load_csv(filename) -> DataFrame[RecordingData]:
    return pd.read_csv(filename, index_col=False)

@pa.check_types
def load_tsv(filename) -> DataFrame[RecordingData]:
    return pd.read_csv(filename, sep='\t', index_col=False)

@pa.check_types
def load_xlsx(filename) -> DataFrame[RecordingData]:
    return pd.read_excel(filename, index_col=False)

@pa.check_types
def load_bulk(
    filenames: list[Path],
    table_name: str | None = None,
    intensities: list[int] | None = None,
    repnum: int | None = None,
) -> DataFrame[RecordingData]:

    recordings = []
    seen_ids = set()

    for file in filenames:
        suffix = file.suffix.lower()

        if suffix == ".abf":
            if intensities is None or repnum is None:
                raise ValueError("ABF loading requires intensities and repnum.")
            df = load_abf(file, intensities=intensities, repnum=repnum)

        elif suffix == ".mat":
            if table_name is None:
                raise ValueError("MAT loading requires table_name.")
            df = load_mat(file, table_name=table_name)

        elif suffix == ".csv":
            df = load_csv(file)

        elif suffix == ".tsv":
            df = load_tsv(file)

        elif suffix == ".xlsx":
            df = load_xlsx(file)

        else:
            raise ValueError(
                f"Error reading {file.name}. "
                "Suffix must be one of: .abf, .mat, .csv, .tsv, or .xlsx"
            )

        file_ids = set(df["id"].unique())
        duplicated_ids = seen_ids.intersection(file_ids)

        if duplicated_ids:
            raise ValueError(
                f"Duplicate id(s) found across files: {sorted(duplicated_ids)}"
            )

        seen_ids.update(file_ids)
        recordings.append(df)

    if not recordings:
        raise ValueError("No files were loaded.")

    return pd.concat(recordings, ignore_index=True)
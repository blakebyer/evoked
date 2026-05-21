import pyabf
from pathlib import Path
from pymatreader import read_mat
import pandas as pd

def load_abf(filename):
    return None

def load_mat(filename):
    return None

def load_csv(filename):
    return None

def load_xlsx(filename):
    return None

def load_bulk(filenames: list[Path]):
    for file in filenames:
        if file.suffix == ".abf":
            load_abf(file)
        elif file.suffix == ".mat":
            load_mat(file)
        elif file.suffix == ".csv" or file.suffix == ".tsv":
            load_csv(file)
        elif file.suffix == ".xlsx":
            load_xlsx(file)
        else: 
            raise ValueError(f"Error reading {file.name}. Suffix must be one of: .abf, .mat, .csv, .tsv, or .xlsx")
    return None
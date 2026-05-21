from __future__ import annotations
import numpy as np
from epspkit.base import IntermediateResult, FeatureResult, FitResult, window_to_indices
from pandera.typing import DataFrame

def center_signal(signal: np.ndarray):
    signal_arr = np.asarray(signal, dtype=float).ravel()
    return signal_arr - float(np.mean(signal_arr))

def r2():
    return 1

def pearson_corr():
    return 0

def build_template(intermediate: DataFrame[IntermediateResult], template_window: tuple[float,float], intensities: list[int]): # probably need to rewrite to make generic to vectors then write super functions for the train/test paradigm
    filtered = intermediate[intermediate["intensity"].isin(intensities)]
    templates = []
    for intensity, group in filtered.groupby("intensity"):
        signal = group["voltage"]
        time = group["time"]
        start_idx, stop_idx = window_to_indices(time, template_window)
        snippet = signal[start_idx:stop_idx]
    
        if snippet.size < 3:
            raise ValueError("Snippet window must contain at least 3 samples.")
        
        templates.append(snippet)
    
    template = np.mean(np.vstack(templates), axis=0) # average snippets to create template
    
    center_idx = template.size // 2
    return template, center_idx

def fit_template(intermediate: DataFrame[IntermediateResult], search_window: tuple[float,float], center_idx: int, intensities: list[int]) -> DataFrame[FitResult]:
    filtered = intermediate[intermediate["intensity"].isin(intensities)]
    for intensity, group in filtered.groupby("intensity"):
        group[""] # need to build_template first
    
        signal_c = center_signal(signal)
        template_c = center_signal(template)

        start_idx, stop_idx = window_to_indices(signal, search_window)
        left = center_idx
        right = template.size - center_idx - 1
        first_center = start_idx + left
        last_center = stop_idx - right - 1


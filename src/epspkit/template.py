from __future__ import annotations
import numpy as np
from epspkit.base import IntermediateResult, FeatureResult, FitResult, window_to_indices
from pandera.typing import DataFrame

def center_signal(signal: np.ndarray):
    signal_arr = np.asarray(signal, dtype=float).ravel()
    return signal_arr - float(np.mean(signal_arr))

def fit_r2(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size or snippet_c.size < 2:
        return np.nan

def fit_nrmse(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)

def pearson_corr(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size or snippet_c.size < 2:
        return np.nan

    snippet_norm = float(np.linalg.norm(snippet_c))
    template_norm = float(np.linalg.norm(template_c))
    if snippet_norm <= 1e-20 or template_norm <= 1e-20:
        return np.nan

    return float(np.dot(snippet_c, template_c) / (snippet_norm * template_norm))

def build_template(x: np.ndarray, y: np.ndarray, template_window: tuple[float,float]):
    signal = np.asarray(y).ravel()
    start_idx, stop_idx = window_to_indices(x, template_window)
    template = signal[start_idx:stop_idx]
    if template.size < 3:
        raise ValueError("Template window must contain at least 3 samples.")
    center_idx = int(template.size // 2)
    return template, center_idx

def fit_template(intermediate: DataFrame[IntermediateResult], template_window: tuple[float,float], search_window: tuple[float,float], center_idx: int, intensities: list[int]) -> DataFrame[FitResult]:
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


from __future__ import annotations
import numpy as np
import pandas as pd
from epspkit.base import IntermediateResult, FitResult, window_to_indices
from pandera.typing import DataFrame

def center_signal(signal: np.ndarray):
    signal_arr = np.asarray(signal, dtype=float).ravel()
    return signal_arr - float(np.mean(signal_arr))

def estimate_scale(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size:
        raise ValueError("Snippet and template must have the same length.")
    denom = float(np.dot(template_c, template_c))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(snippet_c, template_c) / denom)

def estimate_r2(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size or snippet_c.size < 2:
        return np.nan
    scale = estimate_scale(snippet_c, template_c)
    pred = scale * template_c
    sse = float(np.sum((snippet_c - pred) ** 2))
    sst = float(np.sum(snippet_c ** 2))
    if sst <= 1e-20:
        return np.nan
    return float(1.0 - sse / sst)
    
def estimate_corr(snippet: np.ndarray, template: np.ndarray) -> float:
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

def fit_template(
    intermediate: DataFrame[IntermediateResult],
    template: np.ndarray,
    search_window: tuple[float, float],
    center_idx: int,
    intensities: list[int] | None = None
) -> DataFrame[FitResult]:

    if intensities is not None:
        intermediate = intermediate[intermediate["intensity"].isin(intensities)].copy()

    template = np.asarray(template, dtype=float).ravel()
    results = []

    left = center_idx
    right = template.size - center_idx - 1

    for (id_value, intensity), group in intermediate.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        start_idx, stop_idx = window_to_indices(time, search_window)

        first_center = start_idx + left
        last_center = stop_idx - right  # exclusive, like normal Python

        if last_center <= first_center:
            raise ValueError("Search window is too small for this template.")

        best_corr = -np.inf
        best_result = {
            "id": id_value,
            "intensity": intensity,
            "lag": np.nan,
            "scale": np.nan,
            "corr": np.nan,
            "r2": np.nan
        }

        for center in range(first_center, last_center):
            snippet = signal[center - left : center + right + 1]
            corr = estimate_corr(snippet, template)

            if np.isnan(corr) or corr <= best_corr:
                continue

            scale = estimate_scale(snippet, template)
            r2 = estimate_r2(snippet, template)
            lag = float((time[center] - time[first_center]) * 1000)

            best_corr = corr
            best_result = {
                "id": id_value,
                "intensity": intensity,
                "lag": lag,
                "scale": scale,
                "corr": float(corr),
                "r2": r2
            }

        results.append(best_result)

    return pd.DataFrame(results)
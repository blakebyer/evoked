from __future__ import annotations
import numpy as np
import pandas as pd
from epspkit.base import IntermediateResult, FitResult, FeatureResult, RecordingResult, window_to_indices
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

def build_template(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    template_intensities: list[int],
    slope_transform: bool = False,
) -> tuple[np.ndarray, bool]:
    """Builds a template and returns it alongside its slope_transform state."""
    template_data = intermediate[intermediate["intensity"].isin(template_intensities)]
    if template_data.empty:
        raise ValueError("No traces found for template_intensities.")

    template_snippets = []
    for _, group in template_data.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        template_start, template_stop = window_to_indices(time, template_window)
        template_snippets.append(signal[template_start:template_stop])

    template_array = np.mean(np.vstack(template_snippets), axis=0)
    return template_array, slope_transform


def fit_template(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    search_window: tuple[float, float],
    template_package: tuple[np.ndarray, bool],
) -> FeatureResult:
    """Fits a pre-built template package tuple: (template_array, slope_transform)"""
    template_arr, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    center_idx = int(template_arr.size // 2)
    left = center_idx
    right = template_arr.size - center_idx - 1

    results = []
    for (id_value, intensity), group in intermediate.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        start_idx, stop_idx = window_to_indices(time, search_window)
        first_center = start_idx + left
        last_center = stop_idx - right

        if last_center <= first_center:
            raise ValueError("Search window is too small for this template.")

        best_corr = -np.inf
        best_result = {
            "id": id_value, "intensity": intensity, "match_time": np.nan,
            "scale": np.nan, "corr": np.nan, "r2": np.nan,
        }

        for center in range(first_center, last_center):
            snippet = signal[center - left : center + right + 1]
            corr = estimate_corr(snippet, template_arr)

            if np.isnan(corr) or corr <= best_corr:
                continue

            scale = estimate_scale(snippet, template_arr)
            r2 = estimate_r2(snippet, template_arr)
            match_time = float(time[center] * 1000)

            best_corr = corr
            best_result = {
                "id": id_value, "intensity": intensity, "match_time": match_time,
                "scale": scale, "corr": float(corr), "r2": r2,
            }

        results.append(best_result)
    
    return FeatureResult(
        search_window=search_window,
        template_window=template_window,
        result=pd.DataFrame(results)
    )


def build_and_fit_template(
    train_df: DataFrame[IntermediateResult],
    test_df: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    search_window: tuple[float, float],
    template_intensities: list[int],
    slope_transform: bool = False,
) -> FeatureResult:
    """Builds a template from training data and fits it directly onto testing data."""
    template_package = build_template(
        intermediate=train_df,
        template_window=template_window,
        template_intensities=template_intensities,
        slope_transform=slope_transform
    )
    return fit_template(
        intermediate=test_df,
        template_window=template_window,
        search_window=search_window,
        template_package=template_package
    )
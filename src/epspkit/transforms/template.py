from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epspkit.core.context import RecordingContext
from epspkit.core.math import (
    window_to_indices,
    gradient,
)
from epspkit.features.base import apply_smoothing

def build_template(
    x: np.ndarray,
    y: np.ndarray,
    window_ms: tuple[float, float],
    center_idx: int | None = None,
    feature_name: str | None = None,
) -> tuple[np.ndarray, int]:
    signal = np.asarray(y, dtype=float).ravel()

    if feature_name is not None:
        feature_key = str(feature_name).lower()
        if feature_key == "epsp":
            signal = gradient(signal, x)
        elif feature_key in {"fiber_volley", "pop_spike"}:
            pass
        else:
            raise ValueError(
                f"Feature '{feature_name}' does not define a template signal representation."
            )

    start_idx, stop_idx = window_to_indices(x, window_ms)
    template = np.asarray(signal[start_idx:stop_idx], dtype=float).ravel().copy()
    if template.size < 3:
        raise ValueError("Template window must contain at least 3 samples.")

    if center_idx is None:
        center_idx = template.size // 2
    center_idx = int(center_idx)
    if center_idx < 0 or center_idx >= template.size:
        raise ValueError(
            f"template_center_idx={center_idx} is out of bounds "
            f"for template length {template.size}."
        )

    return template, center_idx


def center_signal(signal: np.ndarray) -> np.ndarray:
    signal_arr = np.asarray(signal, dtype=float).ravel()
    return signal_arr - float(np.mean(signal_arr))


def project_template(signal: np.ndarray, template: np.ndarray) -> float:
    """
    Return the least-squares scale for the centered model
        signal_c ≈ scale * template_c
    where *_c denotes mean-centered vectors.
    """
    signal_c = center_signal(signal)
    template_c = center_signal(template)
    if signal_c.size != template_c.size:
        raise ValueError("Signal and template must have the same length.")

    denom = float(np.dot(template_c, template_c))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(signal_c, template_c) / denom)


def pearson_corr(snippet: np.ndarray, template: np.ndarray) -> float:
    """
    Pearson correlation between snippet and template.
    This is cosine similarity of the centered vectors.
    """
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size or snippet_c.size < 2:
        return np.nan

    snippet_norm = float(np.linalg.norm(snippet_c))
    template_norm = float(np.linalg.norm(template_c))
    if snippet_norm <= 1e-20 or template_norm <= 1e-20:
        return np.nan

    return float(np.dot(snippet_c, template_c) / (snippet_norm * template_norm))


def template_fit_r2(
    snippet: np.ndarray,
    template: np.ndarray,
    scale: float,
) -> float:
    """
    R^2 for the centered fit
        snippet_c ≈ scale * template_c.

    This is equivalent to the ordinary linear-model R^2 for
        snippet ≈ scale * template + intercept
    but computed entirely in centered coordinates.
    """
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size or snippet_c.size < 2:
        return np.nan

    model_c = float(scale) * template_c
    ss_res = float(np.sum((snippet_c - model_c) ** 2))
    ss_tot = float(np.sum(snippet_c ** 2))
    if ss_tot <= 1e-20:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def match_template(
    signal: np.ndarray,
    start_idx: int,
    stop_idx: int,
    template: np.ndarray,
    center_idx: int | None = None,
) -> tuple[int | None, float, float, float]:
    """
    Slide the template across the allowed region and, at each candidate center,
    fit the centered snippet to the centered template with a single scale value.

    Returns
    -------
    best_idx : int | None
        Best center index in the signal.
    best_scale : float
        Least-squares scale for the centered fit.
    best_score : float
        Pearson correlation at the best alignment.
    best_r2 : float
        R^2 of the centered fit at the best alignment.
    """
    signal_arr = np.asarray(signal, dtype=float).ravel()
    template_arr = np.asarray(template, dtype=float).ravel()
    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    if center_idx is None:
        center_idx = template_arr.size // 2
    center_idx = int(center_idx)
    if center_idx < 0 or center_idx >= template_arr.size:
        raise ValueError(
            f"template_center_idx={center_idx} is out of bounds "
            f"for template length {template_arr.size}."
        )

    if stop_idx <= start_idx:
        return None, np.nan, np.nan, np.nan

    start_idx = max(0, int(start_idx))
    stop_idx = min(int(stop_idx), signal_arr.size)
    if stop_idx <= start_idx:
        return None, np.nan, np.nan, np.nan

    left = center_idx
    right = template_arr.size - center_idx - 1
    first_center = start_idx + left
    last_center = stop_idx - right - 1
    if last_center < first_center:
        return None, np.nan, np.nan, np.nan

    template_c = center_signal(template_arr)
    template_energy = float(np.dot(template_c, template_c))
    if template_energy <= 1e-20:
        return None, np.nan, np.nan, np.nan

    best_idx = None
    best_scale = np.nan
    best_score = -np.inf
    best_r2 = np.nan

    for candidate_idx in range(first_center, last_center + 1):
        snippet = signal_arr[candidate_idx - left:candidate_idx + right + 1]
        snippet_c = center_signal(snippet)
        snippet_energy = float(np.dot(snippet_c, snippet_c))
        if snippet_energy <= 1e-20:
            continue

        scale = float(np.dot(snippet_c, template_c) / template_energy)
        if scale <= 0: # polarity should not flip
            continue
        score = pearson_corr(snippet, template_arr)
        r2 = template_fit_r2(snippet, template_arr, scale)

        if np.isnan(score):
            continue

        if score > best_score:
            best_idx = candidate_idx
            best_scale = scale
            best_score = score
            best_r2 = r2

    if best_idx is None:
        return None, np.nan, np.nan, np.nan

    return best_idx, float(best_scale), float(best_score), float(best_r2)


def average_templates(templates: Sequence[np.ndarray]) -> np.ndarray:
    if not templates:
        raise ValueError("At least one template snippet is required.")

    template_arrays = [np.asarray(template, dtype=float).ravel() for template in templates]
    expected_size = template_arrays[0].size
    if expected_size < 3:
        raise ValueError("Template snippets must contain at least 3 samples.")

    for template in template_arrays[1:]:
        if template.size != expected_size:
            raise ValueError("All template snippets must have the same length.")

    return np.mean(np.vstack(template_arrays), axis=0)


def capture_template_window(
    context: RecordingContext,
    feature_name: str,
    metadata_key: str | None = None,
    center_idx: int | None = None,
) -> RecordingContext:
    """
    Capture the current context's feature-specific template window in metadata.

    This is mainly a debugging/introspection transform so template windows can be
    inspected from saved metadata without changing feature behavior.
    """
    pipeline_cfg = context.pipeline_cfg
    if pipeline_cfg is None or not pipeline_cfg.features:
        raise ValueError(
            "capture_template_window requires context.pipeline_cfg.features."
        )

    feature_cfg = next(
        (cfg for cfg in pipeline_cfg.features if cfg.name == feature_name),
        None,
    )
    if feature_cfg is None:
        raise ValueError(
            f"capture_template_window could not find feature '{feature_name}' in the pipeline config."
        )

    window_ms = feature_cfg.params.get("window_ms")
    if window_ms is None:
        raise ValueError(
            f"Feature '{feature_name}' requires 'window_ms' to capture a template window."
        )
    template_stim_intensities = feature_cfg.params.get("template_stim_intensities")

    frame = context.averaged
    if frame is None or frame.empty:
        raise ValueError(
            "capture_template_window requires averaged data. "
            "Run average_sweeps before this transform."
        )
    if "time" not in frame.columns:
        raise ValueError("capture_template_window requires a 'time' column.")
    if "mean" not in frame.columns:
        raise ValueError("capture_template_window requires a 'mean' column.")
    if template_stim_intensities is not None:
        if "stim_intensity" not in frame.columns:
            raise ValueError(
                f"Feature '{feature_name}' requested template_stim_intensities, "
                "but averaged data has no 'stim_intensity' column."
            )
        frame = frame.loc[frame["stim_intensity"].isin(template_stim_intensities)]
        if frame.empty:
            raise ValueError(
                f"capture_template_window found no rows for feature '{feature_name}' "
                f"using template_stim_intensities={list(template_stim_intensities)}."
            )

    snippets: list[np.ndarray] = []
    resolved_center = (
        center_idx
        if center_idx is not None
        else feature_cfg.params.get("template_center_idx")
    )

    if "stim_intensity" in frame.columns:
        groups = frame.groupby("stim_intensity", sort=False)
    else:
        groups = [(None, frame)]

    for _, g in groups:
        x = g["time"].to_numpy()
        y = apply_smoothing(
            g["mean"].to_numpy(),
            feature_cfg.smoothing,
            fs=context.fs,
        )

        snippet, resolved = build_template(
            x,
            y,
            window_ms,
            center_idx=resolved_center,
            feature_name=feature_name,
        )
        if resolved_center is None:
            resolved_center = resolved
        snippets.append(snippet)

    template = average_templates(snippets)
    metadata_key = metadata_key or str(feature_name)
    context.metadata.setdefault("templates", {})[metadata_key] = {
        "template": template.tolist(),
        "template_center_idx": int(resolved_center) if resolved_center is not None else None,
        "feature_name": feature_name,
        "template_stim_intensities": (
            list(template_stim_intensities)
            if template_stim_intensities is not None
            else None
        ),
        "window_ms": list(window_ms),
        "signal": "derivative" if str(feature_name).lower() == "epsp" else "voltage",
    }
    return context
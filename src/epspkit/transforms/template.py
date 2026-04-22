from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epspkit.core.context import RecordingContext
from epspkit.features.base import apply_smoothing


def window_to_indices(
    x: np.ndarray,
    window_ms: tuple[float, float],
) -> tuple[int, int]:
    t0, t1 = [value / 1000.0 for value in window_ms]
    return int(np.searchsorted(x, t0)), int(np.searchsorted(x, t1))


def build_template(
    x: np.ndarray,
    y: np.ndarray,
    window_ms: tuple[float, float],
    center_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    start_idx, stop_idx = window_to_indices(x, window_ms)
    template = np.asarray(y[start_idx:stop_idx], dtype=float).ravel().copy()
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


def feature_template_signal(
    feature_name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    feature_key = str(feature_name).lower()
    signal = np.asarray(y, dtype=float).ravel()

    if feature_key == "epsp":
        return np.gradient(signal, x)
    if feature_key in {"fiber_volley", "pop_spike"}:
        return signal

    raise ValueError(
        f"Feature '{feature_name}' does not define a template signal representation."
    )


def build_feature_template(
    feature_name: str,
    x: np.ndarray,
    y: np.ndarray,
    window_ms: tuple[float, float],
    center_idx: int | None = None,
) -> tuple[np.ndarray, int]:
    template_signal = feature_template_signal(feature_name, x, y)
    return build_template(x, template_signal, window_ms, center_idx=center_idx)


def project_template(signal: np.ndarray, template: np.ndarray) -> float:
    signal_arr = np.asarray(signal, dtype=float).ravel()
    template_arr = np.asarray(template, dtype=float).ravel()
    if signal_arr.size != template_arr.size:
        raise ValueError("Signal and template must have the same length.")

    denom = float(np.dot(template_arr, template_arr))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(signal_arr, template_arr) / denom)


def template_fit_r2(
    snippet: np.ndarray,
    template: np.ndarray,
    scale: float,
) -> float:
    snippet_arr = np.asarray(snippet, dtype=float).ravel()
    template_arr = np.asarray(template, dtype=float).ravel()
    if snippet_arr.size != template_arr.size or snippet_arr.size < 2:
        return np.nan

    model = float(scale) * template_arr
    ss_res = float(np.sum((snippet_arr - model) ** 2))
    ss_tot = float(np.sum((snippet_arr - np.mean(snippet_arr)) ** 2))
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

    template_norm = float(np.linalg.norm(template_arr))
    template_energy = float(np.dot(template_arr, template_arr))
    if template_energy <= 1e-20:
        return None, np.nan, np.nan, np.nan

    best_idx = None
    best_scale = np.nan
    best_score = -np.inf
    best_r2 = np.nan

    for candidate_idx in range(first_center, last_center + 1):
        snippet = signal_arr[candidate_idx - left:candidate_idx + right + 1]
        snippet_norm = float(np.linalg.norm(snippet))
        if snippet_norm <= 1e-20:
            continue

        numerator = float(np.dot(snippet, template_arr))
        scale = numerator / template_energy
        score = numerator / (snippet_norm * template_norm)
        r2 = template_fit_r2(snippet, template_arr, scale)
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

        snippet, resolved = build_feature_template(
            feature_name,
            x,
            y,
            window_ms,
            center_idx=resolved_center,
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

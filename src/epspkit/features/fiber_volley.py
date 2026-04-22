"""
Fiber volley feature detection implemented as an Analyzer subclass.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from epspkit.transforms.template import build_template, match_template, window_to_indices
import pandas as pd
import numpy as np


class FiberVolleyFeature(Feature):
    """Detect fiber volley extrema and amplitude for each stimulus intensity."""

    SUPPORTED_METHODS = {"peak", "template"}

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.method = self.resolve_method(default="peak", allowed=self.SUPPORTED_METHODS)
        self.window_ms = params.get("window_ms")  # ms
        if self.window_ms is None:
            raise ValueError("window_ms parameter must be specified for FiberVolleyFeature.")
        self.search_window_ms = params.get("search_window_ms", self.window_ms)
        if self.search_window_ms[1] <= self.search_window_ms[0]:
            raise ValueError("search_window_ms must have stop > start for FiberVolleyFeature.")
        self.height = params.get("height")  # mV
        if self.method == "peak" and self.height is None:
            raise ValueError("height parameter must be specified for FiberVolleyFeature.")
        self.template_r2_threshold = params.get(
            "template_r2_threshold",
            params.get("template_score_threshold"),
        )

    def run(self, context: RecordingContext) -> RecordingContext:
        fs = context.fs  # Hz
        fv_df = self.calculate(context.averaged, fs=fs)
        context.add_result(self.name, fv_df)
        return context

    def calculate(self, abf_df: pd.DataFrame, fs: float | None = None) -> pd.DataFrame:
        results = []
        template = None
        template_center_idx = self.config.params.get("template_center_idx")

        if self.method == "template" and self.config.params.get("template") is not None:
            template = np.asarray(self.config.params["template"], dtype=float).ravel()

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            # Smooth the raw mean once; avoid re-smoothing the pre-smoothed column.
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)
            start_idx, stop_idx = window_to_indices(x, self.search_window_ms)
            fv_idx = None
            fv_amp = fv_s = fv_v = np.nan
            template_score = np.nan
            template_scale = np.nan
            template_r2 = np.nan

            if self.method == "peak":
                y_w = y[start_idx:stop_idx]
                if y_w.size:
                    neg_peaks, neg_props = emath.find_peaks(-y_w, height=self.height)
                    if neg_peaks.size:
                        if "peak_heights" in neg_props:
                            fv_min_rel = int(neg_peaks[np.argmax(neg_props["peak_heights"])])
                        else:
                            fv_min_rel = int(neg_peaks[np.argmin(y_w[neg_peaks])])
                        fv_idx = start_idx + fv_min_rel
                        fv_amp = abs(float(y[fv_idx]))
            else:
                if template is None:
                    template_trace = self.config.params.get("template_trace")
                    if template_trace is None:
                        raise ValueError(
                            "Template matching for FiberVolleyFeature requires "
                            "'template' or 'template_trace'."
                        )
                    template_time = np.asarray(self.config.params.get("template_time", x), dtype=float)
                    template, template_center_idx = build_template(
                        template_time,
                        np.asarray(template_trace, dtype=float),
                        self.window_ms,
                        center_idx=template_center_idx,
                    )

                fv_idx, template_scale, template_score, template_r2 = match_template(
                    y,
                    start_idx,
                    stop_idx,
                    template,
                    center_idx=template_center_idx,
                )
                threshold = self.template_r2_threshold
                if fv_idx is None or (
                    threshold is not None and template_r2 < float(threshold)
                ):
                    fv_idx = None

            if fv_idx is not None:
                fv_s = float(x[fv_idx])
                fv_v = float(y[fv_idx])
                fv_amp = abs(fv_v)

            results.append({
                "stim_intensity": stim,
                "fv_amp": float(fv_amp) if np.isfinite(fv_amp) else np.nan,
                "fv_s": fv_s,
                "fv_v": fv_v,
                "template_score": float(template_score) if np.isfinite(template_score) else np.nan,
                "template_scale": float(template_scale) if np.isfinite(template_scale) else np.nan,
                "template_r2": float(template_r2) if np.isfinite(template_r2) else np.nan,
            })

        return pd.DataFrame(results)

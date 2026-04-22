"""
Module for detecting population spike features.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from epspkit.transforms.template import build_template, match_template, window_to_indices
import pandas as pd
import numpy as np

class PopSpikeFeature(Feature):
    """Detect population spike timing and amplitude for each stimulus intensity."""

    SUPPORTED_METHODS = {"peak", "template"}

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.method = self.resolve_method(default="peak", allowed=self.SUPPORTED_METHODS)
        self.window_ms = params.get("window_ms")
        if self.method == "template" and self.window_ms is None:
            raise ValueError("window_ms parameter must be specified for PopSpikeFeature in template mode.")
        self.search_window_ms = params.get("search_window_ms")
        if self.search_window_ms is None:
            raise ValueError("search_window_ms parameter must be specified for PopSpikeFeature.")
        if self.search_window_ms[1] <= self.search_window_ms[0]:
            raise ValueError("search_window_ms must have stop > start for PopSpikeFeature.")
        self.height = params.get("height")  # mV trough magnitude threshold
        if self.method == "peak" and self.height is None:
            raise ValueError("height parameter must be specified for PopSpikeFeature in peak mode.")
        self.template_r2_threshold = params.get(
            "template_r2_threshold",
            params.get("template_score_threshold"),
        )

    def run(self, context: RecordingContext) -> RecordingContext:
        fs = context.fs  # Hz
        epsp_df = context.get_result("epsp")
        if epsp_df is None or epsp_df.empty:
            raise ValueError("EPSPFeature result is required for PopSpikeFeature.")
        ps_df = self.calculate(context.averaged, epsp_df, fs=fs)
        context.add_result(self.name, ps_df)
        return context

    def calculate(
    self,
    abf_df: pd.DataFrame,
    epsp_df: pd.DataFrame,
    fs: float | None = None,
) -> pd.DataFrame:

        results = []
        template = None
        template_center_idx = self.config.params.get("template_center_idx")

        if self.method == "template" and self.config.params.get("template") is not None:
            template = np.asarray(self.config.params["template"], dtype=float).ravel()

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)

            epsp_row = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            t_s = epsp_row["epsp_s"]
            v_s = epsp_row["epsp_v"]

            i0, i1 = window_to_indices(x, self.search_window_ms)

            ps_idx = None
            template_scale = np.nan
            template_score = np.nan
            template_r2 = np.nan

            if self.method == "peak":
                y_w = y[i0:i1]
                trough_height = float(self.height)
                #if trough_height <= 0:
                    #raise #ValueError(
                        #"PopSpikeFeature peak-mode height must be positive when "
                        #"detecting negative-going spikes with find_peaks(-y)."
                    #)
                peaks, props = emath.find_peaks(y_w, height=trough_height)
                if peaks.size:
                    if "peak_heights" in props:
                        ps_rel = int(peaks[np.argmax(props["peak_heights"])])
                    else:
                        ps_rel = int(peaks[np.argmin(y_w[peaks])])
                    ps_idx = i0 + ps_rel
            else:
                if template is None:
                    template_trace = self.config.params.get("template_trace")
                    if template_trace is None:
                        raise ValueError(
                            "Template matching for PopSpikeFeature requires "
                            "'template' or 'template_trace'."
                        )
                    template_time = np.asarray(self.config.params.get("template_time", x), dtype=float)
                    template, template_center_idx = build_template(
                        template_time,
                        np.asarray(template_trace, dtype=float),
                        self.window_ms,
                        center_idx=template_center_idx,
                    )

                ps_idx, template_scale, template_score, template_r2 = match_template(
                    y,
                    i0,
                    i1,
                    template,
                    center_idx=template_center_idx,
                )
                threshold = self.template_r2_threshold
                if ps_idx is None or (
                    threshold is not None and template_r2 < float(threshold)
                ):
                    ps_idx = None

            ps_amp = ps_s = ps_v = np.nan

            if ps_idx is not None:
                t_p = x[ps_idx]
                v_p = y[ps_idx]

                # ---- 3. Find post-PS baseline anchor ----
                after = slice(ps_idx + 1, i1)
                if after.start < after.stop:
                    b_rel = np.argmin(y[after])

                    b_idx = after.start + b_rel
                    t_b = x[b_idx]
                    v_b = y[b_idx]

                    # ---- 4. Baseline line ----
                    if np.isfinite(t_s) and np.isfinite(v_s) and t_b != t_s:
                        m = (v_b - v_s) / (t_b - t_s)
                        c = v_s - m * t_s
                        v_base = m * t_p + c

                        ps_amp = abs(v_p - v_base)
                        ps_s, ps_v = t_p, v_p

            results.append({
                "stim_intensity": stim,
                "ps_amp": ps_amp,
                "ps_s": ps_s,
                "ps_v": ps_v,
                "template_score": float(template_score) if np.isfinite(template_score) else np.nan,
                "template_scale": float(template_scale) if np.isfinite(template_scale) else np.nan,
                "template_r2": float(template_r2) if np.isfinite(template_r2) else np.nan,
            })

        return pd.DataFrame(results)

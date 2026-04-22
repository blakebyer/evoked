"""
fEPSP slope feature extraction module.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from epspkit.transforms.template import build_template, match_template, window_to_indices
import pandas as pd
import numpy as np

class EPSPFeature(Feature):
    """
    Computes EPSP minima and slopes from averaged traces.
    """

    SUPPORTED_METHODS = {"peak", "template"}

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.method = self.resolve_method(default="peak", allowed=self.SUPPORTED_METHODS)
        self.window_ms = params.get("window_ms")  # ms
        if self.window_ms is None:
            raise ValueError("window_ms parameter must be specified for EPSPFeature.")
        self.search_window_ms = params.get("search_window_ms")
        if self.search_window_ms is None:
            raise ValueError("search_window_ms parameter must be specified for EPSPFeature.")
        if self.search_window_ms[1] <= self.search_window_ms[0]:
            raise ValueError("search_window_ms must have stop > start for EPSPFeature.")
        self.height = params.get("height")  # mV/ms, applied to the negative slope peak
        if self.method == "peak" and self.height is None:
            raise ValueError("height parameter must be specified for EPSPFeature in peak mode.")
        fit_distance = params.get("fit_distance")
        if fit_distance is None:
            raise ValueError("fit_distance parameter must be specified for EPSPFeature.")
        self.fit_distance = int(fit_distance)  # points
        self.template_r2_threshold = params.get(
            "template_r2_threshold",
            params.get("template_score_threshold"),
        )

    def run(self, context: RecordingContext) -> RecordingContext:
        """
        Run EPSP analysis and attach results to the RecordingContext.
        """
        fv_df = context.get_result("fiber_volley")
        fs = context.fs  # Hz
        epsp_df = self.calculate(context.averaged, fv_df, fs=fs)
        context.add_result(self.name, epsp_df)
        return context

    def calculate(
        self,
        abf_df: pd.DataFrame,
        fv_df: pd.DataFrame | None,
        fs: float | None = None,
    ) -> pd.DataFrame:
        results = []
        template = None
        template_center_idx = self.config.params.get("template_center_idx")

        if self.method == "template" and self.config.params.get("template") is not None:
            template = np.asarray(self.config.params["template"], dtype=float).ravel()

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            # Smooth the raw mean once; avoid re-smoothing the pre-smoothed column.
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)
            fv_amp = np.nan
            if fv_df is not None and (fv_df.stim_intensity == stim).any():
                fv_row = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
                fv_amp = float(fv_row["fv_amp"])

            if self.method == "template" and template is None:
                template_trace = self.config.params.get("template_trace")
                if template_trace is None:
                    raise ValueError(
                        "Template matching for EPSPFeature requires "
                        "'template' or 'template_trace'."
                    )
                template_time = np.asarray(self.config.params.get("template_time", x), dtype=float)
                template_source = emath.gradient(
                    np.asarray(template_trace, dtype=float),
                    template_time,
                )
                template, template_center_idx = build_template(
                    template_time,
                    template_source,
                    self.window_ms,
                    center_idx=template_center_idx,
                )

            epsp_idx = None
            slope_center_idx = None
            template_scale = np.nan
            template_score = np.nan
            template_r2 = np.nan
            start_idx, stop_idx = window_to_indices(x, self.search_window_ms)
            dy_full = emath.gradient(y, x)

            if stop_idx > start_idx:
                if self.method == "peak":
                    dy_w = dy_full[start_idx:stop_idx]
                    d2_w = emath.gradient(dy_w, x[start_idx:stop_idx])
                    if dy_w.size:
                        dy_min_rel = int(np.argmin(dy_w))
                        onset_rel = 0
                        if d2_w.size:
                            d2_prefix = d2_w[:dy_min_rel + 1]
                            d2_peaks, _ = emath.find_peaks(d2_prefix)
                            if d2_peaks.size:
                                positive_peaks = d2_peaks[d2_prefix[d2_peaks] > 0]
                                if positive_peaks.size:
                                    onset_rel = int(positive_peaks[-1])
                                else:
                                    onset_rel = int(d2_peaks[np.argmax(d2_prefix[d2_peaks])])
                            else:
                                onset_rel = int(np.argmax(d2_prefix))

                        slope_tail = dy_w[onset_rel:]
                        if slope_tail.size:
                            neg_slope_peaks, neg_slope_props = emath.find_peaks(
                                -slope_tail,
                                height=float(self.height) * 1000.0,
                            )
                            if neg_slope_peaks.size:
                                if "peak_heights" in neg_slope_props:
                                    slope_rel = int(
                                        neg_slope_peaks[np.argmax(neg_slope_props["peak_heights"])]
                                    )
                                else:
                                    slope_rel = int(neg_slope_peaks[np.argmin(slope_tail[neg_slope_peaks])])
                                slope_center_idx = start_idx + onset_rel + slope_rel
                else:
                    slope_center_idx, template_scale, template_score, template_r2 = match_template(
                        dy_full,
                        start_idx,
                        stop_idx,
                        template,
                        center_idx=template_center_idx,
                    )
                    threshold = self.template_r2_threshold
                    if slope_center_idx is None or (
                        threshold is not None and template_r2 < float(threshold)
                    ):
                        slope_center_idx = None

                trough_start_idx = slope_center_idx if slope_center_idx is not None else start_idx
                if stop_idx > trough_start_idx:
                    epsp_idx = trough_start_idx + int(np.argmin(y[trough_start_idx:stop_idx]))

            epsp_s = epsp_v = epsp_amp = np.nan
            if epsp_idx is not None:
                epsp_s = float(x[epsp_idx])
                epsp_v = float(y[epsp_idx])
                epsp_amp = abs(epsp_v)

            slope_mid_s = slope_mid_v = epsp_r2 = epsp_slope = np.nan
            if slope_center_idx is not None:
                slope_mid_s = float(x[slope_center_idx])
                slope_mid_v = float(y[slope_center_idx])

                i0 = max(0, slope_center_idx - self.fit_distance)
                i1 = min(len(y) - 1, slope_center_idx + self.fit_distance)
                if i1 > i0:
                    t_win = x[i0:i1 + 1] - x[slope_center_idx]
                    v_win = y[i0:i1 + 1]
                    fit_slope, _, epsp_r2 = emath.linear_fit(t_win, v_win)
                    fit_slope = float(abs(fit_slope) / 1000.0)
                    epsp_r2 = float(epsp_r2)
                    epsp_slope = fit_slope

            epsp_to_fv = np.nan
            if np.isfinite(fv_amp) and fv_amp > 0 and np.isfinite(epsp_slope):
                epsp_to_fv = float(epsp_slope / fv_amp)

            results.append({
                "stim_intensity": stim,
                "epsp_s": epsp_s,
                "epsp_v": epsp_v,
                "slope_mid_s": slope_mid_s,
                "slope_mid_v": slope_mid_v,
                "epsp_amp": epsp_amp,
                "epsp_to_fv": epsp_to_fv,
                "epsp_slope": epsp_slope,
                "epsp_r2": epsp_r2,
                "template_score": float(template_score) if np.isfinite(template_score) else np.nan,
                "template_scale": float(template_scale) if np.isfinite(template_scale) else np.nan,
                "template_r2": float(template_r2) if np.isfinite(template_r2) else np.nan,
            })

        return pd.DataFrame(results)

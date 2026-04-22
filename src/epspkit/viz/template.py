from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epspkit.core import math as emath
from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.transforms.template import window_to_indices
from epspkit.viz.base import Plot


class TemplatePlot(Plot):
    """
    Visualize template-based detections with raw-trace highlights and match overlays.
    """

    FEATURE_SPECS = {
        "fiber_volley": {
            "center_col": "fv_s",
            "value_col": "fv_v",
            "color": "crimson",
            "label": "Fiber volley",
            "source": "voltage",
            "unit": "mV",
            "display_scale": 1.0,
        },
        "epsp": {
            "center_col": "slope_mid_s",
            "value_col": "slope_mid_v",
            "color": "darkred",
            "label": "EPSP slope",
            "source": "derivative",
            "unit": "mV/ms",
            "display_scale": 1.0 / 1000.0,
        },
        "pop_spike": {
            "center_col": "ps_s",
            "value_col": "ps_v",
            "color": "darkorange",
            "label": "Population spike",
            "source": "voltage",
            "unit": "mV",
            "display_scale": 1.0,
        },
    }

    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map

    def _template_feature_cfgs(self, context: RecordingContext) -> dict[str, dict]:
        pipeline_cfg = context.pipeline_cfg
        if pipeline_cfg is None or not pipeline_cfg.features:
            return {}

        feature_cfgs: dict[str, dict] = {}
        for feature_cfg in pipeline_cfg.features:
            method = str(feature_cfg.params.get("method", "peak")).lower()
            if method != "template":
                continue
            if feature_cfg.name not in self.FEATURE_SPECS:
                continue
            template = feature_cfg.params.get("template")
            if template is None:
                continue
            feature_cfgs[feature_cfg.name] = feature_cfg.params
        return feature_cfgs

    def _extract_segment(
        self,
        x: np.ndarray,
        trace: np.ndarray,
        center_s: float,
        template: np.ndarray,
        center_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        center_sample = int(np.argmin(np.abs(x - center_s)))
        left = int(center_idx)
        right = int(template.size - center_idx - 1)
        start_idx = center_sample - left
        stop_idx = center_sample + right + 1
        if start_idx < 0 or stop_idx > trace.size:
            return None

        segment_x = x[start_idx:stop_idx]
        segment_y = trace[start_idx:stop_idx]
        if segment_y.size != template.size:
            return None
        return segment_x, segment_y, (segment_x - x[center_sample]) * 1000.0

    def _correlation_profile(
        self,
        trace: np.ndarray,
        start_idx: int,
        stop_idx: int,
        template: np.ndarray,
        center_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        trace_arr = np.asarray(trace, dtype=float).ravel()
        template_arr = np.asarray(template, dtype=float).ravel()
        if template_arr.size < 3:
            return np.array([], dtype=int), np.array([], dtype=float)

        start_idx = max(0, int(start_idx))
        stop_idx = min(int(stop_idx), trace_arr.size)
        if stop_idx <= start_idx:
            return np.array([], dtype=int), np.array([], dtype=float)

        left = int(center_idx)
        right = int(template_arr.size - center_idx - 1)
        first_center = start_idx + left
        last_center = stop_idx - right - 1
        if last_center < first_center:
            return np.array([], dtype=int), np.array([], dtype=float)

        template_norm = float(np.linalg.norm(template_arr))
        if template_norm <= 1e-20:
            return np.array([], dtype=int), np.array([], dtype=float)

        indices: list[int] = []
        scores: list[float] = []
        for candidate_idx in range(first_center, last_center + 1):
            snippet = trace_arr[candidate_idx - left:candidate_idx + right + 1]
            snippet_norm = float(np.linalg.norm(snippet))
            if snippet_norm <= 1e-20:
                continue

            numerator = float(np.dot(snippet, template_arr))
            score = numerator / (snippet_norm * template_norm)
            indices.append(candidate_idx)
            scores.append(score)

        return np.asarray(indices, dtype=int), np.asarray(scores, dtype=float)

    def _stim_order(self, context: RecordingContext) -> list[float]:
        abf_df = context.averaged
        stim_order = list(self.stim_intensities) or list(pd.unique(abf_df["stim_intensity"]))
        return [stim for stim in stim_order if (abf_df["stim_intensity"] == stim).any()]

    def _build_feature_figure(
        self,
        context: RecordingContext,
        feature_name: str,
        params: dict,
    ) -> plt.Figure:
        abf_df = context.averaged
        fs = context.fs
        stim_order = self._stim_order(context)
        if not stim_order:
            raise ValueError("TemplatePlot found no matching stim intensities to plot.")

        spec = self.FEATURE_SPECS[feature_name]
        group_cols = 2
        nrows = max(1, ceil(len(stim_order) / group_cols))
        ncols = 2 * group_cols
        width_ratios = [1.35, 1.15] * group_cols

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    squeeze=False,
                    figsize=(11.5, max(4.5, 3.2 * nrows)),
                    gridspec_kw={"width_ratios": width_ratios},
                )

                result_df = context.get_result(feature_name)
                pair_axes: list[tuple[plt.Axes, plt.Axes, float]] = []
                for plot_idx, stim in enumerate(stim_order):
                    row_idx = plot_idx // group_cols
                    group_idx = plot_idx % group_cols
                    ax_template = axes[row_idx, 2 * group_idx]
                    ax_corr = axes[row_idx, 2 * group_idx + 1]
                    pair_axes.append((ax_template, ax_corr, stim))
                    g = abf_df.loc[abf_df["stim_intensity"] == stim]
                    if g.empty:
                        ax_template.set_visible(False)
                        ax_corr.set_visible(False)
                        continue

                    x = g["time"].to_numpy()
                    y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)
                    dy = emath.gradient(y, x)
                    source_trace = y if spec["source"] == "voltage" else dy

                    rows = None
                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                        rows = result_df.loc[result_df["stim_intensity"] == stim]

                    template = np.asarray(params["template"], dtype=float).ravel()
                    center_idx = int(params.get("template_center_idx", template.size // 2))
                    color = spec["color"]
                    label = spec["label"]
                    unit = spec["unit"]
                    display_scale = float(spec["display_scale"])

                    start_idx, stop_idx = window_to_indices(
                        x,
                        params.get("search_window_ms", params["window_ms"]),
                    )
                    corr_idx, corr_scores = self._correlation_profile(
                        source_trace,
                        start_idx,
                        stop_idx,
                        template,
                        center_idx,
                    )

                    ax_template.axvline(0.0, color="0.75", linestyle="--", linewidth=1.0)
                    ax_template.grid(True, alpha=0.3)
                    ax_template.set_ylabel(unit)
                    if row_idx == nrows - 1:
                        ax_template.set_xlabel("Relative time (ms)")

                    ax_corr.axhline(0.0, color="0.8", linewidth=1.0)
                    ax_corr.grid(True, alpha=0.3)
                    ax_corr.set_ylabel("Corr.")
                    if row_idx == nrows - 1:
                        ax_corr.set_xlabel("Time (ms)")

                    if corr_idx.size:
                        corr_time_ms = x[corr_idx] * 1000.0
                        ax_corr.plot(
                            corr_time_ms,
                            corr_scores,
                            color=color,
                            linewidth=1.8,
                            alpha=0.9,
                        )

                    if rows is None or rows.empty:
                        ax_template.text(
                            0.5,
                            0.5,
                            "No result",
                            ha="center",
                            va="center",
                            transform=ax_template.transAxes,
                        )
                        if not corr_idx.size:
                            ax_corr.text(
                                0.5,
                                0.5,
                                "No correlation trace",
                                ha="center",
                                va="center",
                                transform=ax_corr.transAxes,
                            )
                        continue

                    row = rows.iloc[0]
                    center_col = spec["center_col"]
                    center_s = row.get(center_col)
                    template_score = row.get("template_score")
                    template_scale = row.get("template_scale")
                    template_r2 = row.get("template_r2")

                    if pd.isna(center_s):
                        ax_template.text(
                            0.5,
                            0.5,
                            "No accepted match",
                            ha="center",
                            va="center",
                            transform=ax_template.transAxes,
                        )
                        if not corr_idx.size:
                            ax_corr.text(
                                0.5,
                                0.5,
                                "No correlation trace",
                                ha="center",
                                va="center",
                                transform=ax_corr.transAxes,
                            )
                        continue

                    segment = self._extract_segment(x, source_trace, float(center_s), template, center_idx)
                    if segment is None:
                        continue

                    _, segment_y, rel_ms = segment
                    scale = float(template_scale) if pd.notna(template_scale) else 1.0
                    model_y = scale * template

                    ax_template.plot(
                        rel_ms,
                        display_scale * segment_y,
                        color="black",
                        linewidth=1.6,
                        alpha=0.9,
                        label=f"{label} snippet",
                    )
                    ax_template.plot(
                        rel_ms,
                        display_scale * model_y,
                        color=color,
                        linewidth=2.2,
                        alpha=0.9,
                        label=f"{label} template fit",
                    )
                    if pd.notna(template_r2):
                        if feature_name == "fiber_volley":
                            r2_x = 0.98
                            r2_ha = "right"
                        else:
                            r2_x = 0.02                            
                            r2_ha = "left"
                        ax_template.text(
                            r2_x,
                            0.98,
                            rf"$R^2$={float(template_r2):.2f}",
                            transform=ax_template.transAxes,
                            ha=r2_ha,
                            va="top",
                            color="black",
                        )
                    if feature_name == "fiber_volley":
                        legend_loc = "lower left"
                        legend_kwargs = {"loc": legend_loc}
                    elif feature_name == "epsp":
                        legend_loc = "upper left"
                        legend_kwargs = {"loc": legend_loc, "bbox_to_anchor": (0.0, 0.92)}
                    else:
                        legend_loc = "lower right"
                        legend_kwargs = {"loc": legend_loc}
                    ax_template.legend(fontsize=8, **legend_kwargs)

                    ax_corr.axvline(float(center_s) * 1000.0, color=color, linestyle="--", linewidth=1.2)
                    if pd.notna(template_score):
                        ax_corr.scatter(
                            [float(center_s) * 1000.0],
                            [float(template_score)],
                            color=color,
                            edgecolors="black",
                            linewidths=0.8,
                            s=40,
                            zorder=5,
                        )

                for empty_idx in range(len(stim_order), nrows * group_cols):
                    empty_row = empty_idx // group_cols
                    empty_group = empty_idx % group_cols
                    axes[empty_row, 2 * empty_group].set_visible(False)
                    axes[empty_row, 2 * empty_group + 1].set_visible(False)

                fig.suptitle(f"{spec['label']} Template Matching", fontsize=15, fontweight="bold")
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=2.0)
                for ax_template, ax_corr, stim in pair_axes:
                    if not ax_template.get_visible() or not ax_corr.get_visible():
                        continue
                    left_box = ax_template.get_position()
                    right_box = ax_corr.get_position()
                    x_mid = 0.5 * (left_box.x0 + right_box.x1)
                    y_top = max(left_box.y1, right_box.y1) + 0.008
                    fig.text(
                        x_mid,
                        y_top,
                        f"{stim} µA",
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        fontweight="bold",
                    )
                return fig

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        feature_cfgs = self._template_feature_cfgs(context)
        if not feature_cfgs:
            raise ValueError("TemplatePlot requires at least one feature using method='template'.")
        feature_name, params = next(iter(feature_cfgs.items()))
        return self._build_feature_figure(context, feature_name, params)

    def _resolve_feature_output_path(
        self,
        output_path: Path | str,
        feature_name: str,
        output_stem: str | None = None,
        ext: str = "png",
    ) -> Path:
        output = Path(output_path)
        stem = output_stem or "recording"
        filename = f"{stem}_{self.name}_{feature_name}.{ext}"
        if output.suffix:
            output_dir = output.parent
        else:
            output_dir = output
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def render(self, context: RecordingContext) -> None:
        feature_cfgs = self._template_feature_cfgs(context)
        if not feature_cfgs:
            raise ValueError("TemplatePlot requires at least one feature using method='template'.")
        for feature_name, params in feature_cfgs.items():
            fig = self._build_feature_figure(context, feature_name, params)
            try:
                plt.show()
            finally:
                plt.close(fig)

    def save(
        self,
        context: RecordingContext,
        output_path: Path | str,
        output_stem: str | None = None,
    ) -> Path:
        feature_cfgs = self._template_feature_cfgs(context)
        if not feature_cfgs:
            raise ValueError("TemplatePlot requires at least one feature using method='template'.")

        first_path: Path | None = None
        for feature_name, params in feature_cfgs.items():
            fig = self._build_feature_figure(context, feature_name, params)
            save_path = self._resolve_feature_output_path(
                output_path,
                feature_name,
                output_stem=output_stem,
            )
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            if first_path is None:
                first_path = save_path

        assert first_path is not None
        return first_path

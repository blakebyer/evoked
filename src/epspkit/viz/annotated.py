from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot

class AnnotatedPlot(Plot):
    """
    Plots averaged sweeps with feature annotations.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map

    def _get_epsp_fit_distance(self, context: RecordingContext) -> int | None:
        pipeline_cfg = context.pipeline_cfg
        if pipeline_cfg is None or not pipeline_cfg.features:
            return None
        for feature_cfg in pipeline_cfg.features:
            if feature_cfg.name == "epsp":
                fit_distance = feature_cfg.params.get("fit_distance")
                if fit_distance is None:
                    return None
                try:
                    return int(fit_distance)
                except (TypeError, ValueError):
                    return None
        return None

    def _get_pop_spike_search_window_ms(self, context: RecordingContext) -> tuple[float, float] | None:
        pipeline_cfg = context.pipeline_cfg
        if pipeline_cfg is None or not pipeline_cfg.features:
            return None
        for feature_cfg in pipeline_cfg.features:
            if feature_cfg.name == "pop_spike":
                search_window_ms = feature_cfg.params.get("search_window_ms")
                if search_window_ms is None:
                    return None
                try:
                    start_ms, stop_ms = search_window_ms
                    return float(start_ms), float(stop_ms)
                except (TypeError, ValueError):
                    return None
        return None

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        abf_df = context.averaged
        fs = context.fs
        fv_df = context.get_result("fiber_volley")
        epsp_df = context.get_result("epsp")
        ps_df = context.get_result("pop_spike")
        if not any(isinstance(df, pd.DataFrame) and not df.empty for df in [fv_df, epsp_df, ps_df]):
            raise ValueError(
                "AnnotatedPlot requires at least one feature result "
                "(e.g., fiber_volley, epsp, or pop_spike)."
            )

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):
                
                fig, ax = plt.subplots()
                used_labels: set[str] = set()

                fit_distance = self._get_epsp_fit_distance(context)
                ps_search_window_ms = self._get_pop_spike_search_window_ms(context)
                stim_order = list(self.stim_intensities) or list(pd.unique(abf_df["stim_intensity"]))
                cmap = plt.get_cmap(self.color_map)
                n_colors = max(len(stim_order), 1)

                for idx, stim in enumerate(stim_order):
                    g = abf_df.loc[abf_df["stim_intensity"] == stim]
                    if g.empty:
                        continue
                    color = cmap(idx / (n_colors - 1)) if n_colors > 1 else 'black'

                    if "mean" not in g.columns:
                        raise ValueError("Expected 'mean' column in averaged data.")

                    x = g["time"].to_numpy()
                    y = g["mean"].to_numpy()
                    y = self.apply_smoothing(y, fs=fs)

                    ax.plot(x, y, label=f"{stim} µA", color=color)
                    self.annotate_features(
                        ax,
                        stim,
                        fv_df,
                        epsp_df,
                        ps_df,
                        used_labels,
                        x=x,
                        y=y,
                        fit_distance=fit_distance,
                        ps_search_window_ms=ps_search_window_ms,
                    )

                ax.set_title('Processed Evoked Field Potential with Detected Features')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Response (mV)')
                ax.legend()
                ax.grid()
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
                fig.tight_layout()
                return fig

    def render(self, context: RecordingContext) -> None:
        """
        Render the sweep plot for the given context.
        """
        self._build_figure(context)
        plt.show()

    def save(
        self,
        context: RecordingContext,
        output_path: Path | str,
        output_stem: str | None = None,
    ) -> Path:
        fig = self._build_figure(context)
        save_path = self._resolve_output_path(
            context,
            output_path,
            output_stem=output_stem,
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def annotate_features(
        self,
        ax: plt.Axes,
        stim: float,
        fv_df: pd.DataFrame | None,
        epsp_df: pd.DataFrame | None,
        ps_df: pd.DataFrame | None,
        used_labels: set[str],
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        fit_distance: int | None = None,
        ps_search_window_ms: tuple[float, float] | None = None,
    ) -> None:
        fv_color = to_rgba("darkviolet", alpha=0.7)
        epsp_slope_color = to_rgba("firebrick", alpha=0.8)
        epsp_min_color = to_rgba("royalblue", alpha=0.8)
        ps_color = to_rgba("darkorange", alpha=0.8)
        epsp_s = epsp_v = None
        invalid_legend_colors = {None, "inherit", "auto"}

        def _resolve_legend_color(value: str | None, fallback: str) -> str:
            if value in invalid_legend_colors:
                return fallback
            try:
                to_rgba(value)
            except ValueError:
                return fallback
            return value

        legend_face = plt.rcParams.get("legend.facecolor", "white")
        legend_edge = plt.rcParams.get("legend.edgecolor", "black")
        legend_alpha = plt.rcParams.get("legend.framealpha", 1.0)
        facecolor = _resolve_legend_color(legend_face, ax.get_facecolor())
        edgecolor = _resolve_legend_color(legend_edge, "black")
        try:
            legend_alpha = float(legend_alpha) if legend_alpha is not None else 1.0
        except (TypeError, ValueError):
            legend_alpha = 1.0

        label_box = {
            "boxstyle": "round,pad=0.2",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "alpha": legend_alpha,
        }

        def _add_amplitude_annotation(
            x0: float,
            y0: float,
            label: str,
            color: tuple[float, ...],
            label_offset_pts: tuple[int, int] = (6, 0),
            arrow_zorder: int = 5,
        ) -> None:
            if not np.isfinite(x0) or not np.isfinite(y0) or y0 == 0:
                return
            ax.annotate(
                "",
                xy=(x0, y0),
                xytext=(x0, 0.0),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": color,
                    "linewidth": 1.5,
                    "alpha": 0.9,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=arrow_zorder,
            )
            if label not in used_labels:
                ax.annotate(
                    label,
                    xy=(x0, 0.5 * y0),
                    xytext=label_offset_pts,
                    textcoords="offset points",
                    color="black",
                    ha="left",
                    va="center",
                    zorder=7,
                    bbox=label_box.copy(),
                )
                used_labels.add(label)

        if isinstance(fv_df, pd.DataFrame) and not fv_df.empty and (fv_df.stim_intensity == stim).any():
            r = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
            fv_s, fv_v = r.get("fv_s"), r.get("fv_v")
            if pd.notna(fv_s) and pd.notna(fv_v):
                label = "Fiber Volley" if "Fiber Volley" not in used_labels else None
                ax.scatter(
                    [fv_s],
                    [fv_v],
                    s=70,
                    marker="v",
                    facecolors=fv_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                    label=label,
                )
                used_labels.add("Fiber Volley")
                _add_amplitude_annotation(
                    fv_s,
                    fv_v,
                    "Fiber volley amplitude",
                    fv_color,
                    label_offset_pts=(6, 12),
                    arrow_zorder=4,
                )
            else:
                fv_s, fv_v = r.get("fv_s"), r.get("fv_v")
                if pd.notna(fv_s) and pd.notna(fv_v):
                    label = "Fiber Volley" if "Fiber Volley" not in used_labels else None
                    ax.scatter(
                        [fv_s],
                        [fv_v],
                        s=70,
                        marker="v",
                        facecolors=fv_color,
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=5,
                        label=label,
                    )
                    used_labels.add("Fiber Volley")
                    _add_amplitude_annotation(
                        fv_s,
                        fv_v,
                        "Fiber volley amplitude",
                        fv_color,
                        label_offset_pts=(6, 0),
                        arrow_zorder=4,
                    )

        if isinstance(epsp_df, pd.DataFrame) and not epsp_df.empty and (epsp_df.stim_intensity == stim).any():
            r = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            slope_s, slope_v = r.get("slope_mid_s"), r.get("slope_mid_v")
            epsp_s, epsp_v = r.get("epsp_s"), r.get("epsp_v")
            if pd.notna(slope_s) and pd.notna(slope_v):
                label = "Max fEPSP slope" if "Max fEPSP slope" not in used_labels else None
                ax.scatter(
                    [slope_s],
                    [slope_v],
                    s=70,
                    marker="s",
                    facecolors=epsp_slope_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label=label,
                )
                used_labels.add("Max fEPSP slope")
                if (
                    x is not None
                    and y is not None
                    and fit_distance is not None
                    and fit_distance > 0
                ):
                    slope_center_idx = int(np.argmin(np.abs(x - slope_s)))
                    i0 = max(0, slope_center_idx - fit_distance)
                    i1 = min(len(y) - 1, slope_center_idx + fit_distance)
                    if i1 - i0 >= 1:
                        x_win = x[i0:i1 + 1]
                        t_win = x_win - x[slope_center_idx]
                        v_win = y[i0:i1 + 1]
                        m, b = np.polyfit(t_win, v_win, 1)
                        y_pred = m * t_win + b
                        label = "fEPSP slope fit" if "fEPSP slope fit" not in used_labels else None
                        ax.plot(
                            x_win,
                            y_pred,
                            color=epsp_slope_color,
                            linewidth=2.0,
                            alpha=0.9,
                            zorder=5,
                            label=label,
                        )
                        used_labels.add("fEPSP slope fit")
            if pd.notna(epsp_s) and pd.notna(epsp_v):
                label = "fEPSP" if "fEPSP" not in used_labels else None
                ax.scatter(
                    [epsp_s],
                    [epsp_v],
                    s=70,
                    marker="o",
                    facecolors=epsp_min_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label=label,
                )
                # used_labels.add("fEPSP")
                # _add_amplitude_annotation(
                #     epsp_s,
                #     epsp_v,
                #     "fEPSP amplitude",
                #     epsp_min_color,
                #     label_offset_pts=(6, 64),
                #     arrow_zorder=4,
                # )

        if isinstance(ps_df, pd.DataFrame) and not ps_df.empty and (ps_df.stim_intensity == stim).any():
            r = ps_df.loc[ps_df.stim_intensity == stim].iloc[0]
            ps_s, ps_v = r.get("ps_s"), r.get("ps_v")
            if pd.notna(ps_s) and pd.notna(ps_v):
                label = "Population Spike" if "Population Spike" not in used_labels else None
                ax.scatter(
                    [ps_s],
                    [ps_v],
                    s=70,
                    marker="d",
                    facecolors=ps_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=7,
                    label=label,
                )
                used_labels.add("Population Spike")
                if (
                    x is not None
                    and y is not None
                    and ps_search_window_ms is not None
                    and pd.notna(epsp_s)
                    and pd.notna(epsp_v)
                ):
                    search_start_ms, search_stop_ms = ps_search_window_ms
                    i0 = int(np.searchsorted(x, search_start_ms / 1000.0))
                    i1 = int(np.searchsorted(x, search_stop_ms / 1000.0))
                    i1 = min(i1, len(x))
                    ps_idx = int(np.argmin(np.abs(x - ps_s)))
                    ps_idx = max(ps_idx, i0)
                    after_start = ps_idx + 1
                    if after_start < i1:
                        b_rel = np.argmin(y[after_start:i1])
                        b_idx = after_start + b_rel
                        t_b = x[b_idx]
                        v_b = y[b_idx]
                        ax.plot(
                            [epsp_s, t_b],
                            [epsp_v, v_b],
                            color=ps_color,
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.8,
                            zorder=4,
                        )
                        denom = t_b - epsp_s
                        if denom != 0:
                            m = (v_b - epsp_v) / denom
                            c = epsp_v - m * epsp_s
                            v_base = m * ps_s + c
                            if np.isfinite(v_base):
                                ax.annotate(
                                    "",
                                    xy=(ps_s, ps_v),
                                    xytext=(ps_s, v_base),
                                    arrowprops={
                                        "arrowstyle": "<->",
                                        "color": ps_color,
                                    "linewidth": 1.5,
                                    "alpha": 0.9,
                                    "shrinkA": 0,
                                    "shrinkB": 0,
                                },
                                    zorder=4,
                                )
                                # ax.annotate(
                                #     "PS amplitude",
                                #     xy=(ps_s, 0.5 * (ps_v + v_base)),
                                #     xytext=(6, 0),
                                #     textcoords="offset points",
                                #     color="black",
                                #     ha="left",
                                #     va="center",
                                #     zorder=7,
                                #     bbox=label_box.copy(),
                                # )

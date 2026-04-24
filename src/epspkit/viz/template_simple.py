from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epspkit.core.math import gradient
from epspkit.core.context import RecordingContext
from epspkit.viz.template import TemplatePlot


class TemplateSimplePlot(TemplatePlot):
    """
    Simpler template plot: one signal panel per stimulus intensity with matched
    segments highlighted.
    """

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        abf_df = context.averaged
        fs = context.fs
        feature_cfgs = self._template_feature_cfgs(context)
        if not feature_cfgs:
            raise ValueError("TemplateSimplePlot requires at least one feature using method='template'.")

        stim_order = list(self.stim_intensities) or list(pd.unique(abf_df["stim_intensity"]))
        stim_order = [stim for stim in stim_order if (abf_df["stim_intensity"] == stim).any()]
        if not stim_order:
            raise ValueError("TemplateSimplePlot found no matching stim intensities to plot.")
        ncols = 2
        nrows = max(1, ceil(len(stim_order) / ncols))

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    squeeze=False,
                    figsize=(12, max(4.5, 3.0 * nrows)),
                    sharex=True,
                )

                legend_handles = None
                legend_labels = None

                for plot_idx, stim in enumerate(stim_order):
                    row_idx = plot_idx // ncols
                    col_idx = plot_idx % ncols
                    ax = axes[row_idx, col_idx]
                    g = abf_df.loc[abf_df["stim_intensity"] == stim]
                    if g.empty:
                        ax.set_visible(False)
                        continue

                    x = g["time"].to_numpy()
                    y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)
                    dy = gradient(y, x)

                    ax.plot(x, y, color="0.35", linewidth=1.8, label="Signal")

                    for feature_name, params in feature_cfgs.items():
                        result_df = context.get_result(feature_name)
                        if not isinstance(result_df, pd.DataFrame) or result_df.empty:
                            continue
                        rows = result_df.loc[result_df["stim_intensity"] == stim]
                        if rows.empty:
                            continue

                        row = rows.iloc[0]
                        spec = self.FEATURE_SPECS[feature_name]
                        center_s = row.get(spec["center_col"])
                        center_v = row.get(spec["value_col"])
                        if pd.isna(center_s):
                            continue

                        template = np.asarray(params["template"], dtype=float).ravel()
                        center_idx = int(params.get("template_center_idx", template.size // 2))
                        source_trace = y if spec["source"] == "voltage" else dy

                        signal_segment = self._extract_segment(x, y, float(center_s), template, center_idx)
                        if signal_segment is None:
                            continue

                        signal_x, signal_y, _ = signal_segment
                        ax.plot(
                            signal_x,
                            signal_y,
                            color=spec["color"],
                            linewidth=3.0,
                            alpha=0.95,
                            label=spec["label"],
                        )
                        if pd.notna(center_v):
                            ax.scatter(
                                [center_s],
                                [center_v],
                                color=spec["color"],
                                edgecolors="black",
                                linewidths=0.8,
                                s=50,
                                zorder=5,
                            )

                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel("Response (mV)")
                    ax.set_title(f"{stim} µA", fontsize=13, fontweight="bold")
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 1000:.0f}"))
                    if row_idx == 0 and col_idx == 0:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()
                    if row_idx == nrows - 1:
                        ax.set_xlabel("Time (ms)")

                for empty_idx in range(len(stim_order), nrows * ncols):
                    empty_row = empty_idx // ncols
                    empty_col = empty_idx % ncols
                    axes[empty_row, empty_col].set_visible(False)

                if legend_handles and legend_labels:
                    seen: set[str] = set()
                    uniq_handles = []
                    uniq_labels = []
                    for handle, label in zip(legend_handles, legend_labels):
                        if label in seen:
                            continue
                        seen.add(label)
                        uniq_handles.append(handle)
                        uniq_labels.append(label)
                    axes[0, 0].legend(uniq_handles, uniq_labels, loc="right", fontsize=8)

                fig.suptitle("Template-Matched Feature Detection", fontsize=15, fontweight="bold")
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
                return fig

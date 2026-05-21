from __future__ import annotations

from dataclasses import dataclass, field
import pandera.pandas as pa
from pandera.typing import DataFrame
import pandas as pd
import numpy as np
from epspkit.base import RecordingResult, IntermediateResult
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from matplotlib.lines import Line2D

# class FitResult(pa.DataFrameModel):
#     id: Series[str]
#     intensity: Series[int] # stimulus intensity
#     lag: Series[float] # difference in ms from center of template to center of best fit
#     scale: Series[float] # vertical scale
#     corr: Series[float] # pearson corr
#     r2: Series[float] # r^2

def plot_io_curve(recording_result: RecordingResult, features: list[str], intensities: list[int], rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(ncols=len(features))
        
        if len(features) == 1: 
            axes = [axes]

        cmap = mpl.colormaps["Set2"]
        
        for i, feature in enumerate(features):
            r_result = recording_result.results.get(feature)
            if r_result is None:
                continue
            
            rdata = r_result.result
            rdata = rdata[rdata['intensity'].isin(intensities)]

            stats = rdata.groupby('intensity')['scale'].agg(['mean', 'sem']).reset_index()
            color_val = cmap(i / max(1, len(features) - 1))

            axes[i].errorbar(
                stats['intensity'],
                stats['mean'],
                yerr=stats['sem'],
                fmt='-o',
                color=color_val,
                capsize=3
            )
            axes[i].set_ylabel('Scale')
            axes[i].set_xlabel('Intensity')
            axes[i].set_title(feature)
        
        fig.suptitle('I-O Curves')
        plt.tight_layout()
        plt.show()

def plot_trace(intermediate_result: DataFrame[IntermediateResult], recording_result: RecordingResult, features: list[str], intensities: list[int], id_value: str,annotated: bool = False, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        intermediate_result = intermediate_result[intermediate_result["id"] == id_value] # plot only one slice at a time
        fig, ax = plt.subplots()

        cmap = mpl.colormaps["viridis"]

        for i, intensity in enumerate(intensities):
            color_val = cmap(i / max(1, len(intensities) - 1)) if len(intensities) > 1 else 'black'
            idata = intermediate_result[intermediate_result["intensity"] == intensity]
            time = idata['time']
            voltage = idata['voltage']
            ax.plot(time, voltage, color=color_val, label=f"{intensity}")
            if annotated:
                feature_cmap = mpl.colormaps["Set2"]

                for j, feature in enumerate(features):
                    r_result = recording_result.results.get(feature)
                    if r_result is None:
                        continue

                    rdata = r_result.result
                    rdata = rdata[
                        (rdata["id"] == id_value) &
                        (rdata["intensity"] == intensity)
                    ]

                    if rdata.empty:
                        continue

                    feature_color = feature_cmap(j / max(1, len(features) - 1))
                    half_width = (r_result.template_window[1] - r_result.template_window[0]) / 2000

                    for mt in rdata["match_time"].to_numpy() / 1000:
                        mask = (time >= mt - half_width) & (time <= mt + half_width)
                        y = np.interp(mt, time, voltage)

                        ax.plot(time[mask], voltage[mask], color=feature_color, linewidth=2.5, zorder=5)
                        ax.scatter(mt, y, color=feature_color, edgecolors="black", zorder=6)
        
        trace_legend = ax.legend(
            title="Stimulus Intensity (µA)",
            loc="lower right"
        )
        ax.add_artist(trace_legend)

        if annotated:
            annotation_handles = [
                Line2D(
                    [0],
                    [0],
                    color=mpl.colormaps["Set2"](i / max(1, len(features) - 1)),
                    linewidth=3,
                    marker="o",
                    markeredgecolor="black",
                    label=feature,
                )
                for i, feature in enumerate(features)
            ]

            ax.legend(
                handles=annotation_handles,
                title="Features",
                loc="center right"
            )

        ax.set_title("Evoked Field Potential")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Response (mV)")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
        plt.tight_layout()
        plt.show()

def plot_fit(intermediate_result: DataFrame[IntermediateResult], recording_result: RecordingResult, intensities: list[int], rc_params: dict | None = None):
    with plt.rc_context(rc_params):
         return

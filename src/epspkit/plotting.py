from __future__ import annotations

from dataclasses import dataclass, field
import pandera.pandas as pa
from pandera.typing import DataFrame
import pandas as pd
import numpy as np
from epspkit.base import RecordingResult, IntermediateResult, window_to_indices
from epspkit.template import build_template, center_signal
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
        
        fig.suptitle('IO Curves')
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

def plot_fit(intermediate_result: DataFrame[IntermediateResult], recording_result: RecordingResult, features: list[str], intensity: int, id_value: str, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        idata = intermediate_result[
            (intermediate_result["id"] == id_value) &
            (intermediate_result["intensity"] == intensity)
            ] # plot only one slice and intensity at a time
        if idata.empty:
            return
        fig, axes = plt.subplots(nrows=len(features), ncols=2)

        cmap = mpl.colormaps["Set2"]

        for i, feature in enumerate(features):
            r_result = recording_result.results.get(feature)
            if r_result is None:
                continue
            color_val = cmap(i / max(1, len(features) - 1))
            
            search_window = r_result.search_window
            slope_transform = r_result.slope_transform

            rdata = r_result.result
            row = rdata[
                (rdata["id"] == id_value) &
                (rdata["intensity"] == intensity)
            ]
            if row.empty:
                continue

            corr_arr = row['corr_arr'].iloc[0]
            corr = row['corr'].iloc[0]
            scale = row['scale'].iloc[0]
            match_time_ms = row['match_time'].iloc[0]

            raw_template = r_result.template
            if raw_template is None:
                raise ValueError(f"No stored template found for {feature}.")

            t_vector = idata["time"].to_numpy()
            v_vector = idata["voltage"].to_numpy()

            signal = v_vector.copy()
            if slope_transform:
                signal = np.gradient(signal, t_vector)

            template_c = center_signal(raw_template)
            template_len = raw_template.size
            center_idx = template_len // 2
            left = center_idx
            right = template_len - center_idx - 1

            match_time_sec = match_time_ms / 1000.0
            match_center_idx = int(np.argmin(np.abs(t_vector - match_time_sec)))

            fit_start = match_center_idx - left
            fit_stop = match_center_idx + right + 1

            snippet = signal[fit_start:fit_stop]
            snippet_mean = np.mean(snippet)

            fitted = scale * template_c + snippet_mean
            fit_time_ms = t_vector[fit_start:fit_stop] * 1000

            s_start, s_stop = window_to_indices(t_vector, search_window)

            axes[i, 0].plot(
                t_vector[s_start:s_stop] * 1000,
                signal[s_start:s_stop],
                color="black",
                label=f"{feature} signal"
            )

            axes[i, 0].plot(
                fit_time_ms,
                fitted,
                color=color_val,
                linewidth=2.5,
                label=f"{feature} template fit"
            )

            fit_label = "mV/ms" if slope_transform else "mV"
            axes[i, 0].set_ylabel(fit_label)
            axes[i, 0].set_xlabel("Time (ms)")
            axes[i, 0].legend(loc="best")
            axes[i, 0].grid(alpha=0.3)

            # Reconstruct valid center times for corr_arr
            search_start, search_stop = window_to_indices(t_vector, search_window)
            first_center = search_start + left
            last_center = search_stop - right

            corr_times_ms = t_vector[first_center:last_center] * 1000

            if len(corr_times_ms) != len(corr_arr):
                n = min(len(corr_times_ms), len(corr_arr))
                corr_times_ms = corr_times_ms[:n]
                corr_arr = corr_arr[:n]

            axes[i, 1].plot(corr_times_ms, corr_arr, color=color_val)

            axes[i, 1].scatter(
                match_time_ms,
                corr,
                facecolor=color_val,
                edgecolor="black",
                zorder=5
            )

            axes[i, 1].set_xlabel("Time (ms)")
            axes[i, 1].set_ylabel("Corr.")
            axes[i, 1].grid(alpha=0.3)
        
        fig.suptitle(f"{intensity} µA")
        plt.tight_layout()
        plt.show()
                

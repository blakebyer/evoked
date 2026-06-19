from __future__ import annotations

from pandera.typing import DataFrame
import numpy as np
from epspkit.base import RecordingResult, IntermediateResult, window_to_indices
from epspkit.template import center_signal
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from matplotlib.lines import Line2D

def plot_io_curve(recording_result: RecordingResult, features: list[str], intensities: list[int], rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(ncols=len(features))
        
        if len(features) == 1: 
            axes = [axes]

        cmap = mpl.colormaps["Paired"]
        
        for i, feature in enumerate(features):
            r_result = recording_result.results[feature]
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
            axes[i].set_xticks(range(0,700,100))
            axes[i].grid(alpha=0.3)
        
        fig.suptitle('IO Curves')
        plt.tight_layout()

        return fig, axes

def plot_trace(intermediate_result: DataFrame[IntermediateResult], recording_result: RecordingResult, features: list[str], intensities: list[int], id_value: str,annotated: bool = False, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        intermediate_result = intermediate_result[intermediate_result["id"] == id_value] # plot only one slice at a time
        fig, ax = plt.subplots()

        cmap = mpl.colormaps["cividis"]

        for i, intensity in enumerate(intensities):
            color_val = cmap(i / max(1, len(intensities) - 1)) if len(intensities) > 1 else 'black'
            idata = intermediate_result[intermediate_result["intensity"] == intensity]
            time = idata['time']
            voltage = idata['voltage']
            ax.plot(time, voltage, color=color_val, label=f"{intensity}")
            if annotated:
                feature_cmap = mpl.colormaps["Paired"]

                for j, feature in enumerate(features):
                    r_result = recording_result.results[feature]
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

                    for mt in rdata["feature_time"].to_numpy() / 1000:
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
                    color=mpl.colormaps["Paired"](i / max(1, len(features) - 1)),
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

        return fig, ax

def plot_fit(
    intermediate_result: DataFrame[IntermediateResult],
    recording_result: RecordingResult,
    features: list[str],
    intensity: int,
    id_value: str,
    rc_params: dict | None = None,
):
    with plt.rc_context(rc_params):
        idata = intermediate_result[
            (intermediate_result["id"] == id_value) &
            (intermediate_result["intensity"] == intensity)
        ]

        if idata.empty:
            return None, None

        fig, axes = plt.subplots(
            nrows=len(features),
            ncols=2,
            squeeze=False,
            figsize=(10.5, 3.0 * len(features)),
            gridspec_kw={"width_ratios": [1.35, 1.15]},
        )

        cmap = mpl.colormaps["Paired"]

        t_vector = idata["time"].to_numpy()
        v_vector = idata["voltage"].to_numpy()

        for i, feature in enumerate(features):
            r_result = recording_result.results[feature]
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

            row = row.iloc[0]

            corr_arr = np.asarray(row["corr_arr"], dtype=float)
            corr = float(row["corr"])
            scale = float(row["scale"])
            r2 = float(row["r2"])
            feature_time_ms = float(row["feature_time"])
            feature_time_s = feature_time_ms / 1000.0

            template = getattr(r_result, "template", None)

            if template is None:
                raise ValueError(
                    f"{feature} has no stored template. "
                    "Store template_arr inside FeatureResultTemplate when fitting."
                )

            template = np.asarray(template, dtype=float).ravel()

            signal = v_vector.copy()
            if slope_transform:
                signal = np.gradient(signal, t_vector)

            center_idx = template.size // 2
            left = center_idx
            right = template.size - center_idx - 1

            center_sample = int(np.argmin(np.abs(t_vector - feature_time_s)))
            fit_start = center_sample - left
            fit_stop = center_sample + right + 1

            if fit_start < 0 or fit_stop > signal.size:
                continue

            snippet = signal[fit_start:fit_stop]

            if snippet.size != template.size:
                continue

            rel_time_ms = (t_vector[fit_start:fit_stop] - t_vector[center_sample]) * 1000.0

            template_centered = center_signal(template)
            fitted = scale * template_centered + np.mean(snippet)

            axes[i, 0].axvline(0.0, color="0.75", linestyle="--", linewidth=1.0)

            axes[i, 0].plot(
                rel_time_ms,
                snippet,
                color="black",
                linewidth=1.6,
                label=f"{feature} snippet",
            )

            axes[i, 0].plot(
                rel_time_ms,
                fitted,
                color=color_val,
                linewidth=2.2,
                label=f"{feature} template fit",
            )

            axes[i, 0].text(
                0.02,
                0.98,
                rf"$R^2$={r2:.2f}",
                transform=axes[i, 0].transAxes,
                ha="left",
                va="top",
                color="black",
            )

            y_label = "mV/ms" if slope_transform else "mV"
            axes[i, 0].set_ylabel(y_label)
            axes[i, 0].set_xlabel("Relative time (ms)")
            axes[i, 0].legend(loc="best")
            axes[i, 0].grid(alpha=0.3)

            s_start, s_stop = window_to_indices(t_vector, search_window)

            first_center = s_start + left
            last_center = s_stop - right

            corr_time_ms = t_vector[first_center:last_center] * 1000.0

            # Defensive length match
            n = min(len(corr_time_ms), len(corr_arr))
            corr_time_ms = corr_time_ms[:n]
            corr_arr_plot = corr_arr[:n]

            axes[i, 1].axhline(0.0, color="0.8", linewidth=1.0)

            axes[i, 1].plot(
                corr_time_ms,
                corr_arr_plot,
                color=color_val,
                linewidth=1.8,
            )

            axes[i, 1].axvline(
                feature_time_ms,
                color=color_val,
                linestyle="--",
                linewidth=1.2,
            )

            axes[i, 1].scatter(
                [feature_time_ms],
                [corr],
                facecolor=color_val,
                edgecolor="black",
                linewidth=0.8,
                s=40,
                zorder=5,
            )

            axes[i, 1].set_ylabel("Corr.")
            axes[i, 1].set_xlabel("Time (ms)")
            axes[i, 1].grid(alpha=0.3)

        fig.suptitle(f"{intensity} µA", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        return fig, axes

def plot_detected(recording_result: RecordingResult, feature: str, rc_params: dict | None = None):
    with plt.rc_context(rc_params):

        r_result = recording_result.results.get(feature)
        detection = r_result.result
        
        plot_df = (
            detection
            .groupby("intensity", as_index=False)
            .agg(
                percent_detected=("detected", lambda x: x.mean() * 100)
            )
        )
        
        
        fig, ax = plt.subplots()

        ax.plot(plot_df["intensity"], plot_df["percent_detected"],
                marker="o", label="Template")
        ax.set_xlabel("Stimulus Intensity (µA)")
        ax.set_ylabel("Detected (%)")
        ax.set_ylim(0, 105)
        ax.legend(frameon=False)
        fig.suptitle(f"{feature} Detected")
        plt.tight_layout()

        return fig, ax

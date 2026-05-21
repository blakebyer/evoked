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

# class FitResult(pa.DataFrameModel):
#     id: Series[str]
#     intensity: Series[int] # stimulus intensity
#     lag: Series[float] # difference in ms from center of template to center of best fit
#     scale: Series[float] # vertical scale
#     corr: Series[float] # pearson corr
#     r2: Series[float] # r^2

def plot_io_curve(recording_result: RecordingResult, features: list[str], intensities: list[int], rc_params: dict):
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(ncols=len(features), sharey=True)
        
        if len(features) == 1: 
            axes = [axes]

        cmap_name = mpl.rcParams.get('image.cmap', 'plasma')
        cmap = mpl.colormaps[cmap_name]
        
        for i, feature in enumerate(features):
            f_result = recording_result.results.get(feature)
            if f_result is None:
                continue
            
            rdata = f_result.result
            fdata = rdata[rdata['intensity'].isin(intensities)]

            stats = fdata.groupby('intensity')[feature].agg(['mean', 'sem']).reset_index()
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
            axes[i].set_title(feature.upper())
        
        fig.suptitle('I-O Curves')
        plt.tight_layout()
        plt.show()

def plot_trace(intermediate_result: DataFrame[IntermediateResult], intensities: list[int], rc_params: dict):
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots()

        cmap_name = mpl.rcParams.get('image.cmap', 'viridis')
        cmap = mpl.colormaps[cmap_name]

        for i, intensity in enumerate(intensities):
            color_val = cmap(i / max(1, len(intensities) - 1)) if len(intensities) > 1 else 'black'
            fdata = intermediate_result[intermediate_result['intensity'].isin(intensities)]
            ax.plot(fdata['time'], fdata['voltage'], color=color_val, label=f"{intensity}")
        
        ax.set_title('Evoked Field Potential')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Response (mV)')
        ax.legend(title='Stimulus Intensity (µA)')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
        plt.tight_layout()
        plt.show()

def plot_fit(intermediate_result: DataFrame[IntermediateResult], recording_result: RecordingResult, intensities: list[int], rc_params: dict):
    with plt.rc_context(rc_params):
        return 

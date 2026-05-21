from __future__ import annotations

from epspkit.base import RecordingData, IntermediateResult, window_to_indices
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d

def baseline_correct(recording: DataFrame[RecordingData], baseline_window: tuple[float,float]) -> DataFrame[RecordingData]:
    corrected = []
    for intensity, group in recording.groupby(["intensity","sweepNumber"],group_keys=False):
        time = group["time"].to_numpy()
        voltage = group["voltage"].to_numpy()
        start_idx, stop_idx = window_to_indices(time, baseline_window)
        baseline = np.mean(voltage[start_idx:stop_idx])
        group["voltage"] = group["voltage"] - baseline
        
        corrected.append(group)

    return pd.concat(corrected, ignore_index=True)

def remove_stim_artifact(recording: DataFrame[RecordingData], artifact_window: tuple[float, float], artifact: str = "template") -> DataFrame[RecordingData]:
    removed = []
    for intensity, group in recording.groupby(["intensity","sweepNumber"],group_keys=False):
        time = group["time"].to_numpy()
        voltage = group["voltage"].to_numpy()
        start_idx, stop_idx = window_to_indices(time, artifact_window)
        if artifact == "zero":
           voltage[start_idx:stop_idx] = 0
        elif artifact == "interp":
            x = [time[start_idx - 1], time[stop_idx]]
            y = [voltage[start_idx - 1], voltage[stop_idx]]

            voltage[start_idx:stop_idx] = np.interp(
                time[start_idx:stop_idx],
                x,
                y
            )
        elif artifact == "template":
            template = build_template(voltage) # it needs to stack them this isn't going to work as written
        else:
            raise ValueError("Artifact removal method must be one of: zero, interp (interpolation), or template (template-subtract)") 
        
        group["voltage"] = voltage

        removed.append(group)

    return pd.concat(removed, ignore_index=True)

def average_traces(
    recording: DataFrame[RecordingData]
) -> DataFrame[IntermediateResult]:
    averaged = []

    for intensity, group in recording.groupby("intensity"):
        traces = []
        for _, sweep in group.groupby("sweepNumber"):
            traces.append(sweep["voltage"].to_numpy())

        average_voltage = np.mean(np.vstack(traces), axis=0)
        time = group[group["sweepNumber"] == group["sweepNumber"].iloc[0]]["time"].to_numpy()

        avg_df = pd.DataFrame({
            "time": time,
            "voltage": average_voltage,
            "intensity": intensity
        })

        averaged.append(avg_df)

    return pd.concat(averaged, ignore_index=True)

def apply_smoothing(intermediate: DataFrame[IntermediateResult], smoothing: str, 
                    smoothing_params: dict
) -> DataFrame[IntermediateResult]:
    smoothed = []
    for intensity, group in intermediate.groupby(["intensity"],group_keys=False):
        time = group["time"].to_numpy()
        voltage = group["voltage"].to_numpy()
        if smoothing == "none":
            return intermediate
        elif smoothing == "uniform":
            size = smoothing_params.get("size")
            smoothed_voltage = uniform_filter1d(voltage,size=size,mode="nearest")
        elif smoothing == "savgol":
            polyorder = smoothing_params.get("polyorder")
            window_length = smoothing_params.get("window_length")
            if window_length % 2 == 0:
                window_length += 1
            smoothed_voltage = savgol_filter(voltage, window_length=window_length, polyorder=polyorder)
        elif smoothing == "butter":
            def butter_lowpass(y: np.ndarray, cutoff: float, fs: float, order: int):
                b, a = butter(order, cutoff, btype='low', fs=fs)
                return filtfilt(b, a, y)
            cutoff = smoothing_params.get("cutoff")
            order = smoothing_params.get("order")
            fs = float(1 / (time[1] - time[0])) # fs = 1/dt
            smoothed_voltage = butter_lowpass(voltage, cutoff=cutoff, fs=fs, order=order)
        else:
            raise ValueError("Smoothing method must be one of: none, savgol, or butter.")
        
        group["voltage"] = smoothed_voltage
        smoothed.append(group)
        
    return pd.concat(smoothed, ignore_index=True)

def preprocess( # super function
        recording: DataFrame[RecordingData], 
        baseline_window: tuple[float,float], 
        artifact_window: tuple[float,float], 
        artifact="template", 
        smoothing="savgol", 
        smoothing_params: dict = {
                            "size":7, # size for uniform filter
                            "window_length":15, # window_length and polyorder for savitzky-golay
                            "polyorder": 3, 
                            "cutoff":2000.0, # cutoff and order for butterworth lowpass
                            "order":3,
                    }
) -> DataFrame[IntermediateResult]:
    corrected = baseline_correct(recording, baseline_window=baseline_window)
    removed = remove_stim_artifact(corrected, artifact_window=artifact_window, artifact=artifact)
    averaged = average_traces(removed)
    smoothed = apply_smoothing(averaged, smoothing=smoothing, smoothing_params=smoothing_params)
    return smoothed
from __future__ import annotations

from evoked.base import RecordingData, IntermediateResult, PreprocessParams, window_to_indices
from evoked.ols import estimate_scale_ols
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d

def baseline_correct(recording: DataFrame[RecordingData], baseline_window: tuple[float,float]) -> DataFrame[RecordingData]:
    corrected = []
    for _, group in recording.groupby(["id","intensity","sweepNumber"],group_keys=False, sort=False):
        group = group.copy()
        time = group["time"].to_numpy()
        voltage = group["voltage"].to_numpy()
        start_idx, stop_idx = window_to_indices(time, baseline_window)
        baseline = np.mean(voltage[start_idx:stop_idx])
        group["voltage"] = group["voltage"] - baseline
        
        corrected.append(group)

    return pd.concat(corrected, ignore_index=True)

def remove_stim_artifact(
    recording: DataFrame[RecordingData],
    artifact_window: tuple[float, float],
    artifact: str
) -> DataFrame[RecordingData]:
    if artifact not in ["none","zero", "interp", "template"]:
        raise ValueError("Artifact removal method must be one of: none, zero, interp, or template.")

    output_df = recording.copy()
    removed = []

    if artifact == "none":
        return recording

    elif artifact in ["zero", "interp"]:
        for _, group in output_df.groupby(["id", "intensity", "sweepNumber"], group_keys=False, sort=False):
            group = group.copy()
            time = group["time"].to_numpy()
            voltage = group["voltage"].to_numpy(copy=True)

            start_idx, stop_idx = window_to_indices(time, artifact_window)

            if artifact == "zero":
                voltage[start_idx:stop_idx] = 0.0
            else:
                voltage[start_idx:stop_idx] = np.interp(
                    time[start_idx:stop_idx],
                    [time[start_idx - 1], time[stop_idx]],
                    [voltage[start_idx - 1], voltage[stop_idx]],
                )

            group["voltage"] = voltage
            removed.append(group)

        return pd.concat(removed, ignore_index=True)

    for _, group in output_df.groupby(["id", "intensity"], group_keys=False, sort=False):
        sweeps = [g.copy() for _, g in group.groupby("sweepNumber", sort=False)]

        time = sweeps[0]["time"].to_numpy()
        start_idx, stop_idx = window_to_indices(time, artifact_window)

        snippets = np.array([
            sweep["voltage"].to_numpy()[start_idx:stop_idx]
            for sweep in sweeps
        ])

        artifact_template = np.mean(snippets, axis=0)

        for sweep, snippet in zip(sweeps, snippets):
            voltage = sweep["voltage"].to_numpy(copy=True)
            scale = estimate_scale_ols(snippet, artifact_template)

            if not np.isnan(scale):
                voltage[start_idx:stop_idx] = snippet - scale * artifact_template

            sweep["voltage"] = voltage
            removed.append(sweep)

    return pd.concat(removed, ignore_index=True)

def average_traces(
    recording: DataFrame[RecordingData]
) -> DataFrame[IntermediateResult]:
    averaged = []

    for (intensity,id_value), group in recording.groupby(["intensity","id"], group_keys=False, sort=False):
        traces = []
        for _, sweep in group.groupby("sweepNumber", sort=False):
            traces.append(sweep["voltage"].to_numpy())

        average_voltage = np.mean(np.vstack(traces), axis=0)
        time = group[group["sweepNumber"] == group["sweepNumber"].iloc[0]]["time"].to_numpy()

        avg_df = pd.DataFrame({
            "id": id_value,
            "time": time,
            "voltage": average_voltage,
            "intensity": intensity
        })

        averaged.append(avg_df)

    return pd.concat(averaged, ignore_index=True)

def apply_smoothing(intermediate: DataFrame[IntermediateResult], smoothing: str,
                    smoothing_params: dict,
) -> DataFrame[IntermediateResult]:
    if smoothing == "none":
            return intermediate
    
    smoothed = []
    for _, group in intermediate.groupby(["intensity","id"],group_keys=False, sort=False):
        group = group.copy()
        time = group["time"].to_numpy()
        voltage = group["voltage"].to_numpy()

        if smoothing == "uniform":
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
            fs = float(1.0 / np.mean(np.diff(time))) # fs = 1/dt
            smoothed_voltage = butter_lowpass(voltage, cutoff=cutoff, fs=fs, order=order)
        else:
            raise ValueError("Smoothing method must be one of: none, uniform, savgol, or butter.")
        
        group["voltage"] = smoothed_voltage
        smoothed.append(group)
        
    return pd.concat(smoothed, ignore_index=True)

def preprocess( # super function
        recording: DataFrame[RecordingData], 
        params: PreprocessParams
) -> DataFrame[IntermediateResult]:
    corrected = baseline_correct(recording, baseline_window=params.baseline_window)
    removed = remove_stim_artifact(corrected, artifact_window=params.artifact_window, artifact=params.artifact)
    averaged = average_traces(removed)
    smoothed = apply_smoothing(averaged, smoothing=params.smoothing, smoothing_params=params.smoothing_params)
    return smoothed
from __future__ import annotations
import numpy as np
import pandas as pd
from epspkit.base import IntermediateResult, SNRResult, FeatureResultSNR, RecordingResult, window_to_indices
from pandera.typing import DataFrame

def snr_detect(test_df: DataFrame[IntermediateResult], 
               search_window: tuple[float, float],
               noise_window: tuple[float,float],
               snr_threshold: float = 3.0,
               polarity: str = "positive",
               slope_transform: bool = False)-> FeatureResultSNR:
    """
    Peak/slope detection with SNR vs baseline noise
    Arguments:
        test_df: pd.DataFrame preprocessed data
        search_window: tuple[float,float] where to search for feature
        baseline_window: tuple[float,float] baseline noise
        polarity: str direction of peak
        slope_transform: bool to search for slope instead of amplitude
    """
    results = []
    for (id_value, intensity), group in test_df.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        search_start, search_stop = window_to_indices(time, search_window)
        noise_start, noise_stop = window_to_indices(time, noise_window)

        snippet = signal[search_start:search_stop]
        noise = signal[noise_start:noise_stop]


        noise_sd = np.std(noise, ddof=1)

        noise_mean = np.mean(noise)
        noise_sd = np.std(noise, ddof=1)

        if noise_sd <= 0 or not np.isfinite(noise_sd):
            continue

        centered = snippet - noise_mean

        if polarity == "negative":
            local_idx = np.argmin(centered)
        elif polarity == "positive":
            local_idx = np.argmax(centered)
        elif polarity == "absolute":
            local_idx = np.argmax(np.abs(centered))
        else:
            raise ValueError("polarity must be 'negative', 'positive', or 'absolute'")

        peak_idx = search_start + local_idx
        value = signal[peak_idx]
        feature_time = time[peak_idx]
        amplitude = abs(value - noise_mean)
        snr = amplitude / noise_sd

        detected = np.isfinite(snr) and snr >= snr_threshold

        results.append({
            "id": id_value,
            "intensity": intensity,
            "feature_time":feature_time,
            "value": amplitude,
            "noise_sd":noise_sd,
            "snr":snr,
            "detected": detected
        })
    
    return FeatureResultSNR(
        search_window=search_window,
        noise_window=noise_window,
        slope_transform=slope_transform,
        result=pd.DataFrame(results)
    )
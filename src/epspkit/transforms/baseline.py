from __future__ import annotations

import numpy as np
import pandas as pd
from epspkit.core.context import RecordingContext
from epspkit.core.math import window_to_indices

def baseline_correction(
    context: RecordingContext,
    baseline_window_ms: tuple[float, float] = (0.0, 0.1)
) -> pd.DataFrame:
    """
    Apply baseline correction to tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    baseline_window_ms
        Time window (start_ms, end_ms) in milliseconds to use for baseline calculation.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with baseline-corrected voltage values.
    """
    tidy_df = context.tidy
    def correct_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("time")
        x = g["time"].to_numpy()
        y = g["voltage"].to_numpy()

        # time-based indices
        start_idx, stop_idx = window_to_indices(x, baseline_window_ms)

        baseline = np.mean(y[start_idx:stop_idx])
        g["voltage"] = g["voltage"] - baseline
        return g

    context.tidy = tidy_df.groupby(["stim_intensity", "sweep_number"], group_keys=False)[
            ["stim_intensity", "sweep_number", "time", "voltage"]
        ].apply(correct_group)
    return context
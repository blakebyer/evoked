from __future__ import annotations
import numpy as np
import pandas as pd
from epspkit.core.context import RecordingContext
from epspkit.core.math import window_to_indices
from epspkit.transforms.template import build_template, project_template

def average_sweeps(context: RecordingContext) -> RecordingContext:
    tidy = context.tidy
    if tidy is None or tidy.empty:
        context.averaged = tidy.copy()
        return context

    group_cols = ["stim_intensity", "time"]
    missing = [col for col in group_cols if col not in tidy.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns for averaging: {missing_str}")

    if "mean" in tidy.columns:
        context.averaged = (
            tidy.sort_values(group_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        return context

    if "voltage" not in tidy.columns:
        raise ValueError("Expected 'voltage' column in tidy data.")

    averaged = (
        tidy.groupby(group_cols, sort=False)
        .agg(
            mean=("voltage", "mean"),
            sem=("voltage", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
        .sort_values(group_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    context.averaged = averaged
    return context

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

def crop_stim_artifact(
    context: RecordingContext,
    window_ms: tuple[float, float] = (0.0, 1.25)
) -> pd.DataFrame:
    """
    Crop stimulation artifacts from tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    window_ms
        Time window (start_ms, end_ms) in milliseconds to remove after stimulus onset.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with stimulation artifacts removed.
    """
    tidy_df = context.tidy
    def crop_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("time")
        x = g["time"].to_numpy()

        # time-based indices
        start_idx, stop_idx = window_to_indices(x, window_ms)

        cropped = pd.concat([g.iloc[:start_idx], g.iloc[stop_idx:]], ignore_index=True)

        # re-zero time so all traces align perfectly
        cropped["time"] = cropped["time"] - float(cropped["time"].iloc[0])
        return cropped

    context.tidy = tidy_df.groupby(["stim_intensity", "sweep_number"], group_keys=False)[["stim_intensity", "sweep_number", "time", "voltage"]].apply(crop_group)
    return context

def template_subtract_stim_artifact(
    context: RecordingContext,
    window_ms: tuple[float, float] = (0.0, 1.25),
) -> pd.DataFrame:
    """
    Template subtract stimulation artifacts from tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    window_ms
        Time window (start_ms, end_ms) in milliseconds to remove after stimulus onset.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with stimulation artifacts removed.
    """
    tidy_df = context.tidy
    def subtract_template(g_int: pd.DataFrame) -> pd.DataFrame:
        # g_int = all sweeps for one stim_intensity
        g_int = g_int.sort_values(["sweep_number", "time"]).copy()

        # Build template using columns only
        template_df = (
            g_int
            .groupby("time", sort=True)["voltage"]
            .mean()
            .reset_index()
        )

        t = template_df["time"].to_numpy()
        T = template_df["voltage"].to_numpy()

        start_idx, stop_idx = window_to_indices(t, window_ms)
        template, _ = build_template(t, T, window_ms)

        out = []
        for sw, g in g_int.groupby("sweep_number", sort=False):
            g = g.sort_values("time").copy()
            x = g["time"].to_numpy()
            y = g["voltage"].to_numpy()

            # sanity: time grids must match
            # (can be removed once you're confident)
            if not np.allclose(x, t):
                raise ValueError("Time grid mismatch between sweep and template")

            a = project_template(y[start_idx:stop_idx], template)
            if not np.isfinite(a):
                out.append(g)
                continue

            y[start_idx:stop_idx] -= a * template
            g["voltage"] = y
            out.append(g)

        return pd.concat(out, ignore_index=True)

    context.tidy = tidy_df.groupby("stim_intensity", group_keys=False)[["stim_intensity", "sweep_number", "time", "voltage"]].apply(subtract_template)
    return context

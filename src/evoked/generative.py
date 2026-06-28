from __future__ import annotations
import numpy as np
import pandas as pd
from evoked.base import IntermediateResult, FeatureResult, window_to_indices
from pandera.typing import DataFrame

def generative_epsp_model(
        intermediate: IntermediateResult, 
        artifact_window: tuple, 
        fv_window: tuple, 
        ps_window: tuple,
        template_intensities: list[int]):
    """A generative fEPSP model based on sequential template matching."""
    template_data = intermediate[intermediate["intensity"].isin(template_intensities)]
    if template_data.empty:
        raise ValueError("No traces found for template_intensities.")

    template_snippets = []
    for _, group in template_data.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        template_start, template_stop = window_to_indices(time, template_window)
        template_snippets.append(signal[template_start:template_stop])

    template_array = np.mean(np.vstack(template_snippets), axis=0)
    return template_array, slope_transform
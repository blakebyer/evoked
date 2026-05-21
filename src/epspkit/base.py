from __future__ import annotations

from dataclasses import dataclass, field
import pandera.pandas as pa
import pandas as pd
import numpy as np
from pandera.typing import Series, DataFrame

def window_to_indices(
    x: np.ndarray,
    window_ms: tuple[float, float],
) -> tuple[int, int]:
    t0, t1 = [value / 1000.0 for value in window_ms]
    return int(np.searchsorted(x, t0)), int(np.searchsorted(x, t1))

class RecordingData(pa.DataFrameModel):
    time: Series[float]
    voltage: Series[float]
    intensity: Series[int]
    sweepNumber: Series[int]

class IntermediateResult(pa.DataFrameModel):
    time: Series[float]
    voltage: Series[float]
    intensity: Series[int]

class FitResult(pa.DataFrameModel):
    intensity: Series[int] # stimulus intensity
    lag: Series[float] # difference in ms from center of template to center of best fit
    scale: Series[float] # vertical scale
    stretch: Series[float] # horizontal stretch
    corr: Series[float] # pearson corr
    r2: Series[float] # r^2

@dataclass
class FeatureResult:
    search_window: tuple[float, float]
    template_window: tuple[float, float]
    result: DataFrame[FitResult] = field(default_factory=pd.DataFrame)

@dataclass
class RecordingResult:
    results: dict[str, FeatureResult] = field(default_factory=dict)

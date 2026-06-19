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
    id: Series[str]
    time: Series[float]
    voltage: Series[float]
    intensity: Series[int]
    sweepNumber: Series[int]

class IntermediateResult(pa.DataFrameModel):
    id: Series[str]
    time: Series[float]
    voltage: Series[float]
    intensity: Series[int]

class FitResult(pa.DataFrameModel):
    """
    A class for the template fit result.
    Attributes:
        id (str): The animal id
        intensity (int): The current injected into the slice
        feature_time (float): The time in ms at the maximum correlation between fitted template and snippet
        scale (float): The least squares coefficient
        corr (float): The maximum Pearson correlation between fitted template and snippet
        corr_arr (np.ndarray): A numpy array containing correlation between fitted template and snippet at every point
        r2 (float): The coefficient of determination between fitted template and snippet
    """
    id: Series[str] 
    intensity: Series[int]
    feature_time: Series[float] 
    scale: Series[float]
    corr: Series[float] 
    corr_arr: Series[object] = pa.Field(nullable=False)
    r2: Series[float]
    detected: Series[bool]

@dataclass
class PreprocessParams:
    baseline_window: tuple[float, float]
    artifact_window: tuple[float, float]
    artifact: str
    smoothing: str
    smoothing_params: dict = field(default_factory=dict)

@dataclass
class FeatureResultTemplate:
    search_window: tuple[float, float]
    template_window: tuple[float, float]
    slope_transform: bool
    template: np.ndarray | None = None
    result: DataFrame[FitResult] = field(default_factory=pd.DataFrame)
    
@dataclass
class RecordingResult:
    preprocess_params: PreprocessParams
    results: dict[str, FeatureResultTemplate] = field(default_factory=dict)
    def add(self, result_key: str, feature_result: FeatureResultTemplate) -> None:
        self.results[result_key] = feature_result


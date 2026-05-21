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
    id: Series[str]
    intensity: Series[int] # stimulus intensity
    match_time: Series[float] # time in ms at max corr
    scale: Series[float] # vertical scale
    corr: Series[float] # pearson corr
    r2: Series[float] # r^2

@dataclass
class PreprocessParams:
    baseline_window: tuple[float, float]
    artifact_window: tuple[float, float]
    artifact: str
    smoothing: str
    smoothing_params: dict = field(default_factory=dict)

@dataclass
class FeatureResult:
    search_window: tuple[float, float]
    template_window: tuple[float, float]
    result: DataFrame[FitResult] = field(default_factory=pd.DataFrame)

@dataclass
class RecordingResult:
    preprocess_params: PreprocessParams
    results: dict[str, FeatureResult] = field(default_factory=dict)
    def get(self, result_key: str) -> FeatureResult:
        return self.results[result_key]
    def add(self, result_key: str, feature_result: FeatureResult) -> None:
        self.results[result_key] = feature_result

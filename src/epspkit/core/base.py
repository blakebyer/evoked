from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence
import numpy as np

@dataclass
class RecordingContext:
    traces: list[Trace, any]
    stimuli: Sequence[int, any]

@dataclass
class Trace:
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    voltage: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

@dataclass
class Template:
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    voltage: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

@dataclass
class FitResult:
    lag: float
    scale: float
    r2: float

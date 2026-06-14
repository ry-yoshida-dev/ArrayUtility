from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..types import FloatArray


@dataclass
class EMACalculator:
    """
    Exponential Moving Average calculator.

    Attributes
    ----------
    value : FloatArray
        Current EMA value with shape (*feature_shape).
    alpha : float, default=0.5
        Smoothing factor in [0, 1]. Larger values react faster to new samples.
    """

    value: FloatArray
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], but got {self.alpha}")
        if not np.issubdtype(self.value.dtype, np.floating):
            raise TypeError(f"value dtype must be floating, but got {self.value.dtype}")

    def update(self, value: FloatArray) -> None:
        """
        Update the EMA with a new value.

        Parameters
        ----------
        value : FloatArray
            New sample with shape (*feature_shape).
        """
        if value.shape != self.value.shape:
            raise ValueError(f"value shape {value.shape} must match {self.value.shape}")
        self.value += self.alpha * (value - self.value)

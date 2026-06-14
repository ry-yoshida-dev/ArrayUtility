from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseStandardRingBuffer
from ...types import NumericArray


@dataclass
class RingBuffer(BaseStandardRingBuffer):
    @property
    def mean(self) -> NumericArray:
        """
        Get the mean of the buffer.

        Returns
        -------
        NumericArray
            The arithmetic mean of the buffer with shape (*feature_shape).
        """
        return np.mean(self.value, axis=0)

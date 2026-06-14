from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseAveragingRingBuffer
from ...types import NumericArray


@dataclass
class AveragingRingBuffer(BaseAveragingRingBuffer):
    def _sum_chunk(self, values: NumericArray) -> NumericArray:
        """
        Sum vectors over a chunk.

        Parameters
        ----------
        values : NumericArray
            Values with shape (m, *feature_shape).

        Returns
        -------
        NumericArray
            Element-wise sum with shape (*feature_shape).
        """
        return np.sum(values, axis=0)

    @property
    def mean(self) -> NumericArray:
        """
        Get the mean of the buffer.

        Returns
        -------
        NumericArray
            The arithmetic mean of the buffer with shape (*feature_shape).
        """
        return self._accumulator / self.n

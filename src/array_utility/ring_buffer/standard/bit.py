from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .base import BaseStandardRingBuffer
from ...types import NumericArray, UInt8Array


@dataclass
class BitRingBuffer(BaseStandardRingBuffer):
    """
    Ring buffer for bit-encoded vectors (for example, pHash bytes).
    """

    def _validate_dtype(self) -> None:
        """
        Validate that stored vectors are uint8 byte arrays.

        Raises
        ------
        TypeError
            If `value.dtype` is not `np.uint8`.
        """
        if self.value.dtype != np.uint8:
            raise TypeError(f"value dtype must be uint8, but got {self.value.dtype}")

    def update(self, value: NumericArray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : NumericArray
            The new vector with shape (*feature_shape) to be stored.
        """
        if value.dtype != np.uint8:
            raise TypeError(f"value dtype must be uint8, but got {value.dtype}")
        super().update(value)

    def extend(self, values: NumericArray) -> None:
        """
        Extend the buffer with new vectors.

        Parameters
        ----------
        values : NumericArray
            The new vectors with shape (m, *feature_shape) to be stored.
        """
        if values.dtype != np.uint8:
            raise TypeError(f"values dtype must be uint8, but got {values.dtype}")
        super().extend(values)

    @property
    def mean(self) -> UInt8Array:
        """
        Get bitwise majority value over the whole buffer.

        Returns
        -------
        UInt8Array
            Bitwise-majority representative with shape (*feature_shape),
            where each bit is selected by majority vote across n entries.
        """
        bit_sum = np.sum(np.unpackbits(cast(UInt8Array, self.value), axis=-1), axis=0)
        majority_bits = (bit_sum >= (self.n / 2)).astype(np.uint8)
        return np.packbits(majority_bits, axis=-1)

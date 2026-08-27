from __future__ import annotations
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from ..types import IntegerArray, NumericArray


BaseRingBufferT = TypeVar("BaseRingBufferT", bound="BaseRingBuffer")


@dataclass
class BaseRingBuffer(ABC):
    """
    Fixed-length ring buffer for storing feature vectors.

    Attributes
    ----------
    n : int
        The maximum number of frames stored in the buffer.
    value : NumericArray
        Buffer array of shape (n, *feature_shape).
    _index : int
        Internal pointer to the next write position.
    """

    n: int
    value: NumericArray
    _index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be greater than 0, but got {self.n}")
        if self.value.ndim == 1:
            raise ValueError(f"value must be 2D or higher, but got {self.value.ndim}")
        if self.value.shape[0] != self.n:
            raise ValueError(f"value first axis length ({self.value.shape[0]}) must match n ({self.n})")

    @abstractmethod
    def update(self, value: NumericArray) -> None:
        """
        Update the buffer with a new vector.

        Parameters
        ----------
        value : NumericArray
            The new vector with shape (*feature_shape) to be stored.
        """

    @abstractmethod
    def extend(self, values: NumericArray) -> None:
        """
        Extend the buffer with new vectors.

        Parameters
        ----------
        values : NumericArray
            The new vectors with shape (n, *feature_shape) to be stored.
        """

    def get_last_k(self, k: int) -> NumericArray:
        """
        Get the last k vectors from the buffer.

        Parameters
        ----------
        k : int
            The number of vectors to get.
        """
        if not (0 <= k <= self.n):
            raise ValueError(f"k ({k}) must be in range [0, {self.n}]")
        start = self._index - k
        if start >= 0:
            return self.value[start : self._index]
        return np.concatenate([self.value[start:], self.value[: self._index]], axis=0)

    def _validate_vector_shape(self, value: NumericArray) -> None:
        expected_shape = self.value.shape[1:]
        if value.shape != expected_shape:
            raise ValueError(f"value shape {value.shape} must match {expected_shape}")

    def _validate_batch_shape(self, values: NumericArray) -> None:
        if values.ndim != self.value.ndim:
            raise ValueError(f"values ndim {values.ndim} must match {self.value.ndim}")
        expected_shape = self.value.shape[1:]
        if values.shape[1:] != expected_shape:
            raise ValueError(f"values shape[1:] {values.shape[1:]} must match {expected_shape}")

    @property
    def latest(self) -> NumericArray:
        """
        Get the most recently appended vector.

        Returns
        -------
        NumericArray
            The most recently stored vector with shape (*feature_shape).
        """
        return self.value[self._index - 1]

    @property
    def ordered_indices(self) -> IntegerArray:
        """
        Get the buffer rows in the order of the oldest to latest.

        Indexing `value` with this is what `ordered_value` returns, without
        the whole-buffer copy `np.roll` makes. A caller that only wants part
        of the ordered buffer -- some of its features, or one row per feature
        -- should index with this instead, so the copy is the size of what it
        asked for rather than of the buffer.

        Returns
        -------
        IntegerArray
            Row indices with shape ``(n,)``.
        """
        return (np.arange(self.n) + self._index) % self.n

    @property
    def ordered_value(self) -> NumericArray:
        """
        Get the value of the buffer in the order of the oldest to latest.

        Returns
        -------
        NumericArray
            The ordered value of the buffer with shape (n, *feature_shape).
        """
        return np.roll(self.value, -self._index, axis=0)

    @property
    @abstractmethod
    def mean(self) -> NumericArray:
        """
        Get the mean of the buffer.

        Returns
        -------
        NumericArray
            The mean of the buffer with shape (*feature_shape).
        """

    @classmethod
    def build(
        cls: type[BaseRingBufferT],
        n: int,
        init_value: NumericArray,
    ) -> BaseRingBufferT:
        """
        Create a buffer instance pre-filled with an initial tensor of shape.

        Parameters
        ----------
        n : int
            The maximum number of frames stored in the buffer.
        init_value : NumericArray
            Initial value with shape (*feature_shape) used to fill all frames.

        Returns
        -------
        BaseRingBufferT
            The buffer instance.
        """

        init_buffer = np.empty((n, *init_value.shape), dtype=init_value.dtype)
        init_buffer[:] = init_value
        return cls(n=n, value=init_buffer)

    def __len__(self) -> int:
        """
        Get the number of frames stored in the buffer.
        """

        return self.n

    def _update_index(self, step: int = 1) -> None:
        """
        Update write index with wrap-around.
        """

        self._index = (self._index + step) % self.n


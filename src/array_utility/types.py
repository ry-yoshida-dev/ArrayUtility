"""Shared NumPy array type aliases for array_utility."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer[Any] | np.floating[Any]]
type FloatArray = NDArray[np.floating[Any]]
type IntegerArray = NDArray[np.integer[Any]]
type UInt8Array = NDArray[np.uint8]

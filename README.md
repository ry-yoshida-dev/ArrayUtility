# ArrayUtility

## Overview

`ArrayUtility` provides typed, NumPy-based utilities for streaming array operations: exponential moving average (EMA) and fixed-size ring buffers, including bit-oriented variants for `uint8` data.

## Installation

Requires Python 3.12+.

```bash
pip install .
```

## Usage

```python
import numpy as np
from array_utility import EMACalculator, AveragingRingBuffer

# EMA
ema = EMACalculator(alpha=0.2, value=np.array([0.0, 0.0], dtype=np.float32))
ema.update(np.array([1.0, 3.0], dtype=np.float32))
print(ema.value)

# Ring buffer + running mean
buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.update(np.array([1.0, 2.0], dtype=np.float32))
buf.update(np.array([2.0, 4.0], dtype=np.float32))
print(buf.latest)  # [2.0, 4.0]
print(buf.mean)    # mean over fixed window size (n=3)
```

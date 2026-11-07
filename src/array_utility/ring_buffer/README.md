# ring_buffer

## Overview

Fixed-length ring buffers for array features. `standard` buffers compute `mean` on demand; `averaging` buffers cache accumulators so frequent `mean` queries stay cheap. Bit variants store `uint8` bit-packed vectors (e.g. pHash bytes) and use the bitwise majority as `mean`.

## Components

| Component | Description |
| --------- | ----------- |
| [base.py](base.py) | Abstract base class with common buffer operations |
| [averaging](averaging/README.md) | `AveragingRingBuffer` and `BitAveragingRingBuffer`, with cached accumulation |
| [standard](standard/README.md) | `RingBuffer` and `BitRingBuffer`, computing `mean` on demand |

## Examples

```python
import numpy as np
from array_utility import AveragingRingBuffer

buf = AveragingRingBuffer.build(n=3, init_value=np.array([0.0, 0.0], dtype=np.float32))
buf.update(np.array([1.0, 2.0], dtype=np.float32))
buf.extend(np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

print(buf.latest)        # most recent vector
print(buf.get_last_k(2)) # last k vectors
print(buf.mean)          # mean over all buffer slots
```

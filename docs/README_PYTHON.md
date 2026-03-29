# Custom Loss Functions - Python Extension

High-performance C++ implementations of custom loss functions for financial forecasting, with Python bindings via pybind11.

## Overview

This package provides **C++ accelerated** implementations of custom loss functions that penalize prediction direction errors more heavily than magnitude errors. Perfect for financial time series forecasting where getting the direction right (up vs down) matters more than exact values.

### Performance Benefits

- **2-5x faster** forward pass compared to pure Python
- **1.5-3x faster** backward pass (gradient computation)
- Lower memory usage
- Production-ready C++ code
- Seamless PyTorch integration

## Installation

The Makefile keeps an isolated environment in **`.venv`** at the project root (add `.venv` to `.gitignore`; it is not committed).

### Recommended (Makefile)

```bash
make develop    # creates .venv if needed, installs torch + pybind11, then pip install -e . --no-build-isolation
```

### Verify Installation

```bash
.venv/bin/python -c "import custom_loss; print('✓ Installation successful!')"
```

### Manual (same layout)

```bash
python3.14 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel torch pybind11
pip install -e . --no-build-isolation
```

### Other installs

**Regular install into `.venv`:** `make install` (or `pip install . --no-build-isolation` after activating `.venv`).

**Wheel:** `make wheel`, then `pip install dist/custom_loss_cpp-*.whl`.

The **Python extension** is compiled when you `pip install` / `make develop` (via `setup.py`), not with CMake. An optional standalone C++ executable (`examples/test_losses.cpp`) is documented under CMake in the root `README.md` if you want that binary.

## Quick Start

```python
import torch
from custom_loss import AdjMSELoss1

# Create loss function
criterion = AdjMSELoss1(alpha=2.0)

# Use in training
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
```

## Benchmarking

Run the included benchmark to compare Python vs C++ performance:

```bash
python scripts/benchmark.py
# or: make benchmark
```

Expected results will show performance comparisons across different batch sizes.

## Examples

Smoke test (import, forward, backward):

```bash
python scripts/smoke_test.py
# or: make test
```

## Citation

```bibtex
@article{dessain2021custom,
  title={Custom Loss Functions for Asset Return Prediction},
  author={Dessain, J.},
  journal={SSRN},
  year={2021},
  url={https://ssrn.com/abstract=3973086}
}
```

## License

All rights reserved - Copyright Navagne (2021)

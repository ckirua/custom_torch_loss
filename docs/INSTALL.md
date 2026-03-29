# Installation Guide - Custom Loss Functions Python Extension

This guide walks through installing the C++ accelerated PyTorch loss functions.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Platform-Specific Notes](#platform-specific-notes)

## System Requirements

### Required

- **Python**: 3.14 (project `Makefile` defaults `PYTHON_FOR_VENV` to `python3.14`)
- **PyTorch**: 1.8.0 or higher
- **C++ Compiler**: 
  - Linux: GCC 5+ or Clang 3.4+
  - macOS: Clang (via Xcode Command Line Tools)
  - Windows: MSVC 2017+ or MinGW

### Build Dependencies

```bash
pip install torch pybind11 setuptools wheel
```

## Quick Installation

### Using Makefile (Recommended)

Creates **`.venv`** in the project directory and installs everything there (isolated from system Python).

```bash
make develop    # .venv + torch + pybind11 + editable package

make test       # smoke test via .venv/bin/python
make benchmark
```

### Manual Installation

With the same **`.venv`** layout:

```bash
python3.14 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel torch pybind11
pip install -e . --no-build-isolation

# Production mode into .venv
pip install . --no-build-isolation
```

## Detailed Installation Steps

### Step 1: Clone/Download the Code

```bash
git clone <repository-url>
cd custom_torch_loss   # project root (directory name may differ)
```

### Step 2: Create `.venv` and install dependencies

```bash
python3.14 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install torch pybind11
```

Or run **`make develop`** from the project root (creates `.venv` and runs the equivalent).

### Step 3: Build the extension

The C++ extension must compile against the **same** PyTorch as in `.venv`, so use **`--no-build-isolation`**:

```bash
# Option A: Development install (recommended)
pip install -e . --no-build-isolation

# Option B: Regular install
pip install . --no-build-isolation

# Option C: Wheel
python setup.py bdist_wheel
pip install dist/custom_loss_cpp-*.whl
```

### Step 4: Verify installation

```bash
.venv/bin/python -c "import custom_loss; print('✓ Success!')"
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/benchmark.py
```

## Verification

### Import Test

```python
import torch
from custom_loss import AdjMSELoss1, AdjMSELoss2, AdjMSELoss3

print("✓ All imports successful")
```

### Functional Test

```python
import torch
from custom_loss import AdjMSELoss1

# Create test data
outputs = torch.randn(10, 1, requires_grad=True)
labels = torch.randn(10, 1)

# Test loss computation
criterion = AdjMSELoss1(alpha=2.0)
loss = criterion(outputs, labels)

# Test backpropagation
loss.backward()

print(f"✓ Loss computed: {loss.item():.6f}")
print(f"✓ Gradients computed: {outputs.grad is not None}")
```

## Troubleshooting

### Error: "C++ compiler not found"

You need a working C++ toolchain (e.g. GCC/Clang on Linux, Xcode command-line tools on macOS, MSVC on Windows). Fix that in your environment; this repo does not install system packages for you.

### Error: "torch/extension.h not found"

**Solution:** Ensure PyTorch is properly installed

```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### Error: "pybind11 not found"

```bash
pip install pybind11
```

### `pip` targets another `.venv` / `libtorch_global_deps.so` missing

If install logs show **`site-packages`** under a **different** project path than this repo’s `.venv`, but `.venv/bin/python` imports a **broken** `torch` (e.g. missing `libtorch_global_deps.so`), the **`.venv/bin/pip` script** usually has a **stale shebang** from a copied or moved environment.

This project’s **`Makefile` uses `.venv/bin/python -m pip`** so packages install into the **same** interpreter CMake uses. After pulling that change, recreate a clean env if needed:

```bash
rm -rf .venv build
make develop
```

Or reinstall torch only: `.venv/bin/python -m pip install --force-reinstall torch`.

### CMake: "Could not find Torch" or "TorchConfig.cmake"

CMake must point at LibTorch via **`torch.utils.cmake_prefix_path`** (not the raw `torch` package directory). This applies only if you build the **optional** `test_losses` demo (see root `README.md`). The **`custom_loss_cpp`** module is built by **`setup.py`** / `pip install`, not CMake.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(../.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build . --config Release
```

Use your venv’s `python` path if it differs. Run the binary as `./bin/test_losses` from `build/` (or `build/bin/test_losses` from the repo root).

### CMake: "CUDA cannot be found" / "Your installed Caffe2 version uses CUDA"

The **pip** PyTorch wheel is **CUDA-linked** (default on Linux PyPI). **CMake** then expects a **full CUDA toolkit** on the machine (`nvcc`, libraries), even if you only wanted `build/bin/test_losses`.

- **Default in this repo:** **`make deps`** installs the **CPU** wheel (`TORCH_INDEX_URL` defaults to PyTorch’s CPU index) so **`make build-cpp`** works without installing CUDA. If you already installed a CUDA wheel earlier, switch with:
  ```bash
  rm -rf build .venv
  make develop
  make build-cpp
  ```
  Or only replace torch: `.venv/bin/python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch`
- **GPU PyTorch + CMake:** use `make deps TORCH_INDEX_URL=` (empty) or your CUDA wheel source, **install a matching CUDA toolkit**, set `CUDA_HOME` / `PATH` for `nvcc`, then configure CMake again.

### Error: "ImportError: custom_loss_cpp"

This means the C++ extension wasn't built properly.

```bash
# Clean and rebuild
make clean
make develop

# Or manually (with torch already in that env):
rm -rf build/ dist/ *.egg-info
pip install -e . --no-build-isolation
```

### Compilation succeeds but import fails

**Check Python path:**
```python
import sys
print(sys.path)
```

The extension should be in one of these paths. If using `develop`, it should be in your current directory.

### Performance not as expected

1. **Ensure CUDA is being used** (if available):
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

2. **Test with larger batch sizes**:
   - Performance benefits are more apparent with batch_size > 1000

3. **Disable gradient computation for inference**:
```python
with torch.no_grad():
    loss = criterion(outputs, labels)
```

## Platform-Specific Notes

Use **`make develop`** (or the manual `.venv` + `pip install -e . --no-build-isolation` flow) on Linux, macOS, or Windows once you have Python 3.14 and a C++ compiler; that path builds the extension via **`setup.py`**. The optional CMake demo (`test_losses`) is separate—see the root `README.md` if you need it. PyTorch wheels must match your platform (CPU vs CUDA, ARM vs x86); get those from [pytorch.org](https://pytorch.org/) if the default `pip install torch` is wrong for your machine.

## Build Options

### Optimizations

Add compiler flags for better performance:

**setup.py modification:**
```python
extra_compile_args=[
    '-O3',           # Maximum optimization
    '-march=native', # CPU-specific optimizations
    '-ffast-math',   # Fast math operations
]
```

### Debug Build

For debugging:

```python
extra_compile_args=[
    '-g',            # Debug symbols
    '-O0',           # No optimization
]
```

## Uninstallation

### Using pip

```bash
pip uninstall custom_loss_cpp
```

### Using Makefile

```bash
make uninstall
make clean
```

## Next Steps

After successful installation:

1. **Run smoke test**: `python scripts/smoke_test.py` or `make test`
2. **Run benchmarks**: `python scripts/benchmark.py` or `make benchmark`
3. **Read documentation**: See `docs/README_PYTHON.md`
4. **Integrate into your project**: See usage examples

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Verify all requirements are installed
3. Try a clean rebuild: `make clean && make develop`
4. Check PyTorch compatibility
5. Review the error messages carefully

## Quick Reference

```bash
# Clean build
make clean

# Development install
make develop

# Run tests
make test

# Run benchmarks  
make benchmark

# Build wheel
make wheel

# Uninstall
make uninstall

# Develop then smoke test
make develop && make test
```

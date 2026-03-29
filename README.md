# Custom Loss Functions for Asset Return Prediction

LibTorch (PyTorch C++) implementation of custom loss functions for financial forecasting models.

## Overview

This implementation provides 6 custom loss functions that penalize prediction direction errors more heavily than magnitude errors. This is particularly useful for financial applications where getting the direction right (up vs down) is more important than the exact magnitude.

**Loss Functions:**
- **AdjMSELoss1**: Step-function adjustment (MSE-based)
- **AdjMSELoss2**: Sigmoid-like adjustment (MSE-based)  
- **AdjMSELoss3**: ReLU-like adjustment (MSE-based)
- **AdjMAELoss1**: Step-function adjustment (MAE-based)
- **AdjMAELoss2**: Sigmoid-like adjustment (MAE-based)
- **AdjMAELoss3**: ReLU-like adjustment (MAE-based)

## Repository layout

| Path | Purpose |
|------|---------|
| `custom_loss/` | Python package (`nn.Module` wrappers around `custom_loss_cpp`) |
| `cpp/` | LibTorch loss implementation and pybind11 bindings |
| `examples/` | Standalone C++ demo binary (`test_losses`) |
| `scripts/` | `benchmark.py`, `smoke_test.py` |
| `docs/` | Python-focused README and detailed install guide |

## Python (PyTorch extension)

Use the project virtualenv at `.venv` (not committed): from this directory run `make develop` (creates `.venv` with **`python3.14`**, installs torch/pybind11, then editable install with `--no-build-isolation`). **`make install`** / **`develop`** compiles the `custom_loss_cpp` extension via **`setup.py`** (Torch `CppExtension` + pybind11); **CMake is not involved** for Python.

Override the interpreter with `make PYTHON_FOR_VENV=/other/python develop`. Then `make test` / `make benchmark` use that interpreter. Details: [docs/README_PYTHON.md](docs/README_PYTHON.md), [docs/INSTALL.md](docs/INSTALL.md).

## Optional: standalone C++ demo (`test_losses`, CMake)

Only needed if you want the small LibTorch **executable** in `examples/`—not for `import custom_loss`. For that, use the previous section.

### Prerequisites

- **CMake** 3.14+
- **`torch` in `.venv`** (same as Python workflow): run `make deps` or `make develop` first so `import torch` works.

Optional: a separate [LibTorch](https://pytorch.org/) tree only if you do not use pip PyTorch (see below).

### Build Instructions

**From the repo root** (recommended): CMake is told explicitly which Python has torch.

```bash
make deps        # CPU PyTorch by default (so CMake does not need nvcc); see Makefile TORCH_INDEX_URL
make build-cpp
./build/bin/test_losses
```

**Manual CMake:** `CMakeLists.txt` sets `CMAKE_PREFIX_PATH` from `torch.utils.cmake_prefix_path` automatically, using (in order) `-DTORCH_PYTHON=...` if you pass it, else `.venv/bin/python` under this repo, else `python3` on `PATH`. So from `build/`:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./bin/test_losses
```

If discovery picks the wrong interpreter, pass `-DTORCH_PYTHON=/absolute/path/to/python` (must be able to `import torch`).

**Standalone LibTorch** (no pip torch): disable auto-discovery and point CMake at the SDK:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_USE_PIP_PYTHON=OFF \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release
./bin/test_losses
```

Add `-I/path/to/this/repo/cpp` (or equivalent) when compiling your own targets so `#include "custom_loss_functions.h"` resolves.

## Usage Example

```cpp
#include "custom_loss_functions.h"
#include <torch/torch.h>

int main() {
    // Create sample predictions and labels
    torch::Tensor outputs = torch::randn({100, 1});
    torch::Tensor labels = torch::randn({100, 1});
    
    // Initialize loss function
    AdjMSELoss1 loss_fn(2.0);  // alpha = 2.0
    
    // Calculate loss
    torch::Tensor loss = loss_fn->forward(outputs, labels);
    
    std::cout << "Loss: " << loss.item<double>() << std::endl;
    
    // Use in training loop
    loss.backward();
    
    return 0;
}
```

## Integration with Training Loop

```cpp
// Define your model
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 1));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
    
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    // Initialize model, optimizer, and custom loss
    auto model = std::make_shared<Net>();
    torch::optim::Adam optimizer(model->parameters(), 0.001);
    AdjMSELoss1 criterion(2.0);
    
    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        // Your data loading here
        torch::Tensor inputs = torch::randn({32, 10});
        torch::Tensor labels = torch::randn({32, 1});
        
        optimizer.zero_grad();
        torch::Tensor outputs = model->forward(inputs);
        torch::Tensor loss = criterion->forward(outputs, labels);
        loss.backward();
        optimizer.step();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " 
                      << loss.item<double>() << std::endl;
        }
    }
    
    return 0;
}
```

## Extending the Loss Functions

### Adding New Parameters

To make parameters configurable:

```cpp
class AdjMSELoss1Impl : public torch::nn::Module {
public:
    double alpha;
    double threshold;  // NEW parameter
    
    AdjMSELoss1Impl(double alpha_val = 2.0, double thresh = 0.0) 
        : alpha(alpha_val), threshold(thresh) {}
    
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels) {
        outputs = outputs.squeeze();
        torch::Tensor loss = torch::pow(outputs - labels, 2);
        torch::Tensor adj = torch::mul(outputs, labels);
        
        // Use threshold instead of 0
        adj = torch::where(adj > threshold,
                          torch::full_like(adj, 1.0 / alpha),
                          torch::full_like(adj, alpha));
        
        loss = loss * adj;
        return torch::mean(loss);
    }
};
```

### Creating New Loss Variants

Example: Add exponential adjustment

```cpp
class AdjMSELoss4Impl : public torch::nn::Module {
public:
    double decay_rate;
    
    explicit AdjMSELoss4Impl(double rate = 0.5) : decay_rate(rate) {}
    
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels) {
        outputs = outputs.squeeze();
        torch::Tensor loss = torch::pow(outputs - labels, 2);
        torch::Tensor product = torch::mul(outputs, labels);
        
        // Exponential decay for correct predictions
        torch::Tensor adj = torch::exp(-decay_rate * torch::abs(product));
        adj = torch::where(product > 0, adj, 2.0 - adj);
        
        loss = loss * adj;
        return torch::mean(loss);
    }
};

TORCH_MODULE(AdjMSELoss4);
```

### Adding Weighted Samples

```cpp
class WeightedAdjMSELoss1Impl : public torch::nn::Module {
public:
    double alpha;
    torch::Tensor sample_weights;
    
    explicit WeightedAdjMSELoss1Impl(double alpha_val = 2.0) : alpha(alpha_val) {}
    
    void set_weights(torch::Tensor weights) {
        sample_weights = weights;
    }
    
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels) {
        outputs = outputs.squeeze();
        torch::Tensor loss = torch::pow(outputs - labels, 2);
        torch::Tensor adj = torch::mul(outputs, labels);
        
        adj = torch::where(adj > 0,
                          torch::full_like(adj, 1.0 / alpha),
                          torch::full_like(adj, alpha));
        
        loss = loss * adj;
        
        // Apply sample weights if provided
        if (sample_weights.defined()) {
            loss = loss * sample_weights;
        }
        
        return torch::mean(loss);
    }
};

TORCH_MODULE(WeightedAdjMSELoss1);
```

## Loss Function Comparison

| Loss Function | Adjustment Type | Best For | Parameters |
|--------------|----------------|----------|------------|
| AdjMSELoss1  | Step function  | Clear direction signals | alpha (1.5-2.0) |
| AdjMSELoss2  | Sigmoid smooth | Noisy data | beta (2.25-2.5) |
| AdjMSELoss3  | ReLU-like     | Small penalties | gamma (0.1) |
| AdjMAELoss*  | Same as MSE   | Outlier robustness | Same as MSE |

## Parameter Tuning Guidelines

- **alpha** (AdjLoss1): Higher values = stronger penalty for wrong direction
  - Start with 2.0
  - Increase to 2.5-3.0 for highly directional tasks
  - Decrease to 1.5 for balanced tasks

- **beta** (AdjLoss2): Controls sigmoid smoothness
  - 2.5 for standard use
  - 2.25 for gentler transitions
  - 3.0 for sharper transitions

- **gamma** (AdjLoss3): Penalty scaling factor
  - 0.1 for standard use
  - Lower (0.05) for minimal penalty difference
  - Higher (0.2) for stronger correct-direction rewards

## Testing (C++ demo)

After a CMake build:

```bash
./build/bin/test_losses
```

This prints sample scalar losses for all six modules.

## References

- Original Paper: https://ssrn.com/abstract=3973086
- LibTorch Documentation: https://pytorch.org/cppdocs/
- Original Python Implementation: See source code header

## License

All rights reserved - Copyright Navagne (2021)
Based on work by JDE65 (Github)

## Contact

For questions about the original implementation:
- j.dessain@navagne.com
- j.dessain@ieseg.fr
- www.navagne.com

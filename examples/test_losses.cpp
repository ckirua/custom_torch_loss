/*
 * Minimal LibTorch demo: print scalar loss for each custom loss module.
 * Optional CMake demo only (Python uses setup.py). Build per README.md; run: build/bin/test_losses
 */

#include <iostream>
#include <torch/torch.h>

#include "custom_loss_functions.h"

int main() {
    torch::manual_seed(42);
    torch::Tensor outputs = torch::randn({100, 1});
    torch::Tensor labels = torch::randn({100, 1});

    AdjMSELoss1 mse1(2.0);
    AdjMSELoss2 mse2(2.5);
    AdjMSELoss3 mse3(0.1);
    AdjMAELoss1 mae1(2.0);
    AdjMAELoss2 mae2(2.5);
    AdjMAELoss3 mae3(0.1);

    std::cout << "Custom loss sample values (batch=100):\n";
    std::cout << "  AdjMSELoss1: " << mse1->forward(outputs, labels).item<double>() << "\n";
    std::cout << "  AdjMSELoss2: " << mse2->forward(outputs, labels).item<double>() << "\n";
    std::cout << "  AdjMSELoss3: " << mse3->forward(outputs, labels).item<double>() << "\n";
    std::cout << "  AdjMAELoss1: " << mae1->forward(outputs, labels).item<double>() << "\n";
    std::cout << "  AdjMAELoss2: " << mae2->forward(outputs, labels).item<double>() << "\n";
    std::cout << "  AdjMAELoss3: " << mae3->forward(outputs, labels).item<double>() << "\n";
    return 0;
}

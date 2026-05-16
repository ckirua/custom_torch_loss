/*
 * Custom Loss Functions for Asset Return Prediction - Header File
 * Using PyTorch C++ API (LibTorch)
 */

#ifndef CUSTOM_LOSS_FUNCTIONS_H
#define CUSTOM_LOSS_FUNCTIONS_H

#include <torch/torch.h>

/* ============================================================================
 * MSE-BASED CUSTOM LOSS FUNCTIONS
 * ============================================================================ */

/**
 * AdjMSELoss1: Step-function adjustment
 * 
 * Penalizes incorrect direction predictions more heavily.
 * When outputs and labels have same sign (correct direction): multiply by 1/alpha
 * When outputs and labels have opposite sign (wrong direction): multiply by alpha
 * 
 * @param alpha: Adjustment factor (default: 2.0, also try 1.5)
 */
class AdjMSELoss1Impl : public torch::nn::Module {
public:
    double alpha;
    
    explicit AdjMSELoss1Impl(double alpha_val = 2.0);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMSELoss1);


/**
 * AdjMSELoss2: Sigmoid-like adjustment
 * 
 * Smooth adjustment using sigmoid function for gradual penalty transition.
 * 
 * @param beta: Adjustment factor (default: 2.5, also try 2.25)
 */
class AdjMSELoss2Impl : public torch::nn::Module {
public:
    double beta;
    
    explicit AdjMSELoss2Impl(double beta_val = 2.5);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMSELoss2);


/**
 * AdjMSELoss3: ReLU-like adjustment
 * 
 * Linear penalty adjustment based on prediction direction.
 * When correct direction: multiply by gamma
 * When incorrect direction: multiply by (1 + gamma)
 * 
 * @param gamma: Adjustment factor (default: 0.1)
 */
class AdjMSELoss3Impl : public torch::nn::Module {
public:
    double gamma;
    
    explicit AdjMSELoss3Impl(double gamma_val = 0.1);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMSELoss3);


/* ============================================================================
 * MAE-BASED CUSTOM LOSS FUNCTIONS
 * ============================================================================ */

/**
 * AdjMAELoss1: Step-function adjustment with MAE base
 */
class AdjMAELoss1Impl : public torch::nn::Module {
public:
    double alpha;
    
    explicit AdjMAELoss1Impl(double alpha_val = 2.0);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMAELoss1);


/**
 * AdjMAELoss2: Sigmoid-like adjustment with MAE base
 */
class AdjMAELoss2Impl : public torch::nn::Module {
public:
    double beta;
    
    explicit AdjMAELoss2Impl(double beta_val = 2.5);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMAELoss2);


/**
 * AdjMAELoss3: ReLU-like adjustment with MAE base
 */
class AdjMAELoss3Impl : public torch::nn::Module {
public:
    double gamma;
    
    explicit AdjMAELoss3Impl(double gamma_val = 0.1);
    torch::Tensor forward(torch::Tensor outputs, torch::Tensor labels);
};

TORCH_MODULE(AdjMAELoss3);

#endif // CUSTOM_LOSS_FUNCTIONS_H

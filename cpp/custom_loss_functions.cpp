/*
 * Custom Loss Functions for Asset Return Prediction
 * Using PyTorch C++ API (LibTorch)
 *
 * Original author: JDE65 (Github)
 * Original paper: https://ssrn.com/abstract=3973086
 * LibTorch conversion: 2026
 *
 * Implementation only; declarations in custom_loss_functions.h
 */

#include "custom_loss_functions.h"

/* ============================================================================
 * MSE-BASED CUSTOM LOSS FUNCTIONS
 * ============================================================================ */

AdjMSELoss1Impl::AdjMSELoss1Impl(double alpha_val) : torch::nn::Module(), alpha(alpha_val) {}

torch::Tensor AdjMSELoss1Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::pow(outputs - labels, 2);
    torch::Tensor adj = torch::mul(outputs, labels);
    adj = torch::where(adj > 0,
                      torch::full_like(adj, 1.0 / alpha),
                      torch::full_like(adj, alpha));
    loss = loss * adj;
    return torch::mean(loss);
}

AdjMSELoss2Impl::AdjMSELoss2Impl(double beta_val) : torch::nn::Module(), beta(beta_val) {}

torch::Tensor AdjMSELoss2Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::pow(outputs - labels, 2);
    torch::Tensor product = torch::mul(outputs, labels);
    torch::Tensor adj_loss = beta - (beta - 0.5) /
                             (1.0 + torch::exp(10000.0 * product));
    loss = beta * loss / (1.0 + adj_loss);
    return torch::mean(loss);
}

AdjMSELoss3Impl::AdjMSELoss3Impl(double gamma_val) : torch::nn::Module(), gamma(gamma_val) {}

torch::Tensor AdjMSELoss3Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::pow(outputs - labels, 2);
    torch::Tensor adj = torch::mul(outputs, labels);
    adj = torch::where(adj > 0,
                      torch::full_like(adj, gamma),
                      torch::full_like(adj, 1.0 + gamma));
    loss = loss * adj;
    return torch::mean(loss);
}

/* ============================================================================
 * MAE-BASED CUSTOM LOSS FUNCTIONS
 * ============================================================================ */

AdjMAELoss1Impl::AdjMAELoss1Impl(double alpha_val) : torch::nn::Module(), alpha(alpha_val) {}

torch::Tensor AdjMAELoss1Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::abs(outputs - labels);
    torch::Tensor adj = torch::mul(outputs, labels);
    adj = torch::where(adj > 0,
                      torch::full_like(adj, 1.0 / alpha),
                      torch::full_like(adj, alpha));
    loss = loss * adj;
    return torch::mean(loss);
}

AdjMAELoss2Impl::AdjMAELoss2Impl(double beta_val) : torch::nn::Module(), beta(beta_val) {}

torch::Tensor AdjMAELoss2Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::abs(outputs - labels);
    torch::Tensor product = torch::mul(outputs, labels);
    torch::Tensor adj_loss = beta - (beta - 0.5) /
                             (1.0 + torch::exp(10000.0 * product));
    loss = beta * loss / (1.0 + adj_loss);
    return torch::mean(loss);
}

AdjMAELoss3Impl::AdjMAELoss3Impl(double gamma_val) : torch::nn::Module(), gamma(gamma_val) {}

torch::Tensor AdjMAELoss3Impl::forward(torch::Tensor outputs, torch::Tensor labels) {
    outputs = outputs.squeeze();
    labels = labels.squeeze();
    torch::Tensor loss = torch::abs(outputs - labels);
    torch::Tensor adj = torch::mul(outputs, labels);
    adj = torch::where(adj > 0,
                      torch::full_like(adj, gamma),
                      torch::full_like(adj, 1.0 + gamma));
    loss = loss * adj;
    return torch::mean(loss);
}

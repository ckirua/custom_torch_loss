"""
Custom Loss Functions - Python Wrapper
High-level PyTorch nn.Module wrappers around C++ implementations

This module provides PyTorch-compatible loss functions that can be used
just like standard PyTorch losses (MSELoss, L1Loss, etc.)
"""

import torch
import torch.nn as nn

try:
    import custom_loss_cpp
except ImportError:
    raise ImportError(
        "custom_loss_cpp C++ extension not found. Build it from the repo root, e.g.:\n"
        "  make develop\n"
        "or: pip install -e . --no-build-isolation (with torch already in that env)"
    )


class AdjMSELoss1(nn.Module):
    """
    Adjusted MSE Loss with Step-function adjustment

    Penalizes wrong-direction predictions more heavily than correct-direction ones.
    When prediction and label have same sign: penalty *= 1/alpha
    When prediction and label have opposite sign: penalty *= alpha

    Args:
        alpha (float): Adjustment factor. Higher values penalize wrong direction more.
                       Recommended: 1.5 to 2.0. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output.
                        Currently only 'mean' is supported.

    Shape:
        - Input: (N, *) where * means any number of dimensions
        - Target: (N, *), same shape as input
        - Output: scalar

    Examples:
        >>> loss_fn = AdjMSELoss1(alpha=2.0)
        >>> outputs = torch.randn(10, 1, requires_grad=True)
        >>> labels = torch.randn(10, 1)
        >>> loss = loss_fn(outputs, labels)
        >>> loss.backward()
    """

    def __init__(self, alpha=2.0, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMSELoss1(alpha)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, labels):
        """
        Args:
            outputs (Tensor): Model predictions
            labels (Tensor): Ground truth values

        Returns:
            Tensor: Computed loss value
        """
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'alpha={self.alpha}'


class AdjMSELoss2(nn.Module):
    """
    Adjusted MSE Loss with Sigmoid-like adjustment

    Uses a smooth sigmoid function for gradual penalty adjustment based on
    prediction direction correctness.

    Args:
        beta (float): Smoothness parameter for sigmoid adjustment.
                      Recommended: 2.25 to 2.5. Default: 2.5
        reduction (str): Specifies the reduction to apply to the output.
                        Currently only 'mean' is supported.

    Shape:
        - Input: (N, *) where * means any number of dimensions
        - Target: (N, *), same shape as input
        - Output: scalar
    """

    def __init__(self, beta=2.5, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMSELoss2(beta)
        self.beta = beta
        self.reduction = reduction

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'beta={self.beta}'


class AdjMSELoss3(nn.Module):
    """
    Adjusted MSE Loss with ReLU-like adjustment

    Applies linear penalty scaling based on prediction direction.
    Correct direction: penalty *= gamma
    Wrong direction: penalty *= (1 + gamma)

    Args:
        gamma (float): Penalty scaling factor. Lower values reward correct
                       direction more. Recommended: 0.1. Default: 0.1
        reduction (str): Specifies the reduction to apply to the output.
                        Currently only 'mean' is supported.

    Shape:
        - Input: (N, *) where * means any number of dimensions
        - Target: (N, *), same shape as input
        - Output: scalar
    """

    def __init__(self, gamma=0.1, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMSELoss3(gamma)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'gamma={self.gamma}'


class AdjMAELoss1(nn.Module):
    """
    Adjusted MAE Loss with Step-function adjustment

    MAE-based variant of AdjMSELoss1. More robust to outliers.

    Args:
        alpha (float): Adjustment factor. Default: 2.0
        reduction (str): Currently only 'mean' is supported.
    """

    def __init__(self, alpha=2.0, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMAELoss1(alpha)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'alpha={self.alpha}'


class AdjMAELoss2(nn.Module):
    """
    Adjusted MAE Loss with Sigmoid-like adjustment

    MAE-based variant of AdjMSELoss2. More robust to outliers.

    Args:
        beta (float): Smoothness parameter. Default: 2.5
        reduction (str): Currently only 'mean' is supported.
    """

    def __init__(self, beta=2.5, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMAELoss2(beta)
        self.beta = beta
        self.reduction = reduction

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'beta={self.beta}'


class AdjMAELoss3(nn.Module):
    """
    Adjusted MAE Loss with ReLU-like adjustment

    MAE-based variant of AdjMSELoss3. More robust to outliers.

    Args:
        gamma (float): Penalty scaling factor. Default: 0.1
        reduction (str): Currently only 'mean' is supported.
    """

    def __init__(self, gamma=0.1, reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise NotImplementedError("Only 'mean' reduction is currently supported")
        self.loss_fn = custom_loss_cpp.AdjMAELoss3(gamma)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def extra_repr(self):
        return f'gamma={self.gamma}'


# Convenience dictionary for easy access
LOSS_FUNCTIONS = {
    'adj_mse_1': AdjMSELoss1,
    'adj_mse_2': AdjMSELoss2,
    'adj_mse_3': AdjMSELoss3,
    'adj_mae_1': AdjMAELoss1,
    'adj_mae_2': AdjMAELoss2,
    'adj_mae_3': AdjMAELoss3,
}


def get_loss_function(name, **kwargs):
    """
    Factory function to get loss function by name

    Args:
        name (str): Loss function name (e.g., 'adj_mse_1', 'adj_mae_2')
        **kwargs: Arguments to pass to the loss function constructor

    Returns:
        nn.Module: Instantiated loss function

    Example:
        >>> loss_fn = get_loss_function('adj_mse_1', alpha=2.0)
        >>> loss = loss_fn(outputs, labels)
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function: {name}. "
            f"Available: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[name](**kwargs)


__all__: tuple[str, ...] = (
    'AdjMSELoss1', 'AdjMSELoss2', 'AdjMSELoss3',
    'AdjMAELoss1', 'AdjMAELoss2', 'AdjMAELoss3',
    'get_loss_function', 'LOSS_FUNCTIONS',
)

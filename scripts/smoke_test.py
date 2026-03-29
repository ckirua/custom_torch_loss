"""Quick import and forward/backward check for the custom_loss package."""

import sys

import torch

from custom_loss import AdjMSELoss1, AdjMSELoss2, AdjMSELoss3


def main() -> int:
    for cls, kwargs in (
        (AdjMSELoss1, {"alpha": 2.0}),
        (AdjMSELoss2, {"beta": 2.5}),
        (AdjMSELoss3, {"gamma": 0.1}),
    ):
        outputs = torch.randn(8, 1, requires_grad=True)
        labels = torch.randn(8, 1)
        loss = cls(**kwargs)(outputs, labels)
        loss.backward()
        if outputs.grad is None:
            print(f"FAIL: no grad for {cls.__name__}", file=sys.stderr)
            return 1
    print("OK: custom_loss forward/backward smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

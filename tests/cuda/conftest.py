"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container)."""

import importlib.util


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]

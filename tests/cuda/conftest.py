"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container)."""

import gc
import importlib.util
import os

import pytest

# GLM's engines keep the MTP head beside DFlash2 here (TF_GLM_MTP=1), so both drafters stay under test
os.environ.setdefault("TF_GLM_MTP", "1")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]


@pytest.fixture(autouse=True, scope="module")
def _give_back_gpu_memory():
    """After each module: free its tensors and torch's cached blocks, so the next engine admits as a fresh process."""

    yield
    if not _cuda():
        return
    import torch

    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


def pytest_collection_modifyitems(config, items):
    """Below compute capability 8.9 (no FP8 MMA) the FP8-prompt cases (a parameter ``fp8`` set True) skip."""

    if not _cuda():
        return
    from tensorfold.cuda import build

    if build.has(build.FP8):
        return
    skip = pytest.mark.skip(reason="FP8 prompts need compute capability 8.9")
    for item in items:
        spec = getattr(item, "callspec", None)
        if spec is not None and spec.params.get("fp8") is True:
            item.add_marker(skip)

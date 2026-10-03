"""A GGUF whose output head is unquantized (Plain) gives the DFlash2 drafter a Plain sub-head of the span rows."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda.dflash2 import _plain_sub_head  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Plain  # noqa: E402


def test_plain_sub_head_takes_the_span_rows_in_order():
    w = torch.arange(40 * 8, dtype=torch.float32).reshape(40, 8).to(torch.bfloat16)
    sub = _plain_sub_head(Plain(w), ((3, 7), (20, 22), (39, 40)))
    assert isinstance(sub, Plain) and sub.layout == "b16"
    assert torch.equal(sub.weight, torch.cat([w[3:7], w[20:22], w[39:40]]))
    assert sub.weight.is_contiguous() and not hasattr(sub, "scales")

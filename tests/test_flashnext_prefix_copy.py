"""Prefix copies preserve cache codes and scales exactly and never copy a future partial pool."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
from tensorfold.families.qwen4_exp.cuda.state import State


def slot(dtype, capacity, pos):
    st = object.__new__(State)
    st.capacity, st.pos, st.mtp_len, st.ratio, st.kv_dtype = capacity, pos, max(0, pos - 1), 4, dtype
    st.kc = [KVCache(capacity, 2, 64, "cpu", dtype) for _ in range(2)]
    st.ikc = [torch.empty((capacity, 16), dtype=torch.bfloat16) for _ in st.kc]
    st.pooled = [torch.empty(((capacity + 3) // 4, 16), dtype=torch.bfloat16) for _ in st.kc]
    st.mtp_kc = KVCache(capacity, 2, 64, "cpu", dtype)
    st.mtp_ikc = torch.empty((capacity, 16), dtype=torch.bfloat16)
    st.mtp_pooled = torch.empty(((capacity + 3) // 4, 16), dtype=torch.bfloat16)
    return st


def tensors(st, pos, mtp):
    for cache in st.kc:
        for key in ("k", "v", "ks", "vs"):
            yield getattr(cache, key), pos if key in ("k", "v") or cache.quantized else 0
    for value in st.ikc:
        yield value, pos
    for value in st.pooled:
        yield value, pos // st.ratio
    for key in ("k", "v", "ks", "vs"):
        yield getattr(st.mtp_kc, key), mtp if key in ("k", "v") or st.mtp_kc.quantized else 0
    yield st.mtp_ikc, mtp
    yield st.mtp_pooled, mtp // st.ratio


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("pos", [17, 18, 20])
def test_only_prefix_bytes_are_copied_and_destination_writes_cannot_change_source(dtype, pos):
    src, dst = slot(dtype, 64, 50), slot(dtype, 32, 0)
    gen = torch.Generator().manual_seed(31)
    for t, _ in tensors(src, pos, pos - 1):
        t.view(torch.uint8).random_(0, 256, generator=gen)
    for t, _ in tensors(dst, pos, pos - 1):
        t.view(torch.uint8).fill_(165)
    saved = [t.clone() for t, _ in tensors(src, pos, pos - 1)]
    dst.copy_prefix(src, pos, pos - 1)
    for (got, n), (want, _) in zip(tensors(dst, pos, pos - 1), tensors(src, pos, pos - 1)):
        assert got.data_ptr() != want.data_ptr()
        assert torch.equal(got[:n].view(torch.uint8), want[:n].view(torch.uint8))
        assert torch.all(got[n:].view(torch.uint8) == 165)
        got.zero_()
    assert dst.pos == 0 and dst.mtp_len == 0
    assert all(torch.equal(t.view(torch.uint8), old.view(torch.uint8))
               for (t, _), old in zip(tensors(src, pos, pos - 1), saved))


def test_copy_refuses_aliases_incompatible_formats_and_unavailable_rows():
    src, dst = slot("bf16", 64, 50), slot("bf16", 32, 0)
    for target, pos, mtp in ((src, 10, 9), (slot("int8", 32, 0), 10, 9), (dst, 33, 32), (dst, 10, 33)):
        with pytest.raises(ValueError):
            target.copy_prefix(src, pos, mtp)

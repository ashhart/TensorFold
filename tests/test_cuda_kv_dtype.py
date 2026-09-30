"""Flash Next's quantized KV cache before any weight loads: its bytes, the window admission gives it, the rank handshake."""

import pytest

from tests.test_cuda_capacity import Loaded, checkpoint, fake_runtime, small_config  # noqa: F401

torch = pytest.importorskip("torch")

WEIGHTS = [("lm_head.weight", "U32", [64, 8], 2048)]
BITS = {"bf16": 16, "int8": 8, "int4": 4}


def build(path, kv_dtype, streams=1, requested=None, explicit=None, **kw):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    obj = FlashNextEngine.__new__(FlashNextEngine)
    return obj, lambda: obj.__init__(path, max_len=requested, context_explicit=explicit, depth=3, kv_dtype=kv_dtype,
                                     streams=streams, **kw)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_the_cache_holds_what_admission_counts(kv_dtype):
    from tensorfold.cuda.geometry import kv_bytes
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    cache = KVCache(64, 2, 256, "cpu", kv_dtype)
    scalars = 4 if kv_dtype == "bf16" else 0                 # a bf16 cache keeps two one-element scale tensors
    assert cache.nbytes == 64 * 2 * 2 * kv_bytes(256, BITS[kv_dtype]) + scalars
    # the shipped checkpoint: 12 attention layers and the MTP head, 2 KV heads of 256, indexer keys of 128 at ratio 4
    token = 13 * (2 * 2 * kv_bytes(256, BITS[kv_dtype]) + (128 + 128 // 4) * 2)
    assert token == {"bf16": 30784, "int8": 18304, "int4": 11648}[kv_dtype]


def test_the_family_and_the_cache_list_the_same_dtypes():
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import kvcache

    assert qwen4_exp.CUDA_KV_DTYPES == kvcache.DTYPES == tuple(kvcache.BITS_OF)


@pytest.mark.parametrize("streams", [1, 4])
def test_quantized_caches_admit_longer_windows_on_the_same_budget(tmp_path, monkeypatch, fake_runtime, streams):  # noqa: F811
    from tensorfold.cuda.geometry import gdn_geometry, indexed_stream_geometry

    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, capacity = fake_runtime
    text = small_config()
    bf16 = gdn_geometry(text, 1, 4, indexed=True, mtp=True) if streams == 1 else \
        indexed_stream_geometry(text, streams, 4, 8, mtp=True)
    budget = bf16.needed(12000) + 32768                      # bf16 fits about 12,000 tokens
    monkeypatch.setattr(capacity, "available_bytes", lambda t: budget)
    windows = {}
    for kv_dtype in ("bf16", "int8", "int4"):
        obj, go = build(tmp_path, kv_dtype, streams)
        with pytest.raises(Loaded):
            go()
        windows[kv_dtype] = obj.capacity_plan["context_window"]
        assert obj.capacity_plan["total_bytes_estimate"] <= budget
    assert 12000 <= windows["bf16"] < windows["int8"] < windows["int4"] < 65536, windows
    calls.clear()
    over = windows["bf16"] + 1000                            # past bf16's largest window, inside int8's
    _, refused = build(tmp_path, "bf16", streams, requested=over, explicit=True)
    with pytest.raises(ValueError, match="largest fitting"):
        refused()
    assert not calls
    _, admitted = build(tmp_path, "int8", streams, requested=over, explicit=True)
    with pytest.raises(Loaded):
        admitted()
    assert len(calls) == 1


def test_a_quantized_cache_needs_a_head_dim_of_whole_groups(tmp_path, fake_runtime):  # noqa: F811
    config = dict(small_config(), head_dim=48)
    checkpoint(tmp_path, config, WEIGHTS)
    calls, _ = fake_runtime
    _, go = build(tmp_path, "int8")
    with pytest.raises(ValueError, match="multiple of 32"):
        go()
    assert not calls
    _, plain = build(tmp_path, "bf16")
    with pytest.raises(Loaded):
        plain()


@pytest.mark.parametrize("kv_dtype", ["fp8", "int2"])
def test_an_unknown_cache_is_refused_before_loading(tmp_path, fake_runtime, kv_dtype):  # noqa: F811
    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, _ = fake_runtime
    _, go = build(tmp_path, kv_dtype)
    with pytest.raises(ValueError, match="kv-dtype"):
        go()
    assert not calls


@pytest.mark.parametrize("peer", ["bf16", "int4"])
def test_two_ranks_with_different_caches_refuse_to_start(fake_runtime, peer):  # noqa: F811
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    class Comm:
        def __init__(self, other=None):
            self.other, self.sent = other, None

        def all_gather(self, send, recv):
            self.sent = send.clone()
            recv.copy_(torch.cat([send, send if self.other is None else self.other]))

    def rank(kv_dtype, comm):
        obj = FlashNextEngine.__new__(FlashNextEngine)
        obj.depth, obj.confidence, obj.max_len, obj.kv_dtype, obj.comm = 6, 0.3, 8192, kv_dtype, comm
        return obj

    theirs = Comm()
    rank(peer, theirs)._same_settings(torch, None)            # a rank agrees with itself
    with pytest.raises(RuntimeError, match="different settings"):
        rank("int8", Comm(theirs.sent))._same_settings(torch, None)


def test_two_ranks_with_different_prompt_precision_refuse_to_start(fake_runtime):  # noqa: F811
    from tensorfold.cuda import prompt_precision
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    class Comm:
        def __init__(self, other=None):
            self.other, self.sent = other, None

        def all_gather(self, send, recv):
            self.sent = send.clone()
            recv.copy_(torch.cat([send, send if self.other is None else self.other]))

    def rank(comm):
        obj = FlashNextEngine.__new__(FlashNextEngine)
        obj.depth, obj.confidence, obj.max_len, obj.kv_dtype, obj.comm = 6, 0.3, 8192, "bf16", comm
        return obj

    theirs = Comm()
    with prompt_precision.using(True):                        # rank 1 started with --prefill-fp8
        rank(theirs)._same_settings(torch, None)
    with prompt_precision.using(False), pytest.raises(RuntimeError, match="prompt precision.*--prefill-fp8"):
        rank(Comm(theirs.sent))._same_settings(torch, None)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_a_draft_confidence_outside_0_to_1_is_refused_before_loading(tmp_path, fake_runtime, confidence):  # noqa: F811
    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, _ = fake_runtime
    _, go = build(tmp_path, "bf16", confidence=confidence)
    with pytest.raises(ValueError, match="probability from 0 to 1"):
        go()
    assert not calls

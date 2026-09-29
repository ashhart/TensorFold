"""GLM-5.3-Flash's host side: each rank's share of the checkpoint (read in place, or from a folder the split
wrote) and the request header both ranks decode, on a small synthetic checkpoint (no GPU work)."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch

from tensorfold.families.glm5_next.cuda import split
from tensorfold.families.glm5_next.cuda.engine import _f64_ints, _ints_f64, decode_policy, encode_policy

L = "model.language_model."
TENSORS = {           # name: (dtype, shape); the split rule each one takes is in the name
    L + "embed_tokens.weight": ("U32", [16, 8]),
    L + "embed_tokens.scales": ("BF16", [16, 1]),
    L + "norm.weight": ("BF16", [16]),
    L + "layers.3.mlp.experts.0.gate_proj.weight": ("U32", [8, 8]),
    L + "layers.3.mlp.experts.0.gate_proj.scales": ("BF16", [8, 2]),
    L + "layers.3.mlp.experts.0.down_proj.weight": ("U32", [16, 4]),
    L + "layers.3.mlp.experts.0.down_proj.scales": ("BF16", [16, 2]),
    L + "layers.3.self_attn.A_log": ("F32", [4]),
    L + "layers.3.self_attn.o_proj.weight": ("U32", [8, 4]),
    "lm_head.weight": ("U32", [16, 8]),
    "model.visual.blocks.0.attn.qkv.weight": ("BF16", [4, 4]),
}
KINDS = {name: split.rule(name) for name in TENSORS}


def _checkpoint(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    raw = {name: rng.integers(0, 256, size=int(np.prod(shape)) * split.DTYPE_BYTES[dtype], dtype=np.uint8)
           for name, (dtype, shape) in TENSORS.items()}
    names = sorted(TENSORS)
    files = {"model-00001-of-00002.safetensors": names[:5], "model-00002-of-00002.safetensors": names[5:]}
    weight_map = {}
    for file, part in files.items():
        split.write(str(tmp_path / file), [(n, TENSORS[n][0], TENSORS[n][1], raw[n]) for n in part], {"format": "mlx"})
        weight_map.update({n: file for n in part})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text("{}")
    return raw


def _whole(raw: np.ndarray, dtype: str, shape: list[int]) -> torch.Tensor:
    torch_dtype = {"U32": torch.uint32, "BF16": torch.bfloat16, "F32": torch.float32}[dtype]
    return torch.from_numpy(raw.copy()).view(torch_dtype).reshape(shape)


def test_split_rules_cover_every_kind():
    assert {KINDS[L + "layers.3.mlp.experts.0.gate_proj.weight"], KINDS[L + "layers.3.self_attn.A_log"]} == {"row"}
    assert KINDS[L + "layers.3.mlp.experts.0.down_proj.weight"] == KINDS[L + "layers.3.self_attn.o_proj.weight"] == "col"
    assert KINDS["lm_head.weight"] == KINDS[L + "norm.weight"] == KINDS[L + "embed_tokens.weight"] == "rep"
    assert KINDS["model.visual.blocks.0.attn.qkv.weight"] == "drop"
    with pytest.raises(ValueError):
        split.rule(L + "layers.3.mystery.weight")


def test_rank_shares_read_in_place_and_after_a_split_agree(tmp_path):
    raw = _checkpoint(tmp_path / "ckpt")
    for rank in (0, 1):
        out = tmp_path / f"rank{rank}"
        split.main([str(tmp_path / "ckpt"), "--rank", str(rank), str(out)])
        assert (out / "config.json").exists() and split.rank_files(out, rank) and not split.rank_files(out, 1 - rank)
        full, folder = split.RankReader(tmp_path / "ckpt", rank), split.RankReader(out, rank)
        assert not full.split and folder.split
        for name, (dtype, shape) in TENSORS.items():
            if KINDS[name] == "drop":
                with pytest.raises(KeyError):
                    full.get(name)
                continue
            a, b = full.get(name), folder.get(name)
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    # the two ranks' parts put back together are the checkpoint's tensor (rank 1's read ahead on threads)
    r0, r1 = split.RankReader(tmp_path / "ckpt", 0), split.RankReader(tmp_path / "ckpt", 1)
    r1.prefetch([name for name in TENSORS if KINDS[name] != "drop"])
    for name, (dtype, shape) in TENSORS.items():
        kind = KINDS[name]
        if kind == "drop":
            continue
        whole = _whole(raw[name], dtype, shape)
        a, b = r0.get(name), r1.get(name)
        if kind == "rep":
            assert torch.equal(a.view(torch.uint8), whole.view(torch.uint8))
            continue
        joined = torch.cat([a.view(torch.uint8), b.view(torch.uint8)], dim=0 if kind == "row" else 1)
        assert torch.equal(joined, whole.view(torch.uint8)), name
    assert not r1.ahead
    r1.close()
    with pytest.raises(ValueError):
        split.RankReader(tmp_path / "rank0", 1)      # rank 0's folder given to rank 1


def test_reads_ahead_in_runs_of_neighbouring_tensors(tmp_path, monkeypatch):
    _checkpoint(tmp_path / "ckpt")
    names = [name for name in TENSORS if KINDS[name] != "drop"]
    for run, gap in ((1 << 20, 1 << 20), (100, 0)):                # one run a file, then about one a tensor
        monkeypatch.setattr(split, "RUN", run)
        monkeypatch.setattr(split, "GAP", gap)
        for rank in (0, 1):
            ahead, alone = split.RankReader(tmp_path / "ckpt", rank), split.RankReader(tmp_path / "ckpt", rank)
            ahead.prefetch(names)
            reads = len({id(f) for f in ahead.ahead.values()})
            assert reads == 2 if run > 100 else reads > 2              # the checkpoint has two files
            for name in names:
                a, b = ahead.get(name), alone.get(name)
                assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            assert not ahead.ahead
            ahead.close()


def test_reads_ahead_uploaded_to_the_gpu_match_the_host_reads(tmp_path, monkeypatch):
    _checkpoint(tmp_path / "ckpt")
    names = [name for name in TENSORS if KINDS[name] != "drop"]
    for run, gap in ((1 << 20, 1 << 20), (100, 0)):
        monkeypatch.setattr(split, "RUN", run)
        monkeypatch.setattr(split, "GAP", gap)
        for rank in (0, 1):
            ahead, alone = split.RankReader(tmp_path / "ckpt", rank), split.RankReader(tmp_path / "ckpt", rank)
            ahead.prefetch(names, "cuda")
            for name in names:
                a, b = ahead.get(name), alone.get(name)
                assert a.is_cuda and a.dtype == b.dtype and a.shape == b.shape, name
                assert torch.equal(a.view(torch.uint8).cpu(), b.view(torch.uint8)), name
            ahead.close()



def test_rank_folders_read_ahead_like_the_checkpoint(tmp_path):
    _checkpoint(tmp_path / "ckpt")
    names = [name for name in TENSORS if KINDS[name] != "drop"]
    for rank in (0, 1):
        out = tmp_path / f"rank{rank}"
        split.main([str(tmp_path / "ckpt"), "--rank", str(rank), str(out)])
        for device in (None, "cuda"):
            ahead, alone = split.RankReader(out, rank), split.RankReader(out, rank)
            ahead.prefetch(names, device)
            for name in names:
                a, b = ahead.get(name), alone.get(name)
                assert a.is_cuda == (device is not None) and a.dtype == b.dtype and a.shape == b.shape, name
                assert torch.equal(a.view(torch.uint8).cpu(), b.view(torch.uint8)), name
            ahead.close()


def test_uploads_queued_without_waits_match_the_host_reads(tmp_path, monkeypatch):
    """Each uploaded tensor cloned on the current stream and dropped at once, with no synchronization until the end:
    runs of about one tensor, so upload buffers are freed and reused while earlier clones may still read them."""

    _checkpoint(tmp_path / "ckpt")
    monkeypatch.setattr(split, "RUN", 100)
    monkeypatch.setattr(split, "GAP", 0)
    names = [name for name in TENSORS if KINDS[name] != "drop"]
    for rank in (0, 1):
        ahead, alone = split.RankReader(tmp_path / "ckpt", rank), split.RankReader(tmp_path / "ckpt", rank)
        ahead.prefetch(names, "cuda")
        taken = [ahead.get(name).clone() for name in names]
        torch.cuda.synchronize()
        for name, a in zip(names, taken):
            b = alone.get(name)
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8).cpu(), b.view(torch.uint8))
        ahead.close()


def test_policies_and_the_request_header():
    assert encode_policy("0") == [0, 0, 0, 0]
    assert encode_policy("2") == [1, 2, 0, 0]
    assert encode_policy("a:0.6:0.85") == [2, 3, 600000, 850000]
    assert encode_policy("c3:0.35") == [3, 3, 350000, 0]
    assert encode_policy("fc5:0.3") == [13, 5, 300000, 0]
    assert encode_policy("f0") == [0, 0, 0, 0]
    assert encode_policy("a") == [2, 3, 800000, 900000]
    assert encode_policy("auto") == [4, 2, 8, 30000] and encode_policy("auto:1:2:0.05") == [5, 1, 2, 50000]
    assert decode_policy(encode_policy("auto")) == ("auto", 2, 8, 0.03, False)
    assert decode_policy(encode_policy("auto:1:2:0.05")) == ("auto", 1, 2, 0.05, True)
    for bad in ("", "x", "c3", "c9:0.3", "9", "-1", "a:0.6", "ab", "auto:1", "auto:0:1:0.1", "auto:1:1:1.5"):
        with pytest.raises(ValueError):
            encode_policy(bad)
    assert decode_policy(encode_policy("0")) is None
    fixed = decode_policy(encode_policy("2"))
    assert fixed.fixed and fixed.most == 2 and fixed.confidence == 0
    conf = decode_policy(encode_policy("fc5:0.3"))
    assert conf.fixed and conf.most == 5 and conf.confidence == 0.3
    run = decode_policy(encode_policy("a:0.6:0.85"))
    assert not run.fixed and (run.low, run.high) == (0.6, 0.85)
    # floats cross the ranks bit for bit, and seeds keep all 64 bits (a prompt-derived seed uses 63)
    for x in (0.0, 1.0, 0.95, 0.1234567, 1e-7, 2.5):
        lo, hi = _f64_ints(x)
        assert -2**31 <= lo < 2**31 and -2**31 <= hi < 2**31 and _ints_f64(lo, hi) == x
        assert struct.pack("<d", _ints_f64(lo, hi)) == struct.pack("<d", x)
    for seed in (0, 1234, 2**62 + 5, 2**63 - 1, 2**64 - 1):
        parts = [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62]
        assert all(0 <= p < 2**31 for p in parts)
        assert (parts[2] << 62) | (parts[1] << 31) | parts[0] == seed


def test_family_has_a_mac_engine_and_cuda_needs_two_ranks(tmp_path):
    from tensorfold.families import glm5_next

    assert glm5_next.MODEL_TYPES == ("glm5_next",) and callable(glm5_next.load)
    assert glm5_next.CUDA_APP.__name__ == "GlmApp"
    with pytest.raises(ValueError, match="two GPUs"):
        glm5_next.cuda_engine(tmp_path, tp=1)
    with pytest.raises(ValueError, match="--master"):
        glm5_next.cuda_engine(tmp_path, tp=2, rank=0, master="")


def test_requests_past_the_context_get_a_400_before_streaming(tmp_path):
    """Past the limit a request gets a 400 before streaming, naming a --context only when startup would admit it."""

    import threading

    from tokenizers import Tokenizer, models, pre_tokenizers

    from tensorfold.families.glm5_next.cuda.app import GlmApp

    words = ["[UNK]", "<|user|>", "<|assistant|>", "<think>", "</think>"] + [f"w{i}" for i in range(50)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    template = "{% for m in messages %}<|{{ m.role }}|> {{ m.content }} {% endfor %}<|assistant|><think>"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))

    class Engine:
        limit = 12
        eos = (0,)
        request = threading.local()
        capacity_plan = {"largest_window": 64}

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            raise AssertionError("not reached")

    app = GlmApp(Engine(), tmp_path, "glm")
    eight = " ".join(f"w{i}" for i in range(8))
    assert app.check({"prompt": eight, "max_tokens": 4}) is None                   # 8 + 4 = 12
    problem = app.check({"prompt": eight, "max_tokens": 5})
    assert "13-token context" in problem and "--context 13" in problem and "started for 12" in problem
    assert app.check({"prompt": eight}) is None                                    # the reply stops at the limit
    assert "--context 13" in app.check({"prompt": " ".join(f"w{i}" for i in range(12))})
    chat = {"messages": [{"role": "user", "content": eight}], "max_tokens": 2}     # <|user|>, 8 words, the tail
    assert app.check(chat) is None
    assert "--context 13" in app.check({**chat, "max_tokens": 3})


def test_check_accepts_mlx_4bit_and_mias_exl3_only(tmp_path):
    from tensorfold.families import glm5_next

    assert glm5_next.QUANT_METHODS == {"mlx": ("mlx",), "cuda": ("mlx", "exl3")}
    cases = {
        "mlx4": ({"quantization": {"bits": 4, "group_size": 64}}, True),
        "mlx8": ({"quantization": {"bits": 8, "group_size": 64}}, False),
        "exl3": ({"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg",
                                          "scope": "glm53_routed_experts_only"}}, True),
        "exl3-3bit": ({"quantization_config": {"quant_method": "exl3", "bits": 3, "codebook": "mcg",
                                               "scope": "glm53_routed_experts_only"}}, False),
        "exl3-all": ({"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "3inst",
                                              "scope": "all"}}, False),
    }
    for name, (quant, ok) in cases.items():
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "glm5_next", **quant}))
        if ok:
            glm5_next.check(d)
        else:
            with pytest.raises(ValueError, match="recipe book"):
                glm5_next.check(d)

"""The tiny DeepSeek-V4.1-Flash checkpoint: the pack's names and dtypes, accepted by check(), the designed roles."""

from __future__ import annotations

import filecmp
import json
import struct
import time
from pathlib import Path

import dsv41_tiny
import numpy as np
import pytest
from dsv41_inventory import inventory
from dsv41_tiny import PAD, VOCAB, write_tiny

from tensorfold.cuda import capacity
from tensorfold.families import deepseek_v41
from tensorfold.families.deepseek_v41.config import Config, Role
from tensorfold.families.deepseek_v41.cuda import split

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture

LIMIT = 60_000_000       # the specified attention, mHC and expert dims alone exceed 32 MB
FOLD = (3, 134)          # E8M0 bytes Mx8Linear's prompt GEMM folds exactly
SPAN = 17                # widest E8M0 spread of one MXFP4 matrix the NVFP4 conversion holds


def _tensors(folder: Path):
    """(name, header entry, raw bytes) of every tensor in the index's shards."""

    files = sorted(set(json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"].values()))
    for file in files:
        raw = (folder / file).read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        for name, info in json.loads(raw[8:8 + size]).items():
            lo, hi = info["data_offsets"]
            yield name, info, raw[8 + size + lo:8 + size + hi]


def test_written_fast_small_and_the_same_for_a_seed(tiny_dir, tmp_path):
    start = time.perf_counter()
    again = write_tiny(tmp_path / "again", seed=0)
    assert time.perf_counter() - start < 5
    assert sum(f.stat().st_size for f in again.iterdir()) < LIMIT
    names = sorted(f.name for f in tiny_dir.iterdir())
    assert names == sorted(f.name for f in again.iterdir())
    assert filecmp.cmpfiles(tiny_dir, again, names, shallow=False)[0] == names
    other = write_tiny(tmp_path / "other", seed=1)
    assert not filecmp.cmp(tiny_dir / "model-00002-of-00007.safetensors",
                           other / "model-00002-of-00007.safetensors", shallow=False)


def test_check_accepts_it(tiny_dir, capsys):
    deepseek_v41.check(tiny_dir)
    assert "two DGX Sparks" in capsys.readouterr().out


def test_config_gives_the_designed_roles(tiny_dir):
    cfg = Config.read(tiny_dir)
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, cfg.vocab_size) == (512, 4, 512, VOCAB)
    assert (cfg.index_topk, cfg.candidate_topk_blocks, cfg.candidate_block_size) == (16, 4, 8)
    assert cfg.candidate_topk_blocks * cfg.candidate_block_size - (cfg.candidate_block_size - 1) >= cfg.index_topk
    dspark = Role(0, "dspark", None, None, False, False, False, "local", None)
    assert cfg.roles == (
        Role(0, "swa", None, None, False, False, False, "local", None),
        Role(0, "swa", None, None, False, False, True, "local", None),
        Role(2, "full", 2, 2, False, False, False, "yarn", None),
        Role(2, "reuse", 2, 2, False, False, True, "yarn", None),
        Role(1, "full", 4, 4, True, False, False, "yarn", None),
        Role(1, "reuse", 4, 4, False, False, False, "yarn", 0),
        Role(1, "reindex", 4, 6, False, True, False, "yarn", 1),
        Role(1, "reuse", 4, 6, False, False, False, "yarn", 2),
        dspark, dspark, dspark)


def test_every_tensor_is_the_packs_name_dtype_and_shape_with_one_split_rule(tiny_dir):
    found = capacity.headers(tiny_dir)                  # also checks each tensor's bytes against its shape
    config = json.loads((tiny_dir / "config.json").read_text())
    vision = {"hidden_size": 32, "intermediate_size": 32, "patch_size": 14, "downsample_ratio": 3,
              "num_hidden_layers": 0}
    want = {name: info for name, info in inventory({**config, "vision_config": vision}).items()
            if split.rule(name) != "drop"}
    assert {k: (v["dtype"], v["shape"]) for k, v in found.items()} == {k: (v["dtype"], v["shape"]) for k, v in want.items()}
    for name, info in found.items():
        if split.rule(name) != "engram":
            for rank in (0, 1):
                split.slice_for(rank, 2, name, info["shape"])
    weights = capacity.estimate_weights(tiny_dir, split.weights_estimate)
    assert weights.resident > 0 and weights.mapped == 0


def test_safetensors_reads_every_shard(tiny_dir):
    safetensors = pytest.importorskip("safetensors")
    index = json.loads((tiny_dir / "model.safetensors.index.json").read_text())
    for file in set(index["weight_map"].values()):
        with safetensors.safe_open(tiny_dir / file, framework="numpy") as f:
            assert set(f.keys()) == {k for k, v in index["weight_map"].items() if v == file}
    assert index["metadata"] == {"total_size": 0}       # as the pack writes it


def test_no_nan_bytes_and_scales_the_engine_holds_exactly(tiny_dir):
    for name, info, raw in _tensors(tiny_dir):
        b = np.frombuffer(raw, dtype=np.uint8)
        if info["dtype"] == "F8_E4M3":
            assert not np.isin(b, (0x7F, 0xFF)).any(), name
        elif info["dtype"] == "F8_E8M0" or name == "lm_head.weight_scale":
            assert FOLD[0] <= b.min() and b.max() <= FOLD[1], name
            if name.startswith("mtp.") and ".experts." in name:
                assert int(b.max()) - int(b.min()) <= SPAN, name


def test_exl3_experts_are_mcg_2_bit_with_small_input_scales(tiny_dir):
    for name, info, raw in _tensors(tiny_dir):
        if name.endswith(".mcg"):
            assert info["dtype"] == "I32" and np.frombuffer(raw, "<u4")[0] == 0xCBAC1FED, name
        elif name.endswith(".trellis"):
            assert info["shape"][-1] == 32, name                     # 2 bits a weight
        elif name.endswith(".suh"):                                  # the fp16 down input stays far from overflow
            assert np.abs(np.frombuffer(raw, "<f2")).max() <= 0.025, name


def test_tokenizer_and_engram_tables_fit_the_config(tiny_dir):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer

    from tensorfold.families.deepseek_v41.engram_hash import Hasher, TokenMap
    from tensorfold.families.deepseek_v41.engram_table import Layout

    cfg = Config.read(tiny_dir)
    tok = Tokenizer.from_file(str(tiny_dir / "tokenizer.json"))
    assert tok.get_vocab_size(with_added_tokens=True) == VOCAB
    text = "<｜begin▁of▁sentence｜><｜User｜>w7 W7<｜Assistant｜></think>"
    assert tok.encode(text).ids == [0, tok.token_to_id("<｜User｜>"), tok.token_to_id("w7"), tok.token_to_id("W7"),
                                    tok.token_to_id("<｜Assistant｜>"), tok.token_to_id("</think>")]
    token_map = TokenMap.build(tiny_dir / "tokenizer.json", cfg.engram_compressed_vocab_size)
    assert token_map.table[tok.token_to_id("W7")] == token_map.table[tok.token_to_id("w7")]
    assert token_map.table[cfg.engram_pad_token_id] != cfg.engram_pad_token_id == PAD
    hasher = Hasher(cfg, token_map)                     # the primes' sums equal the tables' rows
    layout = Layout.read(tiny_dir, cfg.engram_layer_ids, hasher.primes.tolist())
    assert [t.rows for t in layout.tables] == list(cfg.engram_num_embeddings)
    assert all(t.wrow == cfg.engram_head_dim for t in layout.tables)

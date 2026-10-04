"""DeepSeek-V4.1-Flash's config: every hyper-parameter read, each layer's role, YaRN's range, reduced layer sets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tensorfold.families.deepseek_v41.config import Config, Role, yarn_bounds

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json"


# the model specification's layer-role table: (layers, ratio, mode, KV + index-K from, top-k from, extra)
SPEC_TABLE = [
    ((0,), 0, "swa", None, None, ""),
    ((1,), 0, "swa", None, None, "engram"),
    ((2,), 2, "full", 2, 2, ""),
    (range(3, 8), 2, "reuse", 2, 2, ""),
    ((8,), 2, "full", 8, 8, ""),
    (range(9, 14), 2, "reuse", 8, 8, ""),
    ((14,), 2, "full", 14, 14, "engram"),
    (range(15, 20), 2, "reuse", 14, 14, ""),
    ((20,), 1, "full", 20, 20, "candidate source"),
    (range(21, 24), 1, "reuse", 20, 20, ""),
    ((24,), 1, "reindex", 20, 24, "restricted"),
    (range(25, 28), 1, "reuse", 20, 24, ""),
    ((28,), 1, "reindex", 20, 28, "restricted"),
    (range(29, 32), 1, "reuse", 20, 28, ""),
    ((32,), 1, "reindex", 20, 32, "restricted"),
    (range(33, 36), 1, "reuse", 20, 32, ""),
    ((36,), 1, "reindex", 20, 36, "restricted"),
    (range(37, 40), 1, "reuse", 20, 36, "taps"),
    (range(40, 43), 0, "dspark", None, None, ""),
]


def _role(ratio=0, mode="swa", kv=None, idx=None, source=False, uses=False, engram=False, tap=None):
    return Role(ratio, mode, kv, idx, source, uses, engram, "yarn" if ratio else "local", tap)


def _spec_table() -> list[Role]:
    rows = []
    for layers, ratio, mode, kv, idx, extra in SPEC_TABLE:
        rows += [_role(ratio, mode, kv, idx, extra == "candidate source", extra == "restricted", extra == "engram",
                       layer - 37 if extra == "taps" else None) for layer in layers]
    return rows


def test_roles_equal_the_specification_row_by_row():
    cfg = Config.read(FIXTURE.parent)
    expected = _spec_table()
    assert len(cfg.roles) == len(expected) == 43
    for layer, (got, want) in enumerate(zip(cfg.roles, expected)):
        assert got == want, f"layer {layer}"


def test_every_hyper_parameter_is_read():
    cfg = Config.read(FIXTURE.parent)
    assert (cfg.vocab_size, cfg.hidden_size, cfg.moe_intermediate_size) == (129280, 5120, 2304)
    assert (cfg.num_hidden_layers, cfg.num_nextn_predict_layers) == (40, 3)
    assert (cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim) == (64, 512, 64)
    assert (cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank, cfg.sliding_window) == (1280, 8, 1024, 128)
    assert (cfg.rms_norm_eps, cfg.max_position_embeddings) == (1e-20, 1048576)
    assert cfg.compress_ratios == (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    assert cfg.kv_source_layer_ids == (2, 8, 14, 20)
    assert cfg.index_source_layer_ids == (2, 8, 14, 20, 24, 28, 32, 36)
    assert (cfg.candidate_source_layer_id, cfg.candidate_topk_blocks, cfg.candidate_block_size) == (20, 2048, 8)
    assert (cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk) == (32, 128, 512)
    assert (cfg.rope_theta, cfg.compress_rope_theta) == (10000, 160000)
    assert (cfg.original_max_position_embeddings, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow) == (65536, 16, 32, 1)
    assert (cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps) == (4, 20, 1e-6)
    assert (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_shared_experts) == (384, 6, 1)
    assert (cfg.scoring_func, cfg.routed_scaling_factor, cfg.swiglu_limit) == ("sqrtsoftplus", 1.5, 10.0)
    assert (cfg.gate_temp, cfg.norm_topk_prob) == (1.0, True)
    assert (cfg.engram_layer_ids, cfg.engram_n_heads, cfg.engram_head_dim, cfg.engram_max_ngram_size) == \
        ((1, 14), 8, 256, 4)
    assert cfg.engram_num_embeddings == (384006168, 384016682)
    assert (cfg.engram_vocab_size, cfg.engram_compressed_vocab_size, cfg.engram_pad_token_id) == (16000000, 99092, 2)
    assert (cfg.dspark_block_size, cfg.dspark_noise_token_id, cfg.dspark_target_layer_ids) == (5, 128799, (37, 38, 39))
    assert (cfg.dspark_markov_rank, cfg.dspark_n_routed_experts, cfg.dspark_num_experts_per_tok) == (256, 128, 3)
    assert (cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id, cfg.image_token_id) == (0, 1, 2, 129264)
    assert (cfg.codebook, cfg.mtp_experts, cfg.fp8_block, cfg.expert_dtype) == ("mcg", "source", (32, 32), "fp4")


def test_yarn_range_is_15_to_25():
    assert Config.read(FIXTURE.parent).yarn == (15, 25)
    assert yarn_bounds(64, 65536, 160000, 32, 1) == (15, 25)


def test_candidate_pool_covers_index_topk():
    cfg = Config.read(FIXTURE.parent)
    block = cfg.candidate_block_size
    assert cfg.candidate_topk_blocks * block - (block - 1) >= cfg.index_topk


def test_a_reduced_layer_set_recomputes_roles_and_moves_the_taps():
    cfg = Config.read(FIXTURE.parent, layers=8)
    assert cfg.num_hidden_layers == 8 and len(cfg.roles) == 11
    assert cfg.roles[:8] == tuple(_spec_table()[:5]) + tuple(_role(2, "reuse", 2, 2, tap=t) for t in range(3))
    assert cfg.roles[8:] == (_role(0, "dspark"),) * 3
    assert cfg.dspark_target_layer_ids == (5, 6, 7)
    assert (cfg.kv_source_layer_ids, cfg.index_source_layer_ids, cfg.candidate_source_layer_id) == ((2,), (2,), -1)
    assert (cfg.engram_layer_ids, cfg.engram_num_embeddings) == ((1,), (384006168,))
    deeper = Config.read(FIXTURE.parent, layers=25)
    assert deeper.roles[20].candidate_source and deeper.roles[24].uses_candidates
    assert [r.tap for r in deeper.roles[22:25]] == [0, 1, 2]
    with pytest.raises(ValueError, match="layers=2"):
        Config.read(FIXTURE.parent, layers=2)


def test_a_missing_field_is_named(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "config.json")
    config = json.loads(FIXTURE.read_text())
    del config["text_config"]["engram_pad_token_id"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="engram_pad_token_id"):
        Config.read(tmp_path)

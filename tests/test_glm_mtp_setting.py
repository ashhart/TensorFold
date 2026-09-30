"""TF_GLM_MTP on CPU: when GLM's CUDA engine loads the MTP head, and that leaving it out leaves its tensors, cache
rows and decode buffers out of the startup estimate. The engine end to end (replies with and without the head) is
tests/cuda/test_glm_engine.py's."""

from __future__ import annotations

import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda.engine import MTP_DEFAULT, mtp_head, without_mtp


def test_the_setting(monkeypatch):
    # auto: on without a drafter, off beside DFlash2 or without drafts
    for value in ("auto", " AUTO "):
        assert mtp_head(False, False, 1, value) is True
        assert mtp_head(True, False, 1, value) is False
        assert mtp_head(False, True, 1, value) is False
    assert mtp_head(True, False, 1, "1") is True and mtp_head(False, True, 1, "1") is True
    assert mtp_head(False, False, 1, "0") is False
    for value in ("", "1", "0"):                      # a checkpoint without the head never has one
        assert mtp_head(False, False, 0, value) is False
    for bad in ("2", "on", "yes", "-1"):
        with pytest.raises(ValueError, match="TF_GLM_MTP"):
            mtp_head(False, False, 1, bad)
    for args in ((True, False, 1), (False, False, 1), (False, True, 1)):       # unset or empty: MTP_DEFAULT
        assert mtp_head(*args, "") is mtp_head(*args, MTP_DEFAULT), args
        monkeypatch.delenv("TF_GLM_MTP", raising=False)
        assert mtp_head(*args) is mtp_head(*args, MTP_DEFAULT), args
    monkeypatch.setenv("TF_GLM_MTP", "1")
    assert mtp_head(True, False, 1) is True


def test_the_weight_estimate_leaves_out_only_the_head():
    base = lambda name, info: (7, 1)                                         # noqa: E731
    t = without_mtp(base, 45)
    assert t("model.language_model.layers.45.mlp.experts.0.gate_proj.trellis", {}) == (0, 0)
    assert t("model.language_model.layers.45.eh_proj.weight", {}) == (0, 0)
    for kept in ("model.language_model.layers.44.mlp.gate.weight", "model.language_model.layers.4.eh_proj.weight",
                 "model.language_model.layers.450.x", "lm_head.weight", "model.language_model.embed_tokens.weight"):
        assert t(kept, {}) == (7, 1), kept


TEXT = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 8,
        "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1024, "kv_lora_rank": 512,
        "moe_intermediate_size": 512, "num_experts_per_tok": 2}


@pytest.mark.parametrize("latent", [True, False])
def test_the_geometry_follows_the_setting_not_the_config(latent):
    """mla_geometry(mtp=False) on a checkpoint with the head is the geometry of one without it, and smaller by the
    head's cache rows (latent and indexer) at every capacity."""

    with_head = {**TEXT, "num_nextn_predict_layers": 1}
    without = {**TEXT, "num_nextn_predict_layers": 0}
    for slots in (2560, 65536, 1 << 20):
        off = geometry.mla_geometry(with_head, 2, 16, latent=latent, mtp=False).bytes_at(slots)
        assert off == geometry.mla_geometry(without, 2, 16, latent=latent).bytes_at(slots)
        on = geometry.mla_geometry(with_head, 2, 16, latent=latent).bytes_at(slots)
        assert on == geometry.mla_geometry(without, 2, 16, latent=latent, mtp=True).bytes_at(slots)
        rows = slots * 512 * 2 if latent else slots * 4 * 512 * 2
        assert on - off > rows + (2 * slots + slots // 4 + 2) * 128 * 2

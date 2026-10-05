"""Flash Next's rank shares: whole quantization groups a rank, the lower ranks one block more; shared key/value heads."""

import json

import pytest

from tensorfold.families.qwen4_exp.cuda.ranks import share


def test_width_shares_are_whole_groups_lower_ranks_first():
    assert [share(640, r, 4) for r in range(4)] == [(0, 192), (192, 384), (384, 512), (512, 640)]
    assert [share(640, r, 2) for r in range(2)] == [(0, 320), (320, 640)]
    assert [share(320, r, 4) for r in range(4)] == [(0, 128), (128, 192), (192, 256), (256, 320)]
    assert [share(512, r, 4, 32) for r in range(4)] == [(0, 128), (128, 256), (256, 384), (384, 512)]
    assert share(640, 0, 1) == (0, 640)


def test_a_width_that_leaves_a_rank_nothing_is_refused():
    with pytest.raises(ValueError, match="no 64-wide block"):
        share(192, 3, 4)
    with pytest.raises(ValueError, match="not whole 64-wide blocks"):
        share(100, 0, 2)


class _Stop(Exception):
    pass


class _Reader:
    """A checkpoint of one attention layer whose every weight row stores its own row index."""

    rows = {"q_proj": 8 * 2 * 16, "k_proj": 2 * 16, "v_proj": 2 * 16}

    def __init__(self, model_dir, device):
        self.where = {}

    def has(self, name):
        return not name.startswith(("language_model.", "model.language_model.", "lm_head.weight_scale"))

    def get(self, name):
        import torch

        if ".mlp." in name:
            raise _Stop                                      # the attention block is loaded by then
        if name.endswith("norm.weight"):
            return torch.ones(64)
        proj = name.rsplit(".", 2)[-2]
        n, k = self.rows.get(proj, 64), 128
        if name.endswith(".weight"):
            return torch.arange(n, dtype=torch.int32)[:, None].expand(n, k // 8).contiguous()
        return torch.zeros(n, k // 32, dtype=torch.bfloat16)

    def layer_names(self, *args):
        return [[]]

    def queue(self, *args):
        pass

    close = drop = release = queue


@pytest.mark.parametrize("world", [2, 4])
def test_mlx_ranks_keep_the_key_value_head_their_queries_read(tmp_path, monkeypatch, world):
    """2 key/value heads over 2 or 4 ranks: each rank loads whole k/v rows of head rank * 2 // world, never none."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda import weights

    t = {"hidden_size": 64, "num_hidden_layers": 1, "layer_types": ["full_attention"], "vocab_size": 256,
         "rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
         "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
         "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "num_experts": 4, "num_experts_per_tok": 2,
         "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128, "eos_token_id": 1}
    (tmp_path / "config.json").write_text(json.dumps(t))
    stacks = []
    monkeypatch.setattr(weights, "_Reader", _Reader)
    monkeypatch.setattr(weights, "stack_q4", lambda parts, layout="frag": stacks.append(parts))
    monkeypatch.setattr(weights, "make_q4", lambda *a, **k: None)
    for rank in range(world):
        stacks.clear()
        with pytest.raises(_Stop):
            weights.load(tmp_path, "cpu", mtp=False, tp=(rank, world))
        q, k, v, _ = next(p for p in stacks if len(p) == 4)
        head = rank * 2 // world
        want = torch.arange(head * 16, (head + 1) * 16, dtype=torch.int32)
        assert torch.equal(k[0][:, 0], want) and torch.equal(v[0][:, 0], want)
        assert torch.equal(q[0][:, 0], torch.arange(rank * 256 // world, (rank + 1) * 256 // world, dtype=torch.int32))

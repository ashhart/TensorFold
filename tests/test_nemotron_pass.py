"""Nemotron's prompt pass on a tiny model: several chunks in one forward give each chunk's rows and every cache state
of the chunks one forward at a time."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from tensorfold.families.nemotron_h import prompt_pass  # noqa: E402


def _tiny():
    from mlx_lm.models.nemotron_h import Model, ModelArgs

    args = ModelArgs(model_type="nemotron_h", vocab_size=512, hidden_size=256, intermediate_size=256,
                     num_hidden_layers=4, max_position_embeddings=4096, num_attention_heads=4, num_key_value_heads=2,
                     attention_bias=False, mamba_num_heads=8, mamba_head_dim=32, mamba_proj_bias=False,
                     ssm_state_size=16, conv_kernel=4, n_groups=2, mlp_bias=False, layer_norm_epsilon=1e-5,
                     use_bias=False, use_conv_bias=True, hybrid_override_pattern=["M", "E", "*", "E"], head_dim=64,
                     moe_intermediate_size=128, moe_shared_expert_intermediate_size=128, n_group=1,
                     n_routed_experts=4, n_shared_experts=1, topk_group=1, num_experts_per_tok=2,
                     norm_topk_prob=True, routed_scaling_factor=2.5)
    mx.random.seed(3)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("sizes", [(40, 40, 40), (64, 17, 90, 33)])
def test_a_pass_gives_each_chunk_its_own_forwards_bits(sizes):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _tiny()
        backbone = model.backbone
        ids = mx.random.randint(0, 512, (1, sum(sizes)), key=mx.random.key(5))
        one = model.make_cache()
        rows, at = [], 0
        for n in sizes:
            rows.append(backbone(ids[:, at:at + n], cache=one))
            at += n
        solo = mx.concatenate(rows, axis=1)
        passed = model.make_cache()
        both = prompt_pass.hidden(backbone, ids, passed, tuple(sizes))
        mx.eval(solo, both)
        assert bool(mx.array_equal(solo, both).item())
        a = [x for c in one for x in c.state if x is not None]
        b = [x for c in passed for x in c.state if x is not None]
        assert len(a) == len(b) and all(bool(mx.array_equal(x, y).item()) for x, y in zip(a, b))
    finally:
        mx.set_default_device(previous)

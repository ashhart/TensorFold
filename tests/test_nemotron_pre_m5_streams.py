"""Nemotron without tensor units (M1-M4): a shared round of any width gives each stream its own round's bits."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402
from tensorfold.kernels.nemotron.lightning.v1 import rows  # noqa: E402

VOCAB = 4096
IDS = [(37 * i + 11) % 4000 + 50 for i in range(400)]


def _tiny():
    """Three blocks (Mamba, attention, MoE) at the checkpoint's widths, random 4-bit weights."""

    from mlx_lm.models.nemotron_h import Model, ModelArgs

    args = ModelArgs(model_type="nemotron_h", vocab_size=VOCAB, hidden_size=2688, intermediate_size=1856,
                     num_hidden_layers=3, max_position_embeddings=4096, num_attention_heads=32,
                     num_key_value_heads=2, attention_bias=False, mamba_num_heads=64, mamba_head_dim=64,
                     mamba_proj_bias=False, ssm_state_size=128, conv_kernel=4, n_groups=8, mlp_bias=False,
                     layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
                     hybrid_override_pattern=["M", "*", "E"], head_dim=128, moe_intermediate_size=1856,
                     moe_shared_expert_intermediate_size=3712, n_group=1, n_routed_experts=32, n_shared_experts=1,
                     topk_group=1, num_experts_per_tok=6, norm_topk_prob=True, routed_scaling_factor=2.5)
    mx.random.seed(3)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def _load(patch):
    from tensorfold.families.nemotron_h.model import NemotronH

    patch.setattr(K, "tensor_units", lambda: False)          # the path rows.install takes on an M1-M4
    return NemotronH(_tiny(), mtp_path=None, drafts=0, tokenizer=SimpleNamespace(encode=lambda text: IDS))


@pytest.fixture(scope="module")
def pre_m5():
    patch = pytest.MonkeyPatch()
    yield _load(patch)
    patch.undo()


def _same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def _solo_and_shared(nem, windows):
    base = nem.make_cache()
    mx.eval(nem.hidden(mx.array([IDS[:48]], dtype=mx.uint32), base))
    copy = LaneEngine.copy_single_cache
    solo = [nem.head(nem.hidden(mx.array([w], dtype=mx.uint32), copy(base)))[0] for w in windows]
    shared = nem.head(nem.hidden_rows(windows, [copy(base) for _ in windows]))
    mx.eval(shared, *solo)
    at = [sum(len(w) for w in windows[:i]) for i in range(len(windows))]
    return [_same(shared[0, a:a + len(w)], alone) for a, w, alone in zip(at, windows, solo)]


def test_the_path_under_test_keeps_wide_rounds(pre_m5):
    assert not pre_m5.lane_matmul and isinstance(pre_m5.model.lm_head, rows.RowLinear)
    assert pre_m5.exact_width == 16 and pre_m5.batch_rows == 128 and max(pre_m5.shared_costs) == 128


@pytest.mark.parametrize("widths", [(8, 8), (9, 9), (5, 5, 5, 5), (16, 16, 16), (1, 16, 3, 12, 7, 2)])
def test_shared_rounds_equal_each_stream_alone(pre_m5, widths):
    windows = [IDS[48 + 20 * i:48 + 20 * i + w] for i, w in enumerate(widths)]
    assert _solo_and_shared(pre_m5, windows) == [True] * len(widths)


def test_load_refuses_rounds_wider_than_its_exact_ones(monkeypatch):
    """With MLX's kernel past 16 rows (the old routing), the load stops shared rounds at the widest exact window."""

    call = rows.RowLinear.__call__
    monkeypatch.setattr(rows.RowLinear, "__call__", lambda self, x: nn.QuantizedLinear.__call__(self, x)
                        if x.size // x.shape[-1] > rows.MAX_ROWS else call(self, x))
    nem = _load(monkeypatch)
    assert nem.batch_rows == nem.exact_width == 16 and not nem.shared_costs

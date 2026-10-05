"""The calibrated Nemotron-H EXL3 checkpoint is a CUDA, not MLX, format."""

import pytest

from tensorfold import families
from tensorfold.families import nemotron_h


def test_nemotron_exl3_is_readable_only_on_cuda():
    config = {
        "model_type": "nemotron_h",
        "quantization_config": {
            "quant_method": "exl3", "bits": 4.09, "head_bits": 6,
            "mtp_bits": 4, "codebook": "mul1",
        },
    }
    family = families.families()["nemotron_h"]
    families.require_readable(family, config, "cuda")
    assert nemotron_h.refusal(config) is None
    with pytest.raises(ValueError, match="does not read this checkpoint"):
        families.require_readable(family, config, "mlx")


def test_nemotron_dense_dispatch_uses_exl3_decode_and_prefill():
    import torch
    from tensorfold.families.nemotron_h.cuda import glue

    class Projection:
        def __call__(self, x):
            return x + 1

        def prefill(self, x):
            return x + 2

    x = torch.zeros((2, 128), dtype=torch.bfloat16)
    assert torch.equal(glue.dense(x, Projection()), x + 1)
    assert torch.equal(glue.prefill_dense(x, Projection()), x + 2)
    with pytest.raises(ValueError, match="partial"):
        glue.dense(x, Projection(), f32=True)


def test_nemotron_exl3_embedding_reads_plain_rows():
    import torch
    from tensorfold.families.nemotron_h.cuda.engine import Engine

    engine = object.__new__(Engine)
    engine.w = type("Weights", (), {"embed": torch.arange(15, dtype=torch.bfloat16).view(5, 3)})()
    ids = torch.tensor([4, 2, 4], dtype=torch.int32)
    assert torch.equal(engine.embed(ids), engine.w.embed[[4, 2, 4]])


def test_nemotron_exl3_padded_projection_keeps_logical_output():
    import torch
    from tensorfold.families.nemotron_h.cuda.exl3_weights import Projection

    class Packed:
        n = 256
        k = 128

        def __call__(self, x):
            return torch.arange(self.n, dtype=x.dtype).expand(x.shape[0], -1)

        prefill = __call__

    linear = Projection(Packed(), 192)
    x = torch.zeros((3, 128), dtype=torch.bfloat16)
    assert linear.n == 192
    assert torch.equal(linear(x), Packed()(x)[:, :192])
    assert torch.equal(linear.prefill(x), Packed()(x)[:, :192])


def test_nemotron_exl3_attention_keeps_qkv_order():
    import torch
    from tensorfold.families.nemotron_h.cuda.exl3_weights import Concat

    class Part:
        def __init__(self, value):
            self.value = value
            self.n = value

        def __call__(self, x):
            return torch.full((x.shape[0], self.n), self.value, dtype=x.dtype)

        prefill = __call__

    qkv = Concat((Part(4), Part(2), Part(1)))
    x = torch.empty((1, 128), dtype=torch.bfloat16)
    assert qkv.n == 7
    assert torch.equal(qkv(x), torch.tensor([[4, 4, 4, 4, 2, 2, 1]], dtype=x.dtype))
    assert torch.equal(qkv.prefill(x), qkv(x))


def test_nemotron_exl3_reader_loads_groups_across_shards(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    from tensorfold.families.nemotron_h.cuda.exl3_weights import Parts

    prefix = "backbone.layers.0.mixer.in_proj"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    save_file({prefix + ".trellis": torch.zeros((8, 8, 64), dtype=torch.int16),
               "backbone.embeddings.weight": torch.arange(256, dtype=torch.bfloat16).view(2, 128)},
              str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({prefix + ".suh": torch.ones(128, dtype=torch.float16),
               prefix + ".svh": torch.ones(128, dtype=torch.float16)},
              str(tmp_path / "model-00002-of-00002.safetensors"))
    with Parts(tmp_path, "cpu") as parts:
        assert parts.plain("backbone.embeddings.weight").shape == (2, 128)
        linear = parts.projection(prefix)
        assert (linear.n, linear.packed.layer.bits, linear.packed.layer.codebook) == (128, 4, "3inst")
        assert parts.unused() == ([], [])


def test_nemotron_exl3_shared_expert_uses_gateless_relu_squared():
    import torch
    from tensorfold.families.nemotron_h.cuda.glue import relu2_mlp

    class Up:
        def __call__(self, x):
            return x - 2

        prefill = __call__

    class Down:
        def __call__(self, x):
            return x * 3

        prefill = __call__

    x = torch.tensor([[-1.0, 3.0, 5.0]], dtype=torch.bfloat16)
    want = torch.tensor([[0.0, 3.0, 27.0]], dtype=torch.bfloat16)
    assert torch.equal(relu2_mlp(x, Up(), Down()), want)
    assert torch.equal(relu2_mlp(x, Up(), Down(), prefill=True), want)


def test_nemotron_exl3_requires_explicit_serial_single_gpu(tmp_path, monkeypatch):
    import json
    from tensorfold.families.nemotron_h import cuda_engine
    from tensorfold.families.nemotron_h.cuda import app

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h", "quantization_config": {"quant_method": "exl3", "bits": 4.09},
    }))
    with pytest.raises(ValueError, match="--no-drafts"):
        cuda_engine(tmp_path)
    with pytest.raises(ValueError, match="two-rank"):
        cuda_engine(tmp_path, tp=2, no_drafts=True)

    class StubEngine:
        def __init__(self, path, **kwargs):
            self.path, self.options = path, kwargs

    monkeypatch.setattr(app, "NemotronEngine", StubEngine)
    engine = cuda_engine(tmp_path, no_drafts=True)
    assert engine.path == tmp_path and engine.options["drafts"] == 0


def test_nemotron_weight_loader_selects_exl3_without_rewriting_weights(tmp_path, monkeypatch):
    import json
    from tensorfold.families.nemotron_h.cuda import exl3_weights, weights

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "nemotron_h", "quantization_config": {"quant_method": "exl3", "bits": 4.09},
    }))
    expected = object()
    calls = []

    def fake_load(directory, device):
        calls.append((directory, device))
        return expected

    monkeypatch.setattr(exl3_weights, "load", fake_load, raising=False)
    assert weights.load(tmp_path, "cpu", mtp=False) is expected
    assert calls == [(tmp_path, "cpu")]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA only")
def test_nemotron_calibrated_exl3_checkpoint_loads_on_one_gpu():
    from pathlib import Path
    import torch
    from tensorfold.families.nemotron_h.cuda import weights

    model = Path("/home/neo/ai/exl3/nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/quants/4.00bpw-hq-cal")
    if not (model / "model.safetensors.index.json").is_file():
        pytest.skip("local calibrated checkpoint unavailable")
    loaded = weights.load(model, mtp=False)
    assert len(loaded.blocks) == loaded.config.pattern.count("M") + loaded.config.pattern.count("*") + loaded.config.pattern.count("E")
    assert loaded.config.pattern.count("E") == 23
    assert loaded.mtp is None
    assert isinstance(loaded.embed, torch.Tensor)
    assert loaded.head.n == loaded.config.vocab
    assert loaded.blocks[0].mamba.in_proj.n == loaded.config.proj_dim
    assert loaded.blocks[1].moe.experts.count == loaded.config.experts
    del loaded
    torch.cuda.empty_cache()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA only")
def test_nemotron_exl3_engine_routes_and_adds_one_shared_expert():
    from types import SimpleNamespace
    import torch
    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.nemotron_h.cuda.engine import Engine
    from tensorfold.families.nemotron_h.cuda.weights import MoE

    t = torch.zeros((16, 16, 64), dtype=torch.int16, device="cuda")
    scale = torch.ones(256, dtype=torch.float16, device="cuda")
    pairs = [(t.clone(), scale, scale) for _ in range(2)]
    ex = experts.prepare_nemotron(pairs, pairs, "mul1", intermediate_size=192)
    engine = object.__new__(Engine)
    engine.c = SimpleNamespace(top_k=2, scaling=1.0, norm_topk=True, hidden=256)
    engine.ns = 4
    engine.pick = torch.empty((1, 4), dtype=torch.int32, device="cuda")
    engine.wts = torch.empty((1, 4), dtype=torch.float32, device="cuda")
    engine.exl3_window = experts.NemotronScratch(ex, 1, 4)

    class Up:
        def __call__(self, x):
            return x

        prefill = __call__

    class Down:
        def __call__(self, x):
            return 2 * x

        prefill = __call__

    moe = MoE(router=torch.zeros((2, 256), dtype=torch.bfloat16, device="cuda"),
              bias=torch.zeros(2, device="cuda"), experts=ex, shared_up=Up(), shared_down=Down())
    x = torch.ones((1, 256), dtype=torch.bfloat16, device="cuda")
    kind, per_slot, wts = engine.moe(moe, x, 1)
    assert kind == "moe" and per_slot.shape == (4, 256) and wts.shape == (1, 4)
    assert torch.equal(per_slot[2], torch.full((256,), 2.0, device="cuda"))
    assert torch.count_nonzero(per_slot[3]) == 0


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA only")
def test_nemotron_calibrated_exl3_prefill_produces_a_token():
    from pathlib import Path
    import torch
    from tensorfold.families.nemotron_h.cuda.engine import Engine
    from tensorfold.families.nemotron_h.cuda.weights import load

    model = Path("/home/neo/ai/exl3/nvidia-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/quants/4.00bpw-hq-cal")
    if not (model / "model.safetensors.index.json").is_file():
        pytest.skip("local calibrated checkpoint unavailable")
    weights = load(model, mtp=False)
    engine = Engine(weights, max_len=512, graphs=False, prefill_rows=32)
    engine.set_sampling(None)
    engine.prefill_chunk([1, 300, 301])
    token = engine.prefill_token()
    assert isinstance(token, int) and 0 <= token < weights.config.vocab
    assert torch.isfinite(engine.p_hidden[:3]).all()
    logits = engine.forward([token])
    next_token = engine.tokens()[0]
    assert 0 <= next_token < weights.config.vocab
    assert torch.isfinite(logits).all()
    engine.commit(1)
    del engine, weights
    torch.cuda.empty_cache()


def test_nemotron_stops_at_generation_config_eos_as_well_as_model_eos(tmp_path):
    import json
    from tensorfold.families.nemotron_h.cuda.weights import Config

    config = {
        "hidden_size": 256, "vocab_size": 128, "hybrid_override_pattern": ["M"],
        "num_attention_heads": 2, "num_key_value_heads": 2, "mamba_num_heads": 2,
        "mamba_head_dim": 128, "n_groups": 2, "ssm_state_size": 16,
        "conv_kernel": 4, "n_routed_experts": 2, "num_experts_per_tok": 2,
        "moe_intermediate_size": 192, "moe_shared_expert_intermediate_size": 384,
        "eos_token_id": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 11]}))
    assert Config.read(tmp_path).eos == (2, 11)

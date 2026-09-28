"""mlx-lm's converted layout of GLM-5.3-Flash: the tiny checkpoint written that way loads, matches and drafts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import layouts, linear, weights  # noqa: E402
from tensorfold.families.glm5_next import mtp as glm_mtp  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402

N = TEXT["num_hidden_layers"]
# tensors re-quantized at other widths, as mixed conversions store them (routed experts stay 4-bit)
OVERRIDES = {
    "language_model.model.layers.0.self_attn.q_proj": 8, "language_model.model.layers.0.self_attn.k_proj": 8,
    "language_model.model.layers.0.self_attn.v_proj": 8, "language_model.model.layers.0.self_attn.forget_gate.f_a_proj": 8,
    "language_model.model.layers.0.self_attn.forget_gate.f_b_proj": 8, "language_model.model.layers.0.self_attn.g_a_proj": 8,
    "language_model.model.layers.0.self_attn.g_b_proj": 8, "language_model.model.layers.0.self_attn.b_proj": 8,
    "language_model.model.layers.0.self_attn.o_proj": 6,
    "language_model.model.layers.1.self_attn.q_proj": 5, "language_model.model.layers.1.self_attn.k_proj": 5,
    "language_model.model.layers.1.self_attn.v_proj": 5, "language_model.model.layers.1.self_attn.forget_gate.f_a_proj": 5,
    "language_model.model.layers.1.self_attn.forget_gate.f_b_proj": 5, "language_model.model.layers.1.self_attn.g_a_proj": 5,
    "language_model.model.layers.1.self_attn.g_b_proj": 5, "language_model.model.layers.1.self_attn.b_proj": 5,
    "language_model.model.layers.0.mlp.gate_proj": 5, "language_model.model.layers.0.mlp.up_proj": 5,
    "language_model.model.layers.0.mlp.down_proj": 6,
    # an MLA layer whose stacked inputs mix bits (q_a 6 | kv_a 6 | indexer 8; q_b 6 | indexer q 8)
    "language_model.model.layers.3.self_attn.q_a_proj": 6, "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa": 6,
    "language_model.model.layers.3.self_attn.q_b_proj": 6, "language_model.model.layers.3.self_attn.indexer.wq_b": 8,
    "language_model.model.layers.3.self_attn.indexer.wk": 8, "language_model.model.layers.3.self_attn.indexer.weights_proj": 8,
    "language_model.model.layers.3.self_attn.embed_q": 5, "language_model.model.layers.3.self_attn.unembed_out": 5,
    "language_model.model.layers.3.self_attn.o_proj": 8,
    "language_model.model.layers.3.mlp.shared_experts.gate_proj": 8, "language_model.model.layers.3.mlp.shared_experts.up_proj": 8,
    "language_model.model.layers.3.mlp.shared_experts.down_proj": 8,
    "language_model.mtp.0.block.self_attn.embed_q": 8, "language_model.mtp.0.block.self_attn.unembed_out": 8,
    "language_model.lm_head": 8, "language_model.model.embed_tokens": 8,
}


def _requant(t: dict, prefix: str, bits: int) -> None:
    w = mx.dequantize(t[f"{prefix}.weight"], t[f"{prefix}.scales"], t[f"{prefix}.biases"], group_size=64, bits=4)
    t[f"{prefix}.weight"], t[f"{prefix}.scales"], t[f"{prefix}.biases"] = mx.quantize(w, group_size=64, bits=bits)


def to_mlxlm(vontra: dict) -> dict:
    """The fake's tensors in mlx-lm's layout: the conversion mlx-lm's sanitize performs, run backwards."""

    c = TEXT
    h, nope = c["num_attention_heads"], c["qk_nope_head_dim"]
    out: dict = {}
    conv: dict = {}
    for name, v in vontra.items():
        if name.startswith("lm_head."):
            out["language_model." + name] = v
            continue
        assert name.startswith("model.language_model.")
        short = name[len("model.language_model."):]
        if short.startswith(f"layers.{N}."):                             # the MTP layer -> mtp.0.*
            rest = short[len(f"layers.{N}."):]
            if rest == "shared_head.norm.weight":
                out["language_model.mtp.0.norm.weight"] = v
            elif rest.split(".")[0] in ("eh_proj", "enorm", "hnorm"):
                if rest == "eh_proj.weight":
                    out["language_model.mtp.0.eh_proj.weight"] = mx.dequantize(
                        v, vontra[f"{name[:-7]}.scales"], vontra[f"{name[:-7]}.biases"], group_size=64, bits=4
                    ).astype(mx.bfloat16)
                elif not rest.startswith("eh_proj."):
                    out["language_model.mtp.0." + rest] = v
            else:
                out["language_model.mtp.0.block." + rest] = v
            continue
        if ".hc_attn_" in short or ".hc_ffn_" in short:
            short = short.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")
        for p in ("A_log", "dt_bias", "f_a_proj.weight", "f_a_proj.scales", "f_a_proj.biases", "f_b_proj.weight",
                  "f_b_proj.scales", "f_b_proj.biases"):
            if short.endswith(".self_attn." + p):
                short = short[: -len(p)] + "forget_gate." + p
        if short.endswith("_conv1d.weight"):
            layer, which = short[: short.index(".self_attn.")], short[short.index(".self_attn.") + 11]
            conv.setdefault(layer, {})[which] = v                     # [C, 1, T]
            continue
        out["language_model.model." + short] = v
    for layer, parts in conv.items():
        fused = mx.concatenate([parts["q"], parts["k"], parts["v"]], axis=0)   # [3 C, 1, T]
        out[f"language_model.model.{layer}.self_attn.conv1d.weight"] = mx.contiguous(fused.moveaxis(2, 1))  # [3 C, T, 1]
    # kv_b_proj -> embed_q (key half transposed, re-quantized along nope) + unembed_out (value half as stored)
    for name in [n for n in out if n.endswith(".self_attn.kv_b_proj.weight")]:
        prefix = name[: -len(".weight")]
        w = mx.dequantize(out.pop(name), out.pop(prefix + ".scales"), out.pop(prefix + ".biases"), group_size=64, bits=4)
        w = w.reshape(h, nope + c["v_head_dim"], -1)
        wk = mx.contiguous(w[:, :nope, :].swapaxes(-1, -2))
        wv = mx.contiguous(w[:, nope:, :])
        base = prefix[: -len("kv_b_proj")]
        for n, m in (("embed_q", wk), ("unembed_out", wv)):        # quantized once, at their own bits
            out[base + n + ".weight"], out[base + n + ".scales"], out[base + n + ".biases"] = mx.quantize(
                m, 64, OVERRIDES.get(base + n, 4))
    for prefix, bits in OVERRIDES.items():
        if not prefix.endswith(("embed_q", "unembed_out")):
            _requant(out, prefix, bits)
    # a vision tower the text model never reads
    out["vision_model.patch_embed.proj.weight"] = mx.zeros((8, 3, 2, 2), dtype=mx.bfloat16)
    return out


def write_mlxlm_checkpoint(folder: Path, seed: int = 0, overrides: dict | None = None) -> Path:
    src = write_checkpoint(folder / "vontra", seed=seed)
    vontra: dict = {}
    for shard in sorted(src.glob("*.safetensors")):
        vontra.update(mx.load(str(shard)))
    global OVERRIDES
    saved = OVERRIDES
    OVERRIDES = saved if overrides is None else overrides
    try:
        t = to_mlxlm(vontra)
    finally:
        OVERRIDES = saved
    mx.eval(t)
    dst = folder / "mlxlm"
    dst.mkdir()
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(dst / shard), {k: t[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    quant = {"bits": 4, "group_size": 64, "mode": "affine"}
    quant.update({k: {"bits": b, "group_size": 64, "mode": "affine"} for k, b in (saved if overrides is None else overrides).items()})
    config = {"model_type": "glm5_next", "text_config": TEXT, "quantization": quant}
    (dst / "config.json").write_text(json.dumps(config))
    return dst


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        root = tmp_path_factory.mktemp("glm5layouts")
        mlxlm = write_mlxlm_checkpoint(root)
        return root / "vontra", mlxlm
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def clean_pair(tmp_path_factory):
    """The twin with only embed_q / unembed_out re-quantized, at 8 bits: a layout mistake shows as a large error."""

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        root = tmp_path_factory.mktemp("glm5clean")
        keys = [f"language_model.model.layers.{i}.self_attn.{n}" for i in (3, 5) for n in ("embed_q", "unembed_out")]
        keys += [f"language_model.mtp.0.block.self_attn.{n}" for n in ("embed_q", "unembed_out")]
        mlxlm = write_mlxlm_checkpoint(root, overrides={k: 8 for k in keys})
        return root / "vontra", mlxlm
    finally:
        mx.set_default_device(previous)


def tokens(n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(6, TEXT["vocab_size"], size=n)]


def test_names_map_onto_the_vontra_short_names():
    n = 45
    assert layouts.canonical("language_model.model.layers.3.attn_hc.fn", n) == "layers.3.hc_attn_fn"
    assert layouts.canonical("language_model.model.layers.3.ffn_hc.scale", n) == "layers.3.hc_ffn_scale"
    assert layouts.canonical("language_model.model.layers.0.self_attn.forget_gate.A_log", n) == "layers.0.self_attn.A_log"
    assert layouts.canonical("language_model.model.layers.0.self_attn.forget_gate.f_b_proj.scales", n) == "layers.0.self_attn.f_b_proj.scales"
    assert layouts.canonical("language_model.mtp.0.block.self_attn.embed_q.weight", n) == "layers.45.self_attn.embed_q.weight"
    assert layouts.canonical("language_model.mtp.0.eh_proj.weight", n) == "layers.45.eh_proj.weight"
    assert layouts.canonical("language_model.mtp.0.norm.weight", n) == "layers.45.shared_head.norm.weight"
    assert layouts.canonical("language_model.lm_head.weight", n) == "lm_head.weight"
    assert layouts.canonical("language_model.model.embed_tokens.scales", n) == "embed_tokens.scales"
    assert layouts.canonical("vision_model.blocks.0.attn.qkv.weight", n) is None
    assert layouts.canonical("language_model.mtp.1.block.mlp.gate.weight", n) is None
    # vontra names pass through
    assert layouts.canonical("model.language_model.layers.45.eh_proj.weight", n) == "layers.45.eh_proj.weight"
    assert layouts.canonical("model.language_model.layers.0.self_attn.q_conv1d.weight", n) == "layers.0.self_attn.q_conv1d.weight"
    assert layouts.canonical("lm_head.biases", n) == "lm_head.biases"


def test_mlxlm_checkpoint_is_detected_and_has_mtp(pair):
    from tensorfold import families
    from tensorfold.families import glm5_next

    vontra, mlxlm = pair
    assert layouts.detect(json.loads((mlxlm / "model.safetensors.index.json").read_text())["weight_map"]) == layouts.MLXLM
    assert layouts.detect(json.loads((vontra / "model.safetensors.index.json").read_text())["weight_map"]) == layouts.VONTRA
    assert families.detect(mlxlm).module == "tensorfold.families.glm5_next"
    glm5_next.check(mlxlm)
    assert glm5_next.has_mtp(mlxlm) and glm5_next.has_mtp(vontra)


def test_mlxlm_layout_loads_with_mixed_bits(pair):
    _, mlxlm = pair
    model = weights.load_backbone(mlxlm)
    assert model.weights.layout == layouts.MLXLM
    kinds = ["kda" if layer.is_linear else "mla" for layer in model.layers]
    assert kinds == ["kda", "kda", "kda", "mla", "kda", "mla"]
    kda0 = model.layers[0].attn
    assert kda0.in_proj.bits == 8 and kda0.f_b.bits == 8 and kda0.o_proj.bits == 6 and kda0.taps == 4
    assert model.layers[1].attn.in_proj.bits == 5 and model.layers[4].attn.in_proj.bits == 4
    assert model.layers[0].mlp.gate_up.bits == 5 and model.layers[0].mlp.down.bits == 6
    mla = model.layers[3].attn
    assert mla.wk_t and mla.wk.bits == 5 and mla.wv.bits == 5
    assert mla.wk.weight.shape[:2] == (2, 128) and mla.wv.weight.shape[:2] == (2, 64)   # [H, rank, .], [H, v, .]
    assert isinstance(mla.x_proj, linear.QSplit) and isinstance(mla.qr_proj, linear.QSplit)   # 6 | 6 | 8 | 8 and 6 | 8
    assert mla.q_a.bits == 6 and mla.ik_proj.bits == 8 and mla.q_b.bits == 6 and mla.iq.bits == 8
    assert isinstance(model.layers[5].attn.x_proj, linear.Q) and not model.layers[5].attn.wk_t is False
    assert model.layers[3].mlp.shared.gate_up.bits == 8
    assert model.lm_head.bits == 8 and model.embed.bits == 8
    # fp32 tensors holding bf16 values go bf16 losslessly; genuine fp32 values stay fp32
    exact = mx.random.normal((24, 64)).astype(mx.bfloat16).astype(mx.float32)
    assert linear.bf16_if_exact(exact).dtype == mx.bfloat16
    assert linear.bf16_if_exact(exact + 1e-6).dtype == mx.float32
    assert model.layers[0].attn_hc.fn.dtype == mx.float32
    head = glm_mtp.load(model)
    assert isinstance(head.eh_proj, linear.Dense) and head.eh_proj.weight.dtype == mx.bfloat16


def _logits(model, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    whole = model.make_cache()
    a = model.head(model.hidden(mx.array([ids]), whole))[0, -1]
    step = model.make_cache()
    for t in ids:
        b = model.head(model.hidden(mx.array([[t]]), step))[0, -1]
    return np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))


@pytest.mark.parametrize("length", [9, 40])   # 40 > index_topk (16): the sparse selection with pooled blocks
def test_mlxlm_layout_computes_the_vontra_function(clean_pair, length):
    """The same weights in both layouts give the same logits to the 8-bit pair's rounding, prompt and decode path."""

    vontra, mlxlm = clean_pair
    ids = tokens(length)
    va, vb = _logits(weights.load_backbone(vontra), ids)
    ma, mb = _logits(weights.load_backbone(mlxlm), ids)
    for ref, got in ((va, ma), (vb, mb)):
        assert int(ref.argmax()) == int(got.argmax())
        # the 8-bit pair's rounding moves logits a little; a transposed or misread map moves them whole
        assert np.max(np.abs(ref - got)) < 0.06 * np.max(np.abs(ref)) + 0.05, np.max(np.abs(ref - got))


def test_absorbed_pair_is_kv_b_in_the_other_orientation(clean_pair):
    """Absorb and unabsorb agree across layouts: embed_q is read transposed, unembed_out as stored."""

    vontra, mlxlm = clean_pair
    a, b = weights.load_backbone(vontra).layers[3].attn, weights.load_backbone(mlxlm).layers[3].attn
    assert not a.wk_t and b.wk_t

    def rel(x: mx.array, y: mx.array) -> float:
        x, y = np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
        return float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-6))

    q = mx.random.normal((a.heads, 5, a.nope)).astype(mx.bfloat16)
    out = mx.random.normal((a.heads, 5, a.rank)).astype(mx.bfloat16)
    assert rel(a.absorb(q), b.absorb(q)) < 0.05          # measured 0.021: 8-bit re-quantization + bf16 outputs
    assert rel(a.unabsorb(out), b.unabsorb(out)) < 0.05
    # and the orientation matters: the transposed reading of the same tensor is not the same map
    assert rel(a.absorb(q), mx.quantized_matmul(q, b.wk.weight, b.wk.scales, b.wk.biases, transpose=False,
                                                 group_size=b.wk.group, bits=b.wk.bits)
               if b.wk.ins == a.rank else a.absorb(q) * 0) > 0.02 or b.wk.ins != a.rank


@pytest.mark.parametrize("length", [9, 40])
def test_mixed_bits_prefill_path_agrees_with_decode_path(pair, length):
    """With 8-, 6- and 5-bit tensors beside 4-bit ones, prompt and decode paths agree and pick the twin's token."""

    vontra, mlxlm = pair
    ids = tokens(length)
    va, _ = _logits(weights.load_backbone(vontra), ids)
    ma, mb = _logits(weights.load_backbone(mlxlm), ids)
    assert int(ma.argmax()) == int(mb.argmax()) == int(va.argmax())
    assert np.max(np.abs(ma - mb)) < 0.05 * np.max(np.abs(mb)) + 0.05


def test_mlxlm_decode_rows_give_one_row_bits(pair):
    _, mlxlm = pair
    runtime = GLMFlash(weights.load_backbone(mlxlm), check=True)
    assert runtime.multi_row_exact, runtime.check_report


def test_mlxlm_mtp_drafts_change_speed_only(pair):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.glm5_next import engine_settings

    _, mlxlm = pair
    model = weights.load_backbone(mlxlm)
    head = glm_mtp.load(model)
    drafted = GLMFlash(model, head, drafts=3)
    serial = GLMFlash(model, None, drafts=0)
    assert drafted.mtp is not None and serial.mtp is None
    prompt = tokens(21, seed=4)
    out = []
    for runtime in (drafted, serial):
        engine = LaneEngine(runtime, **engine_settings(runtime))
        assert engine.family
        stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=24)
        engine.add_stream(stream)
        while engine.active_count:
            engine.step()
        out.append((engine, stream))
    assert out[0][0].family_mtp and not out[1][0].family_mtp
    assert out[0][0].drafted > 0
    assert out[0][1].emitted == out[1][1].emitted


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mixed_bits_keep_every_rows_bits_in_windows_and_shared_rounds(pair, device):
    """The mixed-bit twin: windows up to 16 rows and streams sharing rounds each keep their own call's bits."""

    from test_glm5_next_family import _run_streams

    from tensorfold.engine.exact_sampling import Sampling

    if device == "gpu":
        if not mx.metal.is_available():
            pytest.skip("needs Metal")
        mx.set_default_device(mx.gpu)
    model = weights.load_backbone(pair[1])
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    if device == "cpu":
        # MLX's CPU rms_norm (fp32) gives a row other bits once a call holds 8 rows or more: CPU rounds stay under 8
        runtime.exact_width = runtime.batch_rows = min(runtime.exact_width, 7)
    if device == "gpu":                             # the row kernels' claim: every width and stream count on Metal
        assert runtime.exact_width == 16 and runtime.streams_exact and runtime.max_streams > 1, runtime.check_report
    specs = [(tokens(21, seed=4), 20, None, True), (tokens(9, seed=5), 14, Sampling(seed=3, temperature=0.8), True),
             (tokens(33, seed=6), 17, None, False)]
    alone, _ = _run_streams(runtime, specs, together=False)
    shared, engine = _run_streams(runtime, specs, together=True)
    assert shared == alone and engine.drafted > 0
    assert engine._shared_rounds > 0 or not runtime.streams_exact

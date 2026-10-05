"""The Q8_0 converter's MTP head from the original FP8 checkpoint: e4m3 decoding, 128x128 block scales, the 8-bit
re-encoding's bound, a grafted head the Mac engine drafts with, and every other output byte left as it was."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_q8_0_gguf_to_mlx as tool  # noqa: E402
import glm5_fakes  # noqa: E402
from test_glm5_next_family import _run_engine, tokens  # noqa: E402
from tensorfold.families import glm5_next  # noqa: E402
from tensorfold.families.glm5_next import weights  # noqa: E402
from tensorfold.families.glm5_next import mtp as glm_mtp  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402

# hidden size 256: the head's FP8 linears span two 128x128 blocks along one axis or the other
TEXT = {**glm5_fakes.TEXT, "hidden_size": 256}
N = TEXT["num_hidden_layers"]
ORIG = "model.language_model."
HEAD = f"{tool.P}layers.{N}."


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _bound(scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Half a step, plus float32 rounding in the encode and in MLX's scale * q + bias."""
    return np.repeat(0.5 * scales + 2.0 ** -20 * (np.abs(biases) + 255 * scales), 32, axis=-1)


def _dequantize(q, s, b) -> np.ndarray:
    return np.array(mx.dequantize(mx.array(q), mx.array(s), mx.array(b), group_size=32, bits=8))


def test_e4m3_table_is_mlx_s_decoding():
    theirs = np.array(mx.from_fp8(mx.array(np.arange(256, dtype=np.uint8)), dtype=mx.float32))
    assert np.array_equal(tool.E4M3, theirs, equal_nan=True)
    assert [int(c) for c in np.nonzero(np.isnan(tool.E4M3))[0]] == [0x7F, 0xFF]
    assert np.signbit(tool.E4M3[0x80]) and tool.E4M3[0x7E] == 448.0 and tool.E4M3[1] == 2.0 ** -9


def test_block_scales_cover_partial_edge_blocks():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 256, (200, 300), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x01
    scale = rng.uniform(0.5, 2.0, (2, 3)).astype(np.float32) * 1e-3          # [ceil(200/128), ceil(300/128)]
    want = (np.array(mx.from_fp8(mx.array(codes), dtype=mx.float32))
            * scale[np.arange(200)[:, None] // 128, np.arange(300)[None, :] // 128])
    assert np.array_equal(tool.fp8_values(codes, scale), want)
    with pytest.raises(ValueError, match="block scales"):
        tool.fp8_values(codes, scale[:, :2])
    with pytest.raises(ValueError, match="block scales"):
        tool.fp8_values(codes, np.ones((2, 2, 1), np.float32))
    nan = codes.copy()
    nan[199, 299] = 0xFF
    with pytest.raises(ValueError, match="NaN"):
        tool.fp8_values(nan, scale)
    with pytest.raises(ValueError, match="non-finite"):
        tool.fp8_values(np.full_like(codes, 0x70), np.full((2, 3), 3e38, np.float32))   # 128 * 3e38


def test_8bit_groups_stay_within_half_a_step():
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((16, 256)) * 0.05).astype(np.float32)
    w[0, :32] = 0.3                                                         # constant group
    w[1, :32] = rng.uniform(-448.0, 448.0, 32)
    w[2, :32] *= 1e-30
    w[3, :32] = np.abs(w[3, :32]) + 7.0                                     # far from zero
    q, s, b = tool.affine8(w)
    assert q.dtype == np.uint32 and q.shape == (16, 64) and s.dtype == b.dtype == np.float32 and s.shape == (16, 8)
    got = _dequantize(q, s, b)
    assert np.all(np.abs(got - w) <= _bound(s, b))
    assert np.array_equal(got[0, :32], w[0, :32])
    codes = q.view(np.uint8).reshape(16, 8, 32)
    assert np.all(codes[1:].min(-1) == 0) and np.all(codes[1:].max(-1) == 255)   # each group spans the range
    with pytest.raises(ValueError, match="groups of 32"):
        tool.affine8(w[:, :48])
    w[5, 40] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tool.affine8(w)


# -- a tiny GGUF and a tiny original checkpoint, from the fake model --------------------------------------------------

def _floats(folder: Path) -> dict[str, np.ndarray]:
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    t = {}
    for shard in sorted(set(index.values())):
        t.update(mx.load(str(folder / shard)))
    out = {}
    for name, v in t.items():
        base = name[: -len(".weight")]
        if name.endswith((".scales", ".biases")):
            continue
        if f"{base}.scales" in t:
            v = mx.dequantize(v, t[f"{base}.scales"], t[f"{base}.biases"], group_size=64, bits=4)
        out[name] = np.array(v.astype(mx.float32))
    return out


def _q8_0(a: np.ndarray) -> np.ndarray:
    x = a.reshape(*a.shape[:-1], -1, 32).astype(np.float32)
    blocks = np.zeros(x.shape[:-1], tool.Q8_DTYPE)
    blocks["d"] = (np.abs(x).max(-1) / 127).astype(np.float16)
    d = blocks["d"].astype(np.float32)[..., None]
    blocks["q"] = np.clip(np.rint(np.divide(x, d, out=np.zeros_like(x), where=d > 0)), -127, 127)
    return blocks


def _write_gguf(path: Path, floats: dict[str, np.ndarray]) -> None:
    """The backbone as llama.cpp's glm5next Q8_0 GGUF holds it (absorbed k_b / v_b, ssm_a, stacked experts)."""
    names = {c + (".weight" if kind == "q" else ""): (g, kind == "q") for g, (c, kind) in tool.LAYER.items()}
    tensors, experts = {}, {}
    h, nope = TEXT["num_attention_heads"], TEXT["qk_nope_head_dim"]
    for name, a in floats.items():
        short = name.removeprefix(ORIG)
        if short in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
            tensors[{"embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",
                     "norm.weight": "output_norm.weight"}[short]] = (a, short != "norm.weight")
            continue
        i, rest = re.match(r"layers\.(\d+)\.(.*)$", short).groups()
        if int(i) == N:
            continue
        if m := re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", rest):
            experts.setdefault(f"blk.{i}.ffn_{m.group(2)}_exps.weight", {})[int(m.group(1))] = a
        elif rest == "self_attn.kv_b_proj.weight":
            kv = a.reshape(h, nope + TEXT["v_head_dim"], -1)
            tensors[f"blk.{i}.attn_k_b.weight"] = (kv[:, :nope].transpose(0, 2, 1), True)
            tensors[f"blk.{i}.attn_v_b.weight"] = (kv[:, nope:], True)
        elif rest == "self_attn.A_log":
            tensors[f"blk.{i}.ssm_a"] = (-np.exp(a), False)
        else:
            g, q = names[rest]
            tensors[f"blk.{i}.{g}"] = (a, q)
    for name, parts in experts.items():
        tensors[name] = (np.stack([parts[e] for e in sorted(parts)]), True)

    def s(x: str) -> bytes:
        return struct.pack("<Q", len(x.encode())) + x.encode()

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    head += s("general.architecture") + struct.pack("<I", 8) + s("glm5next")
    head += s("general.alignment") + struct.pack("<II", 4, 32)
    blobs, off = [], 0
    for name, (a, q) in tensors.items():
        data = _q8_0(a).tobytes() if q else np.ascontiguousarray(a, np.float32).tobytes()
        head += s(name) + struct.pack(f"<I{a.ndim}Q", a.ndim, *reversed(a.shape))
        head += struct.pack("<IQ", 8 if q else 0, off)
        data += b"\0" * ((-len(data)) % 32)
        blobs.append(data)
        off += len(data)
    path.write_bytes(head + b"\0" * ((-len(head)) % 32) + b"".join(blobs))


def _save(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, off = {}, [], 0
    for name, (dtype, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _fp8(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """e4m3 codes and float32 weight_scale_inv [ceil(out/128), ceil(in/128)], as the original checkpoint stores them."""
    out, ins = a.shape
    bo, bi = -(-out // 128), -(-ins // 128)
    pad = np.zeros((bo * 128, bi * 128), np.float32)
    pad[:out, :ins] = np.abs(a)
    s = pad.reshape(bo, 128, bi, 128).max(axis=(1, 3)) / 448
    s[s == 0] = 1
    codes = np.array(mx.to_fp8(mx.array(a / np.repeat(np.repeat(s, 128, 0), 128, 1)[:out, :ins])))
    return codes, s.astype(np.float32)


def _bf16(a: np.ndarray) -> np.ndarray:
    return np.array(mx.array(a).astype(mx.bfloat16).view(mx.uint16))


def _original(folder: Path, floats: dict, edit=None) -> dict[str, tuple[str, np.ndarray]]:
    """Layer N as zai-org stores it: FP8 linears with 128x128 block scales, kv_b_proj, eh_proj, the indexer and norms
    bf16, the router fp32; beside two backbone tensors, over two shards found through the index."""
    t = {}
    for name, a in floats.items():
        short = name[len(f"{ORIG}layers.{N}."):]
        if not name.startswith(f"{ORIG}layers.{N}."):
            continue
        linear = short[: -len(".weight")]
        if ".experts." in short or linear in tool.MTP_Q and linear != "self_attn.kv_b_proj":
            codes, s = _fp8(a)
            t[name], t[name + "_scale_inv"] = ("F8_E4M3", codes), ("F32", s)
        elif short.startswith("mlp.gate."):
            t[name] = ("F32", a)
        else:
            t[name] = ("BF16", _bf16(a))
    t[f"{ORIG}embed_tokens.weight"] = ("BF16", _bf16(floats[f"{ORIG}embed_tokens.weight"]))
    t[f"{ORIG}layers.0.input_layernorm.weight"] = ("BF16", _bf16(floats[f"{ORIG}layers.0.input_layernorm.weight"]))
    if edit:
        edit(t)
    folder.mkdir(parents=True)
    names = sorted(t)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for shard, keys in shards.items():
        _save(folder / shard, {k: t[k] for k in keys})
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for shard, keys in shards.items() for k in keys}}))
    return t


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("glm5q8mtp")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(glm5_fakes, "TEXT", TEXT)
        m.setattr(glm5_fakes, "D", TEXT["hidden_size"])
        floats = _floats(glm5_fakes.write_checkpoint(root / "fake"))
    mx.set_default_device(previous)
    _write_gguf(root / "tiny-Q8_0.gguf", floats)
    stored = _original(root / "zai-org", floats)
    (root / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": TEXT}))
    (root / "tok").mkdir()
    (root / "tok" / "tokenizer_config.json").write_text("{}")
    return root, floats, stored


def _convert(root: Path, out: Path, *extra: str) -> int:
    return tool.main(["--gguf", str(root / "*.gguf"), "--config", str(root / "config.json"),
                      "--tokenizer-dir", str(root / "tok"), "--out", str(out), *extra])


@pytest.fixture(scope="module")
def plain(sources, tmp_path_factory):
    out = tmp_path_factory.mktemp("plain") / "out"
    assert _convert(sources[0], out, "--verify") == 0
    return out


@pytest.fixture(scope="module")
def grafted(sources, tmp_path_factory):
    out = tmp_path_factory.mktemp("grafted") / "out"
    assert _convert(sources[0], out, "--mtp-from", str(sources[0] / "zai-org"), "--verify") == 0
    return out


def _files(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def _index(folder: Path) -> dict[str, str]:
    return json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]


def test_without_the_option_the_output_is_as_before(plain, grafted):
    """Every file of the plain conversion is in the grafted one unchanged, but the index and SHA256SUMS, which only
    gain the head's entries."""
    a, b = _files(plain), _files(grafted)
    assert not any(f.startswith("model-mtp") for f in a)
    assert {f: h for f, h in b.items() if f in a and f not in ("model.safetensors.index.json", "SHA256SUMS")} == {
        f: h for f, h in a.items() if f not in ("model.safetensors.index.json", "SHA256SUMS")}
    ia, ib = _index(plain), _index(grafted)
    assert {k: v for k, v in ib.items() if not k.startswith(HEAD)} == ia
    assert set(ib.values()) - set(ia.values()) == {"model-mtp-00001.safetensors"}
    sums = (grafted / "SHA256SUMS").read_text()
    assert (plain / "SHA256SUMS").read_text().splitlines() == [x for x in sums.splitlines() if "model-mtp" not in x]
    assert f"{b['model-mtp-00001.safetensors']}  model-mtp-00001.safetensors" in sums


def test_the_head_is_the_original_within_the_8bit_bound(sources, grafted):
    _, _, stored = sources
    t = mx.load(str(grafted / "model-mtp-00001.safetensors"))
    pre = f"{ORIG}layers.{N}."
    checked = 0
    for name, (dtype, a) in stored.items():
        if not name.startswith(pre) or name.endswith("_scale_inv"):
            continue
        short = name[len(pre):]
        if dtype == "F8_E4M3" or short[: -len(".weight")] in tool.MTP_Q:
            if dtype == "F8_E4M3":
                s = stored[name + "_scale_inv"][1]
                ref = np.array(mx.from_fp8(mx.array(a), dtype=mx.float32)) * np.repeat(
                    np.repeat(s, 128, 0), 128, 1)[: a.shape[0], : a.shape[1]]
            else:
                ref = (a.astype(np.uint32) << 16).view(np.float32)
            base, e = short[: -len(".weight")], None
            if m := re.match(r"mlp\.experts\.(\d+)\.(\w+)$", base):
                base, e = f"mlp.switch_mlp.{m.group(2)}", int(m.group(1))
            q, sc, bi = (np.array(t[f"{HEAD}{base}.{p}"]) for p in ("weight", "scales", "biases"))
            if e is not None:
                q, sc, bi = q[e], sc[e], bi[e]
            assert t[f"{HEAD}{base}.scales"].dtype == mx.float32
            assert np.all(np.abs(_dequantize(q, sc, bi) - ref) <= _bound(sc, bi)), name
        else:
            got = t[HEAD + short]
            assert got.dtype == (mx.bfloat16 if dtype == "BF16" else mx.float32)
            assert np.array_equal(np.array(got.view(mx.uint16) if dtype == "BF16" else got), a), name
        checked += 1
    assert checked == len([k for k in stored if k.startswith(pre) and not k.endswith("_scale_inv")])
    assert t[f"{HEAD}mlp.switch_mlp.gate_proj.weight"].shape[0] == TEXT["n_routed_experts"]
    assert not any(".experts." in k or "scale_inv" in k for k in t)


def test_the_head_computes_the_source_model_s_head(sources, grafted):
    """Same rows and tokens through the grafted head and the fake checkpoint's own: a row's logits within the FP8
    round trip's ~0.05 (relative, the median row: a routing flip moves one row); experts out of order give ~0.7."""
    model = weights.load_backbone(grafted)
    heads = glm_mtp.load(model), glm_mtp.load(weights.load_backbone(sources[0] / "fake"))
    rng = np.random.default_rng(3)
    h = mx.array(rng.standard_normal((12, TEXT["hidden_size"])).astype(np.float32)).astype(mx.bfloat16)
    ids = mx.array(rng.integers(0, TEXT["vocab_size"], 12).astype(np.uint32))
    got, want = (np.array(m.logits(model, m(model, h, ids, [m.make_cache()], (12,), False)).astype(mx.float32))
                 for m in heads)
    assert np.median(np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)) < 0.15


def test_the_mac_engine_drafts_with_it(grafted, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    glm5_next.check(grafted)
    assert glm5_next.has_mtp(grafted)
    model = weights.load_backbone(grafted)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(21, seed=4)
    engine_a, a = _run_engine(runtime, prompt, 24)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 24)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


def test_mtp_only_grafts_an_earlier_conversion(sources, plain, grafted, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(plain, out)
    before = _files(out)
    assert tool.main(["--mtp-only", "--mtp-from", str(sources[0] / "zai-org"), "--out", str(out), "--verify"]) == 0
    after = _files(out)
    assert {f: after[f] for f in before if f not in ("model.safetensors.index.json", "SHA256SUMS")} == {
        f: h for f, h in before.items() if f not in ("model.safetensors.index.json", "SHA256SUMS")}
    assert after == _files(grafted)                                        # the same bytes as a graft at conversion
    assert not any(p.name.endswith(".part") for p in out.iterdir())

    with pytest.raises(SystemExit, match="already has an MTP layer"):
        tool.main(["--mtp-only", "--mtp-from", str(sources[0] / "zai-org"), "--out", str(out)])
    assert _files(out) == after
    assert tool.main(["--mtp-only", "--replace-mtp", "--mtp-from", str(sources[0] / "zai-org"), "--out",
                      str(out)]) == 0
    replaced = _files(out)
    assert "model-mtp-00001.safetensors" not in replaced
    assert replaced["model-mtp2-00001.safetensors"] == after["model-mtp-00001.safetensors"]
    assert {f: h for f, h in replaced.items() if f.startswith("model-0")} == {
        f: h for f, h in before.items() if f.startswith("model-0")}
    assert set(_index(out).values()) == {f for f in replaced if f.endswith(".safetensors")}


def test_mtp_only_refuses_another_format(sources, plain, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(plain, out)
    config = json.loads((out / "config.json").read_text())
    config["quantization"] = {"bits": 4, "group_size": 64}
    (out / "config.json").write_text(json.dumps(config))
    before = _files(out)
    with pytest.raises(SystemExit, match="8-bit in groups of 32"):
        tool.main(["--mtp-only", "--mtp-from", str(sources[0] / "zai-org"), "--out", str(out)])
    assert _files(out) == before


def _drop(name):
    return lambda t: t.pop(name)


def _nan(t):
    codes = t[f"{ORIG}layers.{N}.mlp.experts.3.up_proj.weight"][1].copy()
    codes[1, 2] = 0x7F
    t[f"{ORIG}layers.{N}.mlp.experts.3.up_proj.weight"] = ("F8_E4M3", codes)


def _scale_shape(t):
    t[f"{ORIG}layers.{N}.self_attn.o_proj.weight_scale_inv"] = ("F32", np.ones((1, 1), np.float32))  # [256, 128]


def _expert_shape(t):
    codes = t[f"{ORIG}layers.{N}.mlp.experts.5.gate_proj.weight"][1]
    t[f"{ORIG}layers.{N}.mlp.experts.5.gate_proj.weight"] = ("F8_E4M3", codes[:32])


def _extra(t):
    t[f"{ORIG}layers.{N}.shared_head.head.weight"] = ("BF16", np.zeros((4, 4), np.uint16))


@pytest.mark.parametrize("edit, match", [
    (_drop(f"{ORIG}layers.{N}.hnorm.weight"), "1 missing, hnorm.weight"),
    (_drop(f"{ORIG}layers.{N}.mlp.experts.7.down_proj.weight"), "1 missing, mlp.experts.7.down_proj.weight"),
    (_drop(f"{ORIG}layers.{N}.self_attn.q_b_proj.weight_scale_inv"), "q_b_proj.weight_scale_inv: not in"),
    (_extra, "1 not mapped, shared_head.head.weight"),
    (_nan, "experts.3.up_proj: NaN codes"),
    (_scale_shape, "o_proj: block scales"),
    (_expert_shape, "experts.5.gate_proj: shape differs"),
])
def test_refusals(sources, plain, tmp_path, edit, match):
    root, floats, _ = sources
    _original(tmp_path / "bad", floats, edit)
    out = tmp_path / "out"
    shutil.copytree(plain, out)
    before = _files(out)
    with pytest.raises(SystemExit, match=match):
        tool.main(["--mtp-only", "--mtp-from", str(tmp_path / "bad"), "--out", str(out)])
    assert _files(out) == before                                           # nothing left behind
    with pytest.raises(SystemExit, match=match):
        _convert(root, tmp_path / "full", "--mtp-from", str(tmp_path / "bad"))
    assert not (tmp_path / "full").exists()                               # refused before the GGUF is read


def _quiet_failure(at: int):
    """affine8 failing on its ``at``-th call, as a disk or memory error would."""
    calls = []
    real = tool.affine8

    def fail(w):
        calls.append(1)
        if len(calls) == at:
            raise OSError(28, "No space left on device")
        return real(w)

    return fail


def test_a_failed_head_leaves_a_finished_conversion(sources, plain, grafted, tmp_path, monkeypatch):
    """Past the check, a full conversion whose head fails is the plain one, and --mtp-only finishes it."""
    root = sources[0]
    linears = len(tool.MTP_Q) + 3 * TEXT["n_routed_experts"]
    monkeypatch.setattr(tool, "affine8", _quiet_failure(linears + 5))     # the check passes, the write fails
    with pytest.raises(OSError):
        _convert(root, tmp_path / "out", "--mtp-from", str(root / "zai-org"))
    monkeypatch.undo()
    assert _files(tmp_path / "out") == _files(plain)
    assert tool.main(["--mtp-only", "--mtp-from", str(root / "zai-org"), "--out", str(tmp_path / "out")]) == 0
    assert _files(tmp_path / "out") == _files(grafted)


def test_mtp_only_rolls_back_a_failure_between_the_renames(sources, plain, tmp_path, monkeypatch):
    out = tmp_path / "out"
    shutil.copytree(plain, out)
    before = _files(out)
    real = os.replace

    def replace(src, dst):
        if Path(dst).name == "SHA256SUMS" and Path(src).name == "SHA256SUMS.part":
            raise OSError(28, "No space left on device")
        return real(src, dst)

    monkeypatch.setattr(tool.os, "replace", replace)
    with pytest.raises(OSError):
        tool.main(["--mtp-only", "--mtp-from", str(sources[0] / "zai-org"), "--out", str(out)])
    assert _files(out) == before                                           # old index back, head files gone


@pytest.mark.parametrize("part", ["model-mtp-00001.safetensors.part", "SHA256SUMS.part"])
def test_mtp_only_refuses_a_link_at_a_part_path(sources, plain, tmp_path, part):
    out = tmp_path / "out"
    shutil.copytree(plain, out)
    target = tmp_path / "elsewhere"
    target.write_bytes(b"keep")
    (out / part).symlink_to(target)
    before = _files(out)
    with pytest.raises(SystemExit, match="symbolic link"):
        tool.main(["--mtp-only", "--mtp-from", str(sources[0] / "zai-org"), "--out", str(out)])
    assert target.read_bytes() == b"keep" and _files(out) == before


def test_arguments(sources, tmp_path):
    root = sources[0]
    for argv in (["--mtp-only", "--out", str(tmp_path)],                        # no --mtp-from
                 ["--mtp-only", "--mtp-from", str(root / "zai-org"), "--out", str(tmp_path), "--gguf", "x"],
                 ["--mtp-from", str(root / "zai-org"), "--out", str(tmp_path / "o")],   # no GGUF
                 ["--mtp-only", "--mtp-from", str(root / "zai-org"), "--out", str(tmp_path), "--extra"]):
        with pytest.raises(SystemExit) as e:
            tool.main(argv)
        assert e.value.code == 2
    with pytest.raises(SystemExit, match="not a finished conversion"):
        tool.main(["--mtp-only", "--mtp-from", str(root / "zai-org"), "--out", str(tmp_path / "missing")])

"""The NVFP4 loader on a tiny modelopt checkpoint: the route's tensors, shapes and exactness.

Runs wherever Triton runs (the CUDA tests' directory; the loader's slicing is torch code, the FP4/BF16
*kernels* are checked in test_flashnext_nvfp4_kernels.py on a GPU). It loads a checkpoint that stores
what the Swift checkpoint stores — FP4 arrays for the routed experts, BF16 for everything else — and
checks the loader reads it into the engine's faces."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import Config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("nvfp4-tiny"))


def test_config_reads_the_modelopt_checkpoint(tiny: Path) -> None:
    cfg = Config.read(tiny)
    assert cfg.quant == "modelopt"
    assert cfg.nvfp4_group == 16
    assert cfg.ple_layers == [1]


def test_the_reader_maps_the_fp8_dtype(tiny: Path) -> None:
    from tensorfold.families.qwen4_exp.cuda.weights import _DT

    assert _DT["F8_E4M3"] is torch.float8_e4m3fn


def test_the_header_names_every_tensor(tiny: Path) -> None:
    shard = tiny / "model-00001-of-00001.safetensors"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    w = hdr["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert w["dtype"] == "U8" and w["shape"] == [128, 128]
    assert hdr["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]["dtype"] == "F8_E4M3"
    assert hdr["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"
    assert hdr["mtp.layers.0.mlp.experts.gate_up_proj"]["dtype"] == "BF16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_the_loader_builds_the_nvfp4_faces(tiny: Path) -> None:
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(tiny, mtp=True, draft_vocab=None)
    assert w.cfg.quant == "modelopt"
    l0 = w.layers[0]
    moe = l0.moe
    assert getattr(moe.experts, "kernel", "") == "nvfp4"
    ex = moe.experts
    assert ex.routed == 2 and ex.width == 128 and ex.dims == 256
    # the FP4 blocks dequantize to the checkpoint's exact weights (the stored values, not a requantization)
    from tensorfold.cuda.nvfp4.experts import dense
    from safetensors import safe_open

    gate = dense(ex.routed_experts, 0, "gate")
    with safe_open(str(tiny / "model-00001-of-00001.safetensors"), framework="pt") as f:
        p = next(n for n in f.keys() if n.endswith("layers.0.mlp.experts.0.gate_proj.weight"))[:-len("weight")]
        from tensorfold.cuda.nvfp4 import format as fmt

        want = torch.from_numpy(fmt.dequant("nvfp4", f.get_tensor(p + "weight").numpy(),
                                            f.get_tensor(p + "weight_scale").view(torch.uint8).numpy(),
                                            float(f.get_tensor(p + "weight_scale_2"))))
    assert gate.shape == (128, 256) and torch.equal(gate.cpu(), want)
    # the non-experts ride the b16 face: DeltaNet (layer 0) and attention (layer 1)
    assert l0.gdn.kernel == "b16" and w.layers[1].attn.kernel == "b16"
    # the MTP layer's bf16 experts ride NVFP4 blocks (they only draft), within NVFP4's rounding of the stored rows
    mtp_ex = w.mtp.layer.moe.experts
    assert getattr(mtp_ex, "kernel", "") == "nvfp4"
    fd = dense(mtp_ex.routed_experts, 0, "down")
    shard = tiny / "model-00001-of-00001.safetensors"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        e = hdr["mtp.layers.0.mlp.experts.down_proj"]
        lo, hi = e["data_offsets"]
        f.seek(8 + n + lo)
        dn = torch.frombuffer(bytearray(f.read(hi - lo)), dtype=torch.bfloat16).reshape(2, 256, 128)
    ref = dn[0].to(device=fd.device, dtype=torch.float32)
    assert float((fd - ref).norm() / ref.norm()) < 0.12


def test_the_reader_finds_the_published_naming(tmp_path: Path) -> None:
    """The published NVFP4 checkpoint spells its language-model tensors ``model.language_model.*`` while its
    lm_head and mtp stay top level; the loader reads the same faces as from the plain ``model.*`` layout."""

    plain = write(tmp_path / "plain")
    named = write(tmp_path / "named", prefix="model.language_model.")
    names = _names(named)
    assert "model.language_model.embed_tokens.weight" in names, "the fixture lost its prefix"
    assert "model.language_model.layers.0.mlp.experts.0.gate_proj.weight" in names
    assert "lm_head.weight" in names and "mtp.fc_embedding.weight" in names

    from tensorfold.families.qwen4_exp.cuda.weights import load

    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    a, b = load(plain, mtp=True, draft_vocab=None), load(named, mtp=True, draft_vocab=None)
    assert b.cfg.quant == "modelopt" and len(b.layers) == len(a.layers)
    for i, (x, y) in enumerate(zip(a.layers, b.layers, strict=True)):
        for face in ("up", "down", "up_scale", "down_scale"):
            got, want = getattr(y.moe.experts.routed_experts, face), getattr(x.moe.experts.routed_experts, face)
            assert torch.equal(got, want), (i, face)
    assert torch.equal(b.embed[0].float(), a.embed[0].float())
    assert a.mtp is not None and b.mtp is not None
    assert torch.equal(b.mtp.layer.moe.experts.routed_experts.down, a.mtp.layer.moe.experts.routed_experts.down)


def _names(dir: Path) -> set[str]:
    """Every tensor name in a tiny checkpoint's index."""

    index = json.loads((dir / "model.safetensors.index.json").read_text())
    return set(index["weight_map"])


def _tensor(dir: Path, name: str) -> torch.Tensor:
    """One tensor of a tiny checkpoint, read out of the file the index names."""

    shard = dir / json.loads((dir / "model.safetensors.index.json").read_text())["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(n))[name]
        lo, hi = entry["data_offsets"]
        f.seek(8 + n + lo)
        raw = bytearray(f.read(hi - lo))
    dtype = {"BF16": torch.bfloat16, "I32": torch.int32}[entry["dtype"]]
    return torch.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def test_the_loader_reads_the_published_bf16_table(tmp_path: Path) -> None:
    """The published revision stores the n-gram table as bf16 rows with no per-shard scales and biases: the
    loader takes that layout as it ships, and a gather hands back the checkpoint's own bytes."""

    from tensorfold.families.qwen4_exp.host_table import BF16Table

    tiny = write(tmp_path / "bf16", ple_bf16=True)
    ngram = Config.read(tiny).ngram(0)
    assert "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales" not in _names(tiny)

    from tensorfold.families.qwen4_exp.cuda.weights import load

    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    w = load(tiny, mtp=True, draft_vocab=None)
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    assert isinstance(ple.table, BF16Table), "a bf16 table must not go through the 4-bit HostTable"
    assert (ple.table.rows, ple.table.width) == (ngram.rows, ngram.dims)
    ids = np.array([0, 17, ngram.rows - 1, 500])
    got = ple.table.gather(ids)
    stored = _tensor(tiny, "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    want = stored[torch.from_numpy(ids)].view(torch.int16).numpy()
    assert got.dtype == np.uint16 and np.array_equal(got.view(np.int16), want)


def test_the_bf16_rows_reach_the_engine_buffers(tmp_path: Path) -> None:
    """``stage_ple_rows`` copies a bf16 table's rows to the device buffer the PLE kernel reads, bit for bit."""

    from tensorfold.families.qwen4_exp.cuda.forward import stage_ple_rows
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    tiny = write(tmp_path / "bf16-stage", ple_bf16=True)
    ngram = Config.read(tiny).ngram(0)
    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(tiny, mtp=True, draft_vocab=None)
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    b = Buffers(w, rows=3, capacity=64)
    ids = (np.arange(3 * ngram.heads) % ngram.rows).reshape(3, ngram.heads)
    stage_ple_rows(ple, b, ids, at=0)
    stored = _tensor(tiny, "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    want = stored.index_select(0, torch.from_numpy(ids.reshape(-1)))
    assert torch.equal(b.ple_v[:ids.size].cpu(), want.cpu())



def test_the_loader_reads_mxfp8_linears_and_an_nvfp4_table(tmp_path: Path) -> None:
    """local-inference-lab's layout: DeltaNet, attention and shared-expert linears in MXFP8 go to the lane matmul as
    stored, the n-gram table's NVFP4 rows come back as bf16(code x scale x table scale), and the engine decodes."""

    from tensorfold.cuda.nvfp4.linear import Mx8Linear
    from tensorfold.families.qwen4_exp.host_table import NVFP4Table

    tiny = write(tmp_path / "mx", mxfp8=True, ple_nvfp4=True, hidden=512)      # PLE kernels: 512-wide streams
    config = json.loads((tiny / "config.json").read_text())
    config["quantization"]["quantized_layers"] = {
        "model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16}}
    (tiny / "config.json").write_text(json.dumps(config))
    from tensorfold.families.qwen4_exp import check

    check(tiny)
    if not torch.cuda.is_available():                                   # the loader builds CUDA tensors
        pytest.skip("the loader builds CUDA tensors")
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(tiny, mtp=True, draft_vocab=None)
    gdn = next(layer.gdn for layer in w.layers if layer.gdn is not None)
    attn = next(layer.attn for layer in w.layers if layer.attn is not None)
    assert all(isinstance(f, Mx8Linear) for f in (gdn.proj, gdn.out, attn.proj, attn.o))
    assert isinstance(w.layers[0].moe.experts.shared.gu, Mx8Linear)
    ple = next(layer.ple for layer in w.layers if layer.ple is not None)
    assert isinstance(ple.table, NVFP4Table)
    name = "model.layers.1.ple.ple_embedding.ngram_embedding."
    from safetensors import safe_open

    with safe_open(str(tiny / "model-00001-of-00001.safetensors"), framework="pt") as f:
        codes, scales = f.get_tensor(name + "shard_0.weight"), f.get_tensor(name + "shard_0.weight_scale")
        g = float(f.get_tensor(name + "weight_scale_2").reshape(-1)[0])
    ids = np.array([0, 3, ple.table.rows - 1])
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6])
    c = codes[torch.from_numpy(ids)].long()
    nib = torch.stack([c & 0xF, c >> 4], -1).reshape(len(ids), -1)
    want = (e2m1[nib] * scales[torch.from_numpy(ids)].float().repeat_interleave(16, 1) * g).to(torch.bfloat16)
    assert np.array_equal(ple.table.gather(ids).view(np.int16), want.view(torch.int16).numpy())
    e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
    first = prefill(e, [5, 17, 99, 250, 7, 64, 30, 11, 12, 13], None)
    assert len(serial_decode(e, first, 8, None).tokens) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_the_loader_reads_block_fp8_linears(tmp_path: Path) -> None:
    """``FP8_PB_WO`` linears, the head's too, reach the lane matmul as stored beside their bf16 neighbours."""

    from safetensors import safe_open

    from tensorfold.cuda.nvfp4 import format as fmt
    from tensorfold.cuda.nvfp4.linear import Concat, Fp8BlockLinear
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    tiny = write(tmp_path / "fp8b", fp8block=True, hidden=512)                 # PLE kernels: 512-wide streams
    w = load(tiny, mtp=True, draft_vocab=None)
    gdn = next(layer.gdn for layer in w.layers if layer.gdn is not None)
    attn = next(layer.attn for layer in w.layers if layer.attn is not None)
    for stack in (gdn.proj, attn.proj):
        assert isinstance(stack, Concat) and isinstance(stack.parts[0], Fp8BlockLinear)
        assert getattr(stack.parts[1], "kernel", "") == "b16"
    assert isinstance(gdn.out, Fp8BlockLinear) and isinstance(attn.o, Fp8BlockLinear)
    with safe_open(str(tiny / "model-00001-of-00001.safetensors"), framework="pt") as f:
        codes, scale = f.get_tensor("lm_head.weight"), f.get_tensor("lm_head.weight_scale_inv")
    stored = Fp8BlockLinear.from_checkpoint(codes.cuda(), scale.cuda())
    assert isinstance(w.head, Fp8BlockLinear) and w.head.lane                   # prompt heads on the lane matmul too
    assert torch.equal(w.head.w8, stored.w8) and torch.equal(w.head.bs, stored.bs)
    x = torch.randn((3, codes.shape[1]), device="cuda").to(torch.bfloat16)
    stored_f64 = torch.from_numpy(fmt.dequant("fp8block", codes.view(torch.uint8).numpy(), scale.numpy())).double()
    want = x.double() @ stored_f64.cuda().t()
    assert torch.allclose(w.head.prefill(x).double(), want, rtol=1e-2, atol=1e-3)
    e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
    first = prefill(e, [5, 17, 99, 250, 7, 64, 30, 11, 12, 13], None)
    assert len(serial_decode(e, first, 8, None).tokens) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-prompts", "fp8-prompts"])
@pytest.mark.parametrize("layout", [{}, {"mxfp8": True, "ple_nvfp4": True}, {"fp8block": True}],
                         ids=["bf16", "mxfp8", "fp8block"])
@pytest.mark.parametrize("seed", [None, 7])
def test_drafts_over_a_draft_vocabulary_keep_the_serial_tokens(tmp_path: Path, layout: dict, seed, fp8) -> None:
    """The draft head holds the draft vocabulary's rows (not the whole head), so drafts map back to their ids; bf16
    prompts and --prefill-fp8 alike."""

    from tensorfold.cuda import prompt_precision
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(write(tmp_path / "tiny", hidden=512, **layout), mtp=True, draft_vocab=128)   # 512-wide PLE streams
    assert w.draft_head.n == len(w.draft_ids) == 128
    assert w.fast_prefill == bool(layout)                               # MXFP8 linears have an FP8 prompt kernel
    sampling = None if seed is None else Sampling(seed=seed, top_k=20, top_p=0.95)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    with prompt_precision.using(fp8):
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
        first = prefill(e, prompt, sampling)
        ref = serial_decode(e, first, 16, sampling).tokens
        for depth, confidence in ((2, 0.0), (4, 0.3)):                 # 0.3: drafts read their probability
            assert prefill(e, prompt, sampling) == first
            assert mtp_decode(e, first, 16, sampling, depth=depth, confidence=confidence).tokens == ref, depth


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_centred_norms_under_the_published_naming_get_their_one_back(tmp_path: Path) -> None:
    """An export that stores RMSNorm weights centred (around 0) under ``model.language_model.*`` loads the same
    scales as one that stores them around 1, so the two decode the same tokens."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    named = "model.language_model."
    a = load(write(tmp_path / "one", prefix=named, hidden=512), mtp=True, draft_vocab=None)   # 512-wide PLE streams
    b = load(write(tmp_path / "zero", prefix=named, hidden=512, centred=True), mtp=True, draft_vocab=None)
    assert a.around_one and not b.around_one
    for x, y in zip(a.layers, b.layers, strict=True):
        assert torch.equal(x.attn_hc.scale, y.attn_hc.scale) and torch.equal(x.mlp_hc.scale, y.mlp_hc.scale)
    tokens = []
    for w in (a, b):
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
        first = prefill(e, [5, 17, 99, 250, 7, 64, 30, 11], None)
        tokens.append(serial_decode(e, first, 8, None).tokens)
    assert tokens[0] == tokens[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_graph_replay_follows_each_steps_experts(tmp_path: Path, monkeypatch) -> None:
    """The FP4 MoE writes its routing plan on the device, so a decode graph captured for one step's experts replays
    another step's: with 8 routed experts (top 2) the picks change from step to step and prompt to prompt, and the
    graphed engine's tokens equal the eager engine's, serial and drafted, greedy and seeded."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(write(tmp_path / "moe8", hidden=512, experts=8), mtp=True, draft_vocab=None)
    assert w.cfg.experts == 8 and w.cfg.top_k == 2
    picks: list[tuple] = []
    real = nvfp4_moe.moe

    def recording(x, xs, router_rows, ex, buf, cfg):
        out = real(x, xs, router_rows, ex, buf, cfg)
        top = buf.logits[:x.shape[0], :ex.routed].float().topk(2, dim=-1).indices.sort(dim=-1).values
        picks.extend(tuple(r) for r in top.tolist())
        return out

    eager = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
    graphed = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=True)
    prompts = ([5, 17, 99, 250, 7, 64, 30, 11], [200, 3, 3, 3, 90, 41], [1, 2, 4, 8, 16, 32, 64, 128, 255, 9])
    for sampling in (None, Sampling(seed=11, top_k=20, top_p=0.95)):
        for prompt in prompts:
            monkeypatch.setattr(nvfp4_moe, "moe", recording)
            first = prefill(eager, prompt, sampling)
            ref = serial_decode(eager, first, 16, sampling).tokens
            monkeypatch.setattr(nvfp4_moe, "moe", real)                  # graphs capture the plain step
            assert prefill(graphed, prompt, sampling) == first
            assert serial_decode(graphed, first, 16, sampling).tokens == ref, (prompt, sampling)
            for e in (eager, graphed):
                assert prefill(e, prompt, sampling) == first
                assert mtp_decode(e, first, 16, sampling, depth=3, confidence=0.0).tokens == ref, (prompt, e.graphs)
    assert len(set(picks)) >= 8, "the routing never changed, so replay was not tested"
    assert 0 < graphed.graphs.captures < 16 * len(prompts), "the graphed engine replayed no captured step"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
def test_an_nvfp4_checkpoint_refuses_ple_on_ssd(tmp_path: Path) -> None:
    """The SSD reader takes the MLX layout's shards only, so --ple-on-ssd on an NVFP4 checkpoint stops by name."""

    from tensorfold.families.qwen4_exp.cuda.weights import load

    with pytest.raises(ValueError, match="ple-on-ssd"):
        load(write(tmp_path / "ssd", ple_nvfp4=True), mtp=True, draft_vocab=None, ple_on_ssd=True)


def test_an_nvfp4_checkpoint_refuses_two_ranks(tiny: Path, monkeypatch) -> None:
    """Two ranks read the MLX checkpoint: an NVFP4 one on --tp 2 stops by name before any rank starts."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setattr("tensorfold.cuda.comm.NCCL", lambda *a, **k: pytest.fail("the ranks started"))
    with pytest.raises(ValueError, match="one GPU"):
        FlashNextEngine(tiny, tp=2, rank=0, master="127.0.0.1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
@pytest.mark.parametrize("scale", ["tensor", "row", "block"])
@pytest.mark.parametrize("sampled", [False, True])
def test_fp8_drafter_experts_draft_as_their_dequantized_bf16(tmp_path: Path, scale: str, sampled: bool,
                                                        monkeypatch) -> None:
    """FP8 MTP experts match their dequantized bf16 reference and preserve serial tokens."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.weights import load

    from tensorfold.cuda.nvfp4.experts import dense

    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
    sampling = Sampling(seed=7, top_k=20, top_p=0.95) if sampled else Sampling(seed=7, temperature=0, top_k=1)
    runs, packed, inputs = {}, {}, {}
    pack = nvfp4_moe.moe4_from_bf16

    def capture(gate_up, down, shared):
        assert kind not in inputs
        inputs[kind] = tuple(t.cpu().contiguous().view(torch.uint8) for t in (gate_up, down))
        return pack(gate_up, down, shared)

    monkeypatch.setattr(nvfp4_moe, "moe4_from_bf16", capture)
    for kind in ("fp8", "fp8_dequant"):
        w = load(write(tmp_path / kind, hidden=512, mtp_experts=kind, mtp_scale=scale), mtp=True, draft_vocab=128)
        experts = w.mtp.layer.moe.experts.routed_experts
        packed[kind] = [dense(experts, i, proj).cpu() for i in range(w.cfg.experts) for proj in ("gate", "up", "down")]
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
        first = prefill(e, prompt, sampling)
        ref = serial_decode(e, first, 16, sampling).tokens
        assert prefill(e, prompt, sampling) == first                  # the prompt's state again
        out = mtp_decode(e, first, 16, sampling, depth=4, confidence=0.0)
        assert out.tokens == ref, kind
        runs[kind] = out
    assert all(torch.equal(a, b) for a, b in zip(inputs["fp8"], inputs["fp8_dequant"]))
    assert all(torch.equal(a, b) for a, b in zip(packed["fp8"], packed["fp8_dequant"]))
    same = ("tokens", "rounds", "drafted", "accepted", "keeps", "widths")    # the same drafts, accepted the same
    assert [getattr(runs["fp8"], k) for k in same] == [getattr(runs["fp8_dequant"], k) for k in same]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
@pytest.mark.parametrize("case", ["main_gate", "main_up", "scale_shape", "scale_bytes", "mixed_dtype", "missing_scale"])
def test_fp8_expert_refusals(tmp_path: Path, monkeypatch, case: str) -> None:
    """Refuse FP8 target experts and malformed MTP scales before packing or decoding them."""

    from tensorfold.families.qwen4_exp.cuda.reader import _Reader
    from tensorfold.families.qwen4_exp.cuda.weights import load

    path = write(tmp_path / case, hidden=512, mtp_experts="fp8")
    get, has = _Reader.get, _Reader.has
    mtp = "mtp.layers.0.mlp.experts."
    main = "model.layers.0.mlp.experts."
    targets = {"main_gate": main + "0.gate_proj.weight", "main_up": main + "1.up_proj.weight",
               "scale_shape": mtp + "0.gate_proj.weight_scale", "scale_bytes": mtp + "0.gate_proj.weight_scale",
               "mixed_dtype": mtp + "1.up_proj.weight", "missing_scale": mtp + "0.gate_proj.weight_scale"}

    def altered(self, name):
        value = get(self, name)
        if name != targets[case]:
            return value
        if case.startswith("main_"):
            return value.to(torch.float8_e4m3fn)
        if case == "scale_shape":
            return torch.ones(512, device=value.device)
        if case == "scale_bytes":
            return value.to(torch.uint8)
        return value.to(torch.float16)

    monkeypatch.setattr(_Reader, "get", altered)
    if case == "missing_scale":
        monkeypatch.setattr(_Reader, "has", lambda self, name: False if name == targets[case] else has(self, name))
    match = {"main_gate": "FP8 routed experts", "main_up": "FP8 is read only in MTP",
             "scale_shape": "one float per tensor or output row", "scale_bytes": "one float per tensor or output row",
             "mixed_dtype": "expected FP8 e4m3 MTP expert weights", "missing_scale": "need a tensor, row"}
    with pytest.raises(ValueError, match=match[case]):
        load(path, mtp=True, draft_vocab=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader builds CUDA tensors")
@pytest.mark.parametrize("scale", ["tensor", "row", "block"])
def test_fp8_mtp_real_projection_shapes_dequantize_byte_exactly(tmp_path: Path, monkeypatch, scale: str) -> None:
    """Capture bf16 inputs before requantization for real 640x2560 gate/up and 2560x640 down projections."""

    from tensorfold.families.qwen4_exp.cuda.weights import load

    actual = write(tmp_path / "fp8", hidden=2560, moe_width=640, mtp_experts="fp8", mtp_scale=scale)
    reference = write(tmp_path / "bf16", hidden=2560, moe_width=640, mtp_experts="fp8_dequant", mtp_scale=scale)
    captured = []
    pack = nvfp4_moe.moe4_from_bf16

    def capture(gate_up, down, shared):
        assert gate_up.dtype == down.dtype == torch.bfloat16
        captured.append(tuple(t.cpu().contiguous().view(torch.uint8) for t in (gate_up, down)))
        return pack(gate_up, down, shared)

    monkeypatch.setattr(nvfp4_moe, "moe4_from_bf16", capture)
    w = load(actual, mtp=True, draft_vocab=128)
    assert w.mtp is not None and len(captured) == 1
    for got, name, shape in zip(captured[0], ("gate_up_proj", "down_proj"), ((2, 1280, 2560), (2, 2560, 640))):
        want = _tensor(reference, "mtp.layers.0.mlp.experts." + name)
        assert tuple(want.shape) == shape
        assert torch.equal(got, want.contiguous().view(torch.uint8)), (scale, name)

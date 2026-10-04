"""The per-rank loader: every object of the tiny checkpoint holds its source tensor's rank slice exactly (world 1 and
both ranks of world 2), Engram tables are never read; real layers stay within split's estimate and staging."""

from __future__ import annotations

import gc
import os
import re
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytest.importorskip("safetensors")

import dsv41_tiny
from dsv41_ref_pack import RefPack

from tensorfold.cuda import capacity, direct_read
from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.cuda.nvfp4 import experts as nvfp4
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import loader, rope, split
from tensorfold.families.deepseek_v41.cuda.geometry import rope_bytes

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
CAP = 64                                # RoPE table slots in these tests
REAL_LAYERS = (0, 1, 2, 20, 39)


@pytest.fixture(scope="module")
def linked(tiny_dir, tmp_path_factory) -> Path:
    """The tiny checkpoint with every shard a symlink into it, as the pack's served snapshot links its shards."""

    out = tmp_path_factory.mktemp("dsv41_linked")
    for f in tiny_dir.iterdir():
        if f.suffix == ".safetensors":
            (out / f.name).symlink_to(f)
        else:
            shutil.copy(f, out / f.name)
    return out


@pytest.fixture
def reads(monkeypatch) -> list:
    """Every (file, first byte, bytes) the loader's reader is asked for."""

    seen, read = [], direct_read.Reader.read

    def spy(self, path, offset, n, *args, **kwargs):
        seen.append((Path(path).resolve(), offset, n))
        return read(self, path, offset, n, *args, **kwargs)

    monkeypatch.setattr(direct_read.Reader, "read", spy)
    return seen


class Source:
    """The checkpoint's tensors on the CPU, each cut to a rank's part by ``split``; every name compared is counted."""

    def __init__(self, model_dir: Path, rank: int, world: int) -> None:
        self.pack, self.rank, self.world, self.seen = RefPack(model_dir), rank, world, set()

    def cut(self, name: str) -> tuple[slice, ...]:
        return split.slice_for(self.rank, self.world, name, self.pack.shape(name))

    def raw(self, name: str) -> torch.Tensor:
        self.seen.add(name)
        return self.pack.tensor(name)[self.cut(name)]

    def dense(self, *modules: str, packed: bool = False) -> torch.Tensor:
        """The modules' fp32 values as fp64 [out, in], each cut as its weight is (``packed``: two values a byte)."""

        out = []
        for m in modules:
            key, scale = m + ".weight", m + (".weight_scale" if m == "lm_head" else ".scale")
            self.seen |= {key, scale}
            rows, cols = self.cut(key)
            if packed:
                cols = slice(2 * cols.start, 2 * cols.stop)
            out.append(self.pack.dense_fp32(m)[rows, cols].double())
        return torch.cat(out)


def _eq(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype and torch.equal(got.cpu(), want.cpu()), what


def _mx8(lin) -> torch.Tensor:
    """An ``Mx8Linear``'s stored values: its e4m3 rows times each row's 1x32 powers of two, fp64 on the CPU."""

    s = lin.scale_rows().cpu().to(torch.int64).repeat_interleave(32, dim=1)
    return torch.ldexp(lin.w8_rows().cpu().to(torch.float64), s - 127)


def _check_attn(a, p: str, layer: int | None, src: Source, cfg: Config) -> None:
    x = p + "attn."
    _eq(_mx8(a.wqa_kv), src.dense(x + "wq_a", x + "wkv"), x + "wq_a|wkv")
    _eq(_mx8(a.wq_b), src.dense(x + "wq_b"), x + "wq_b")
    assert len(a.wo_a) == cfg.o_groups // src.world and all(g.n == cfg.o_lora_rank for g in a.wo_a)
    _eq(torch.cat([_mx8(g) for g in a.wo_a]), src.dense(x + "wo_a"), x + "wo_a")
    _eq(_mx8(a.wo_b), src.dense(x + "wo_b"), x + "wo_b")
    for name, got in (("q_norm.weight", a.q_norm), ("kv_norm.weight", a.kv_norm), ("attn_sink", a.sink)):
        _eq(got, src.raw(x + name), x + name)
    if layer in cfg.kv_source_layer_ids:
        c = x + "compressor."
        kv = [src.raw(c + "wkv.weight")] + ([src.raw(c + "wgate.weight")] if cfg.compress_ratios[layer] == 2 else [])
        assert a.comp.ratio == cfg.compress_ratios[layer]
        _eq(a.comp.wkv.weight, torch.cat(kv), c + "wkv|wgate")
        _eq(a.comp.norm, src.raw(c + "norm.weight"), c + "norm")
        _eq(a.idx.wk.weight, src.raw(x + "indexer.wk.weight"), x + "indexer.wk")
        _eq(a.idx.k_norm, src.raw(x + "indexer.k_norm.weight"), x + "indexer.k_norm")
    else:
        assert a.comp is None and (a.idx is None or a.idx.wk is None)
    if layer in cfg.index_source_layer_ids:
        _eq(_mx8(a.idx.wq_b), src.dense(x + "indexer.wq_b"), x + "indexer.wq_b")
        _eq(a.idx.wproj.weight, src.raw(x + "indexer.weights_proj.weight"), x + "indexer.weights_proj")
    elif a.idx is not None:
        assert a.idx.wq_b is None and a.idx.wproj is None


def _check_exl3(ex, f: str, src: Source, cfg: Config) -> None:
    """Each trellis in its 256-B slot of one buffer, dequantized equal to the source's rows or columns."""

    E, storage = cfg.n_routed_experts, ex.keep[0].untyped_storage().data_ptr()
    assert ex.count == E
    for j, w in enumerate(("w1", "w3", "w2")):
        ptrs = (ex.gate_ptr, ex.up_ptr, ex.down_ptr)[j].tolist()
        suh, svh = (ex.suh_g, ex.suh_u, ex.suh_d)[j], (ex.svh_g, ex.svh_u, ex.svh_d)[j]
        for e in range(E):
            name, t = f"{f}experts.{e}.{w}", ex.keep[j * E + e]
            assert t.data_ptr() == ptrs[e] and t.data_ptr() % loader.SLOT == 0
            assert t.untyped_storage().data_ptr() == storage, "one buffer a layer"
            rows, cols, _ = src.cut(name + ".trellis")
            full = exl3.dequant(src.pack.tensor(name + ".trellis").cuda(), cfg.codebook)
            src.seen |= {name + ".trellis", name + ".mcg"}
            want = full[16 * rows.start:16 * rows.stop, 16 * cols.start:16 * cols.stop]
            _eq(exl3.dequant(t, cfg.codebook), want, name)
            _eq(suh[e], src.raw(name + ".suh"), name + ".suh")
            _eq(svh[e], src.raw(name + ".svh"), name + ".svh")


def _check_experts4(ex, f: str, src: Source, cfg: Config) -> None:
    assert ex.count == cfg.dspark_n_routed_experts and ex.limit == cfg.swiglu_limit
    for e in range(ex.count):
        for w, which in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            name = f"{f}experts.{e}.{w}"
            _eq(nvfp4.dense(ex, e, which).double(), src.dense(name, packed=True), name)


def _check_block(lw, index: int, p: str, layer: int | None, src: Source, cfg: Config) -> None:
    assert lw.index == index and lw.role == cfg.roles[index]
    for site, hc in (("attn", lw.hc_attn), ("ffn", lw.hc_ffn)):
        for part in ("fn", "base", "scale"):
            _eq(getattr(hc, part), src.raw(f"{p}hc_{site}_{part}"), f"{p}hc_{site}_{part}")
    _eq(lw.attn_norm, src.raw(p + "attn_norm.weight"), p + "attn_norm")
    _eq(lw.ffn_norm, src.raw(p + "ffn_norm.weight"), p + "ffn_norm")
    _check_attn(lw.attn, p, layer, src, cfg)
    m, f = lw.moe, p + "ffn."
    for name, got in (("gate.weight", m.gate), ("gate.bias", m.bias), ("gate.bias_vl", m.bias_vl)):
        _eq(got, src.raw(f + name), f + name)
    _eq(_mx8(m.shared_gu), src.dense(f + "shared_experts.w1", f + "shared_experts.w3"), f + "shared w1|w3")
    _eq(_mx8(m.shared_d), src.dense(f + "shared_experts.w2"), f + "shared w2")
    if layer is None:
        _check_experts4(m.experts, f, src, cfg)
    else:
        _check_exl3(m.experts, f, src, cfg)


def _engram_tables(model_dir: Path) -> list[tuple[Path, int, int]]:
    """(file, first byte, end byte) of every Engram table tensor."""

    pack, out = RefPack(model_dir), []
    for name in pack.where:
        if split.rule(name) == "engram":
            base, header = direct_read.read_header(pack.path(name))
            a, b = header[name]["data_offsets"]
            out.append((pack.path(name), base + a, base + b))
    return out


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
def test_every_object_holds_its_source_slice(linked, tiny_dir, reads, world, rank):
    cfg = Config.read(linked)
    w = loader.load(linked, cfg, rank, world, "comm", capacity=CAP)
    here = torch.device("cuda", torch.cuda.current_device())
    assert (w.cfg, w.rank, w.world, w.comm, w.device) == (cfg, rank, world, "comm", here)
    assert w.vocab_offset == rank * cfg.vocab_size // world
    src = Source(tiny_dir, rank, world)
    _eq(w.embed, src.raw("embed.weight"), "embed")
    _eq(w.norm, src.raw("norm.weight"), "norm")
    _eq(_mx8(w.head), src.dense("lm_head"), "lm_head")
    assert [lw.index for lw in w.layers] == list(range(cfg.num_hidden_layers))
    for lw in w.layers:
        _check_block(lw, lw.index, f"layers.{lw.index}.", lw.index, src, cfg)
    assert sorted(w.engram) == list(cfg.engram_layer_ids)
    for i, e in w.engram.items():
        p = f"layers.{i}.engram."
        _eq(_mx8(e.wkv), src.dense(p + "wkv"), p + "wkv")
        _eq(e.wqk, src.raw(p + "q_weight").float() * src.raw(p + "k_weight").float(), p + "q*k")
    d, n, last = w.dspark, cfg.num_hidden_layers, f"mtp.{cfg.num_nextn_predict_layers - 1}."
    for s, stage in enumerate(d.stages):
        _check_block(stage, n + s, f"mtp.{s}.", None, src, cfg)
    _eq(_mx8(d.main_proj), src.dense("mtp.0.main_proj"), "main_proj")
    _eq(d.main_norm, src.raw("mtp.0.main_norm.weight"), "main_norm")
    _eq(d.norm, src.raw(last + "norm.weight"), "dspark norm")
    _eq(d.markov_embed, src.raw(last + "markov_head.embed.weight"), "markov embed")
    _eq(d.markov_head.weight, src.raw(last + "markov_head.head.weight"), "markov head")
    _eq(d.conf, src.raw(last + "confidence_head.proj.weight").float(), "confidence")
    assert sorted(w.rope) == sorted(rope.KINDS)
    for kind, table in w.rope.items():
        _eq(table, rope.tables(cfg, kind, CAP, "cuda"), f"rope {kind}")
    used = {name for name in src.pack.where if split.rule(name) not in ("engram", "drop")}
    assert used - src.seen == set(), "every loaded tensor compared"
    assert reads and not [(f, a, n) for f, a, n in reads for g, lo, hi in _engram_tables(tiny_dir)
                          if f == g and a < hi and lo < a + n], "Engram tables are never read"


def test_a_layer_subset_without_dspark(linked, tiny_dir):
    cfg = Config.read(linked)
    w = loader.load(linked, cfg, 1, 2, None, layers=[4, 1, 4], dspark=False, capacity=CAP)
    assert [lw.index for lw in w.layers] == [1, 4] and w.dspark is None and sorted(w.engram) == [1]
    assert w.layers[1].attn.comp.ratio == cfg.compress_ratios[4]
    src = Source(tiny_dir, 1, 2)
    _check_block(w.layers[0], 1, "layers.1.", 1, src, cfg)
    with pytest.raises(ValueError, match="layers"):
        loader.load(linked, cfg, 0, 2, None, layers=[cfg.num_hidden_layers], capacity=CAP)
    with pytest.raises(ValueError, match="rank 2 of 2"):
        loader.load(linked, cfg, 2, 2, None, capacity=CAP)


def _estimate(model_dir: str, layers, world: int = 2) -> tuple[int, int]:
    """split.weights_estimate's resident bytes of the loaded groups and capacity's staging over them (3x the largest
    tensor or layer)."""

    total, groups, largest = 0, {}, 0
    for name, info in capacity.headers(model_dir).items():
        m = re.match(r"layers\.(\d+)\.", name)
        if m and int(m[1]) not in layers:
            continue
        size, mapped = split.weights_estimate(name, info, world)
        assert mapped == 0
        total, largest = total + size, max(largest, size)
        key = m[1] if m else name
        groups[key] = groups.get(key, 0) + size
    return total, 3 * max(largest, *groups.values())


@needs_model
def test_real_layers_and_dspark_stay_within_the_estimate_and_staging():
    cfg = Config.read(MODEL)
    estimate, staging = _estimate(MODEL, REAL_LAYERS)
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    w = loader.load(MODEL, cfg, 0, 2, None, layers=REAL_LAYERS, capacity=CAP)
    resident = torch.cuda.memory_allocated() - before - rope_bytes(cfg, CAP)
    peak = torch.cuda.max_memory_allocated() - before
    print(f"resident {resident / 2**30:.3f} GiB, estimate {estimate / 2**30:.3f} GiB "
          f"({(resident - estimate) / estimate:+.4%}); peak {peak / 2**30:.3f} GiB <= {(estimate + staging) / 2**30:.3f}")
    assert abs(resident - estimate) <= 0.01 * estimate
    assert peak <= estimate + staging + rope_bytes(cfg, CAP)
    src = Source(Path(MODEL), 0, 2)
    lw = w.layers[REAL_LAYERS.index(20)]
    _eq(_mx8(lw.attn.wq_b)[:64], src.dense("layers.20.attn.wq_b")[:64], "wq_b")
    name = "layers.20.ffn.experts.7.w1"
    want = exl3.dequant(src.pack.tensor(name + ".trellis").cuda(), cfg.codebook)[:, :cfg.moe_intermediate_size // 2]
    _eq(exl3.dequant(lw.moe.experts.keep[7], cfg.codebook), want, name)
    del w, lw
    gc.collect()
    torch.cuda.empty_cache()


@needs_model
def test_one_real_layer_loads_alone():
    cfg = Config.read(MODEL)
    w = loader.load(MODEL, cfg, 1, 2, None, layers=[39], dspark=False, capacity=CAP)
    assert len(w.layers) == 1 and w.dspark is None
    del w
    gc.collect()
    torch.cuda.empty_cache()

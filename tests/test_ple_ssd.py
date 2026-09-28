"""Flash Next's n-gram (PLE) rows read from SSD equal the memory map's bytes, and --ple-on-ssd reaches both loaders."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tensorfold import cli, families, hub
from tensorfold.cuda.geometry import indexed_weights
from tensorfold.families import qwen4_exp
from tensorfold.families.qwen4_exp import host_table, ssd_table
from tensorfold.families.qwen4_exp.host_table import HostTable
from tensorfold.families.qwen4_exp.ssd_table import SSDTable

PARTS = ("weight", "scales", "biases")
COUNTS = (37, 5, 64, 19, 3, 41, 1, 33)
GIB = 1024**3


def _write(path: Path, tensors: dict[str, np.ndarray]) -> dict:
    """A safetensors file of ``tensors`` (uint32 as U32, uint16 as bf16 bits), returning its header."""

    header, blobs, at = {"__metadata__": {"format": "mlx"}}, [], 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": "U32" if array.dtype == np.uint32 else "BF16", "shape": list(array.shape),
                        "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return header


def _shards(counts, groups: int = 5, seed: int = 16) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 2**32, (n, 4 * groups), dtype=np.uint32), rng.integers(0, 2**16, (n, groups),
             dtype=np.uint16), rng.integers(0, 2**16, (n, groups), dtype=np.uint16)) for n in counts]


def _checkpoint(folder: Path, counts=COUNTS, groups: int = 5, interleave: bool = True):
    """Shards ``emb.shard_{i}`` over two files (alternating, or halves with each part's tensors together)."""

    shards = _shards(counts, groups)
    owner = [i % 2 if interleave else int(i >= (len(counts) + 1) // 2) for i in range(len(counts))]
    files = [None] * len(counts)
    for f in range(2):
        mine = [i for i in range(len(counts)) if owner[i] == f]
        order = [(i, p) for i in mine for p in range(3)] if interleave else [(i, p) for p in range(3) for i in mine]
        tensors = {"model.norm.weight": np.arange(21, dtype=np.uint16).reshape(3, 7)}
        tensors.update({f"emb.shard_{i}.{PARTS[p]}": shards[i][p] for i, p in order})
        path = folder / f"model-0000{f + 1}-of-00002.safetensors"
        header = _write(path, tensors)
        for i in mine:
            files[i] = (path, *(header[f"emb.shard_{i}.{part}"] for part in PARTS))
    return files, tuple(np.concatenate(parts) for parts in zip(*shards))


def _same(got, want) -> None:
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype and g.shape == w.shape and g.tobytes() == w.tobytes()


def _data(path: Path) -> int:
    """Where a safetensors file's tensor bytes begin."""

    return 8 + struct.unpack("<Q", path.read_bytes()[:8])[0]


@pytest.fixture
def tables():
    made = []
    yield lambda files, **kw: made.append(SSDTable(files, **kw)) or made[-1]
    for table in made:
        table.close()


@pytest.mark.parametrize("nocache", [True, False])
@pytest.mark.parametrize("interleave", [True, False])
def test_rows_from_ssd_equal_the_memory_map_byte_for_byte(tmp_path, tables, interleave, nocache):
    files, table = _checkpoint(tmp_path, interleave=interleave)
    host, ssd = HostTable(files), tables(files, nocache=nocache)
    assert (ssd.rows, ssd.wrow, ssd.grow) == (host.rows, host.wrow, host.grow) == (sum(COUNTS), 80, 10)
    inner = np.cumsum(COUNTS)[:-1]
    edges = np.concatenate([inner - 1, inner, [0, ssd.rows - 1]])
    cases = [np.random.default_rng(3).integers(0, ssd.rows, (13, 16)),     # unordered
             np.array([5, 5, 5, 200, 5, 0, 200]),                         # repeated
             edges, edges[::-1],                                           # both sides of every shard boundary
             np.arange(ssd.rows),                                          # every row: runs read whole
             np.array(7), np.empty((0, 16), dtype=np.int64), []]           # one id, none
    for ids in cases:
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        want = tuple(part[flat] for part in table)
        _same(host.gather(ids), want)
        _same(ssd.gather(ids), want)


@pytest.mark.parametrize("threads", [1, 2, 16])
def test_a_prompt_chunks_gather_on_threads_gives_the_same_bytes(tmp_path, monkeypatch, threads):
    """A gather past 2 GATHER_SPLIT rows copies its rows on GATHER_THREADS threads: the bytes of one thread."""

    files, table = _checkpoint(tmp_path)
    monkeypatch.setattr(host_table, "GATHER_THREADS", threads)
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)                # the test shards are small: split every 8 rows
    host = HostTable(files)
    calls = []
    real = host._pool.map
    monkeypatch.setattr(host._pool, "map", lambda fn, jobs: calls.append(len(jobs)) or real(fn, jobs))
    for ids in (np.random.default_rng(5).integers(0, host.rows, (40, 16)),   # 640 rows: threaded when threads > 1
                np.arange(host.rows)[::-1], np.array([3, 3, 1]), []):         # every row reversed; a few; none
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        _same(host.gather(ids), tuple(part[flat] for part in table))
    assert (len(calls) > 0) == (threads > 1)


def test_from_checkpoint_picks_the_reader_by_argument_not_environment(tmp_path, monkeypatch):
    _, table = _checkpoint(tmp_path)
    monkeypatch.setenv("TF_NGRAM_HOST", "0")
    mapped = host_table.from_checkpoint(tmp_path, "emb", len(COUNTS))
    read = host_table.from_checkpoint(tmp_path, "emb", len(COUNTS), ssd=True)
    assert type(mapped) is HostTable and type(read) is SSDTable
    ids = np.arange(read.rows)[::-1]
    _same(read.gather(ids), mapped.gather(ids))
    read.close()


@pytest.mark.parametrize("nocache", [True, False])
def test_concurrent_gathers_each_get_their_own_rows(tmp_path, tables, nocache):
    files, _ = _checkpoint(tmp_path)
    host, ssd, start = HostTable(files), tables(files, nocache=nocache), Barrier(8)

    def run(seed: int) -> None:
        rng = np.random.default_rng(seed)
        start.wait(timeout=30)
        for _ in range(40):
            ids = rng.integers(0, ssd.rows, (int(rng.integers(1, 9)), 16))
            _same(ssd.gather(ids), host.gather(ids))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(run, range(8)))


def _recorded(monkeypatch) -> list[tuple[int, int, int]]:
    reads, real = [], os.pread

    def pread(fd: int, size: int, offset: int) -> bytes:
        reads.append((fd, offset, size))
        return real(fd, size, offset)

    monkeypatch.setattr(os, "pread", pread)
    return reads


def test_a_gather_reads_each_run_of_wanted_rows_once(tmp_path, monkeypatch, tables):
    files, table = _checkpoint(tmp_path, counts=(10_000,))
    ssd = tables(files)
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("a gather opened a file"))
    reads = _recorded(monkeypatch)
    ids = [9999, 5, 6, 5, 9999, 7, 4000]
    _same(ssd.gather(ids), tuple(part[ids] for part in table))
    path, *entries = files[0]
    want = []
    for entry, width in zip(entries, (80, 10, 10)):
        base = _data(path) + entry["data_offsets"][0]
        want += [(base + 5 * width, 3 * width), (base + 4000 * width, width), (base + 9999 * width, width)]
    assert sorted((offset, size) for _, offset, size in reads) == sorted(want)
    assert len({fd for fd, _, _ in reads}) == 1


def test_runs_split_at_the_read_bound_and_short_reads_are_finished(tmp_path, monkeypatch, tables):
    files, table = _checkpoint(tmp_path, counts=(20,))
    ssd = tables(files)
    monkeypatch.setattr(ssd_table, "MAX_READ", 200)          # 2 rows of words a read, 20 of scales or biases
    reads = _recorded(monkeypatch)
    _same(ssd.gather(np.arange(20)), table)
    assert sorted(size for _, _, size in reads) == [160] * 10 + [200] * 2
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, size, offset: real(fd, min(size, 33), offset))
    _same(ssd.gather(np.arange(20)[::-1]), tuple(part[::-1] for part in table))


def test_prefetch_and_an_empty_gather_read_nothing(tmp_path, monkeypatch, tables):
    files, _ = _checkpoint(tmp_path)
    ssd = tables(files)
    for name in ("pread", "read", "open"):
        monkeypatch.setattr(os, name, lambda *a, **k: pytest.fail("file I/O"))
    assert ssd.prefetch() == 0.0 and ssd.prefetch(workers=2) == 0.0
    assert [a.shape for a in ssd.gather(np.empty((0, 16), dtype=np.int64))] == [(0, 20), (0, 5), (0, 5)]


@pytest.mark.parametrize("nocache", [True, False])
def test_each_file_is_opened_once_with_one_cache_hint(tmp_path, monkeypatch, tables, nocache):
    files, _ = _checkpoint(tmp_path)
    real_open, opened, hints = os.open, [], []
    monkeypatch.setattr(os, "open", lambda path, *a, **k: opened.append(Path(path).name) or real_open(path, *a, **k))
    if sys.platform == "darwin":
        import fcntl

        real = fcntl.fcntl
        hint = getattr(fcntl, "F_NOCACHE", 48)
        monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, arg=0: hints.append(cmd) or real(fd, cmd, arg))
    elif hasattr(os, "posix_fadvise"):
        real, hint = os.posix_fadvise, os.POSIX_FADV_RANDOM
        monkeypatch.setattr(os, "posix_fadvise", lambda fd, at, n, how: hints.append(how) or real(fd, at, n, how))
    else:
        hint = None
    ssd = tables(files, nocache=nocache)
    for _ in range(3):
        ssd.gather(np.arange(ssd.rows))
    assert sorted(opened) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert [h for h in hints if h == hint] == ([hint] * 2 if nocache and hint is not None else [])


def test_close_and_collection_release_the_files(tmp_path, monkeypatch):
    files, _ = _checkpoint(tmp_path)
    real_open, fds = os.open, []
    monkeypatch.setattr(os, "open", lambda *a, **k: fds.append(real_open(*a, **k)) or fds[-1])
    ssd = SSDTable(files)
    ssd.close()
    ssd.close()
    with pytest.raises(ValueError, match="closed"):
        ssd.gather([0])
    later = SSDTable(files)
    del later
    assert len(fds) == 4
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def _edited(files, index: int, part: int, **fields):
    copy = [(path, *(dict(e) for e in entries)) for path, *entries in files]
    copy[index][part + 1].update(fields)
    return copy


@pytest.mark.parametrize("part, fields, match", [
    (0, {"dtype": "I32"}, "weight must be a U32"), (1, {"dtype": "F16"}, "scales must be a BF16"),
    (2, {"dtype": "U16"}, "biases must be a BF16"), (0, {"shape": [37]}, r"not \[rows, columns\]"),
    (0, {"shape": [0, 20]}, r"not \[rows, columns\]"), (0, {"shape": [37, 20, 1]}, r"not \[rows, columns\]"),
    (0, {"data_offsets": [0, 2**40]}, "disagree"), (1, {"data_offsets": [-2, 368]}, "disagree"),
    (2, {"shape": [37, 4]}, "disagree"),
])
def test_headers_the_reader_cannot_use_are_refused(tmp_path, part, fields, match):
    files, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match=match):
        SSDTable(_edited(files, 0, part, **fields))


def test_shards_that_are_not_4_bit_group_32_rows_or_differ_in_width_are_refused(tmp_path):
    w, s, b = _shards((7,))[0]
    for name, tensors in (("words", (w[:, :19], s, b)), ("rows", (w, s[:6], b[:6])), ("biases", (w, s, b[:, :4]))):
        header = _write(tmp_path / f"{name}.safetensors", dict(zip(PARTS, tensors)))
        with pytest.raises(ValueError, match="not 4-bit rows"):
            SSDTable([(tmp_path / f"{name}.safetensors", *(header[p] for p in PARTS))])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    wide, _ = _checkpoint(tmp_path / "a", counts=(7,), groups=5)
    narrow, _ = _checkpoint(tmp_path / "b", counts=(3,), groups=4)
    with pytest.raises(ValueError, match="differ in row width"):
        SSDTable(wide + narrow)
    with pytest.raises(ValueError, match="no shards"):
        SSDTable([])


def test_a_truncated_file_is_refused_at_open_with_its_files_closed(tmp_path, monkeypatch):
    files, _ = _checkpoint(tmp_path, counts=(7,))
    path = files[0][0]
    path.write_bytes(path.read_bytes()[:-1])
    real_open, fds = os.open, []
    monkeypatch.setattr(os, "open", lambda *a, **k: fds.append(real_open(*a, **k)) or fds[-1])
    with pytest.raises(ValueError, match="pass the file's end"):
        SSDTable(files)
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="truncated safetensors header"):
        SSDTable(files)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("ids, error", [
    ([-1], ValueError), ([sum(COUNTS)], ValueError), (np.array([2**64 - 1], dtype=np.uint64), ValueError),
    ([1.0], TypeError), (np.array([True]), TypeError),
])
def test_ids_outside_the_table_are_refused(tmp_path, tables, ids, error):
    files, _ = _checkpoint(tmp_path)
    with pytest.raises(error):
        tables(files).gather(ids)


def test_a_file_cut_short_after_opening_fails_the_gather(tmp_path, tables):
    files, _ = _checkpoint(tmp_path, counts=(7,))
    ssd = tables(files)
    os.truncate(files[0][0], _data(files[0][0]))
    with pytest.raises(OSError, match="short read"):
        ssd.gather([6])


@pytest.mark.parametrize("ssd", [False, True])
@pytest.mark.parametrize("case", ["missing", "split", "duplicated"])
def test_from_checkpoint_refuses_a_shard_it_cannot_place(tmp_path, case, ssd):
    files, _ = _checkpoint(tmp_path, counts=(7, 3))
    count = 3 if case == "missing" else 2
    if case == "split":
        w, s, b = _shards((7,))[0]
        _write(tmp_path / "model-00001-of-00002.safetensors", {"emb.shard_0.weight": w, "emb.shard_0.biases": b})
        _write(tmp_path / "model-00003-of-00003.safetensors", {"emb.shard_0.scales": s})
    if case == "duplicated":
        shutil.copy(files[1][0], tmp_path / "model-copy.safetensors")
    shard = {"missing": "shard_2", "split": "shard_0", "duplicated": "shard_1"}[case]
    with pytest.raises(ValueError, match=f"emb.{shard}: expected"):
        host_table.from_checkpoint(tmp_path, "emb", count, ssd=ssd)


@pytest.mark.parametrize("raw", [b"", b"123", struct.pack("<Q", 100) + b"{}", struct.pack("<Q", 4) + b"nope",
                                 struct.pack("<Q", 2) + b"[]"])
def test_bad_safetensors_headers_are_refused(tmp_path, raw):
    (tmp_path / "model.safetensors").write_bytes(raw)
    with pytest.raises(ValueError):
        host_table.read_header(tmp_path / "model.safetensors")


def test_only_flash_next_has_ple_tables_and_their_bytes_are_counted(tmp_path):
    assert [k for k, f in families.families().items() if hasattr(f.package, "ple_bytes")] == ["qwen4_exp"]
    files, table = _checkpoint(tmp_path)
    _write(tmp_path / "model-00003-of-00003.safetensors",
           {"language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": table[0][:4],
            "language_model.model.layers.0.ple.key_proj.weight": table[0][4:9]})
    assert qwen4_exp.ple_bytes(tmp_path) == 4 * 80


def _flash_next(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen4_exp",
                                                    "quantization": {"bits": 4, "group_size": 32}}))
    return folder


@pytest.mark.parametrize("repo", [False, True])
def test_a_family_without_ple_tables_refuses_the_flag_before_any_download(tmp_path, monkeypatch, capsys, repo):
    folder = tmp_path / "nemotron"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "nemotron_h",
                                                    "quantization": {"bits": 4, "group_size": 64}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hub, "cached", lambda repo_id, **_: folder)
    for name in ("pull", "resolve"):
        monkeypatch.setattr(hub, name, lambda *a, **k: pytest.fail("downloaded before refusing"))
    model = "owner/model" if repo else str(folder)
    assert cli.main(["serve", model, "--ple-on-ssd", "--no-update-check"]) == 1
    assert "--ple-on-ssd: Nemotron 3.5 Lightning has no n-gram (PLE) tables" in capsys.readouterr().err


def test_the_flag_reaches_the_cuda_engine_only_when_given(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    class Reached(Exception):
        pass

    seen = []
    family = SimpleNamespace(title="fixture", model_type="qwen4_exp",
                             package=SimpleNamespace(cuda_engine=lambda path, **k: seen.append(k) or _raise(Reached)))
    for flag in ([], ["--ple-on-ssd"]):
        args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", *flag])
        with pytest.raises(Reached):
            cli._serve_cuda(args, family, tmp_path, 4096)
    assert "ple_on_ssd" not in seen[0] and seen[1]["ple_on_ssd"] is True
    monkeypatch.setattr(fn_engine, "FlashNextEngine", lambda *a, **k: SimpleNamespace(**k))
    assert qwen4_exp.cuda_engine(tmp_path, no_drafts=True, ple_on_ssd=True).ple_on_ssd is True
    assert qwen4_exp.cuda_engine(tmp_path, no_drafts=True).ple_on_ssd is False


def _raise(exc: type[Exception]):
    raise exc


def test_the_flag_reaches_the_metal_family_only_when_given(tmp_path, mlx_cpu):
    class Reached(Exception):
        pass

    seen = []
    family = SimpleNamespace(title="fixture", model_type="qwen4_exp",
                             package=SimpleNamespace(load=lambda path, **k: seen.append(k) or _raise(Reached)))
    for flag in ([], ["--ple-on-ssd"]):
        args = cli.build_parser().parse_args(["serve", str(tmp_path), "--snapshot-dir", "none", *flag])
        with pytest.raises(Reached):
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
    assert "ple_on_ssd" not in seen[0] and seen[1]["ple_on_ssd"] is True


def test_cuda_admission_counts_no_mapped_pages_for_tables_on_ssd():
    info = {"dtype": "U32", "shape": [1000, 20], "data_offsets": [0, 80_000]}
    name = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    assert indexed_weights(1, True)(name, info) == (0, 80_000)
    assert indexed_weights(1, True, mapped_tables=False)(name, info) == (0, 0)


def test_the_flag_reaches_the_metal_loader(tmp_path, monkeypatch):
    seen = []
    runtime = ModuleType("tensorfold.families.qwen4_exp.runtime")
    runtime.load = lambda path, **k: seen.append(k) or ("model", "tokenizer")
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    assert qwen4_exp.load(tmp_path, ple_on_ssd=True) == ("model", "tokenizer")
    qwen4_exp.load(tmp_path)
    qwen4_exp.load(tmp_path, ssd_experts=24.0)
    assert seen == [{"drafts": 0, "ple_on_ssd": True, "ssd_experts": None},
                    {"drafts": 0, "ple_on_ssd": False, "ssd_experts": None},
                    {"drafts": 0, "ple_on_ssd": False, "ssd_experts": 24.0}]


def _fake_mlx(monkeypatch) -> None:
    core = ModuleType("mlx.core")
    core.set_cache_limit = core.set_memory_limit = lambda value: None
    core.device_info = lambda: {"max_recommended_working_set_size": 64 * GIB, "memory_size": 128 * GIB}
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_tables_left_on_ssd_do_not_count_against_the_memory_budget(tmp_path, monkeypatch):
    folder = _flash_next(tmp_path / "flash")
    ple = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    rows = 3 * GIB // 80
    header = {ple: {"dtype": "U32", "shape": [rows, 20], "data_offsets": [0, rows * 80]},
              "language_model.model.norm.weight": {"dtype": "BF16", "shape": [512],
                                                   "data_offsets": [rows * 80, rows * 80 + 1024]}}
    text = json.dumps(header).encode()
    with open(folder / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        f.truncate(8 + len(text) + rows * 80 + 1024)          # sparse: 3 GiB of tables, no blocks written
    _fake_mlx(monkeypatch)
    for key, value in qwen4_exp.MLX_ENV.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "5")      # 2 GiB for weights beside the 3 GiB process reserve
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)

    class Reached(Exception):
        pass

    monkeypatch.setattr(cli, "_serve_mlx", lambda *a, **k: _raise(Reached))
    args = cli.build_parser().parse_args(["serve", str(folder), "--no-update-check", "--backend", "mlx"])
    with pytest.raises(ValueError, match="do not fit"):
        cli.cmd_serve(args)
    args.ple_on_ssd = True
    with pytest.raises(Reached):
        cli.cmd_serve(args)


def test_routed_experts_left_on_ssd_count_as_their_pool(tmp_path, monkeypatch):
    folder = _flash_next(tmp_path / "flash")
    expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    size = 3 * GIB
    header = {expert: {"dtype": "U32", "shape": [512, size // 2048], "data_offsets": [0, size]},
              "language_model.model.norm.weight": {"dtype": "BF16", "shape": [512], "data_offsets": [size, size + 1024]}}
    text = json.dumps(header).encode()
    with open(folder / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        f.truncate(8 + len(text) + size + 1024)                 # sparse: 3 GiB of experts, no blocks written
    _fake_mlx(monkeypatch)
    for key, value in qwen4_exp.MLX_ENV.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    monkeypatch.setenv("TENSORFOLD_MEMORY_LIMIT_GB", "5")      # 2 GiB for weights beside the 3 GiB process reserve
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)

    class Reached(Exception):
        pass

    monkeypatch.setattr(cli, "_serve_mlx", lambda *a, **k: _raise(Reached))
    args = cli.build_parser().parse_args(["serve", str(folder), "--no-update-check", "--backend", "mlx"])
    with pytest.raises(ValueError, match="--ssd-experts GIB"):
        cli.cmd_serve(args)
    args.ssd_experts = 2.5                                      # a 2.5 GiB pool: still past the 2 GiB left
    with pytest.raises(ValueError, match="do not fit"):
        cli.cmd_serve(args)
    args.ssd_experts = 1.0
    with pytest.raises(Reached):
        cli.cmd_serve(args)


@pytest.fixture
def mlx_cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(previous)


TEXT = {
    "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 97, "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
    "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "hc_count": 4, "hc_lowrank": 16,
    "indexer_n_heads": 2, "indexer_head_dim": 16, "indexer_budget": 8, "indexer_compress_ratio": 4,
    "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ngram_size": 3,
    "heads_per_ngram": 1, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 4, "eos_token_id": 5,
}


def test_metal_loads_give_the_same_logits_with_the_tables_on_ssd(tmp_path, monkeypatch, mlx_cpu):
    """A tiny Flash Next on the CPU device: GPU tables, memory-mapped host rows and rows from SSD agree bit for bit."""

    mx = mlx_cpu
    import mlx.nn as nn
    import mlx_lm.utils
    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp import model as q4

    config = {"model_type": "qwen4_exp", "text_config": TEXT, "quantization": {"group_size": 32, "bits": 4}}
    mx.random.seed(0)
    model = q4.Qwen4Exp(q4.Config.from_dict(config))
    for shard in model.layers[1].ple.ple_embedding.shards:
        shard.weight = shard.weight.astype(mx.bfloat16)          # bf16 scales and biases, as the checkpoint has
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda path, _: ".ple_embedding.shards." in path)
    params = {"language_model." + k.replace(".ple_embedding.shards.", ".ple_embedding.ngram_embedding.shard_"): v
              for k, v in tree_flatten(model.parameters())}
    owner = {n: int(n.split(".shard_")[1].split(".")[0]) if ".shard_" in n else i for i, n in enumerate(params)}
    for f in range(2):                       # a shard's words, scales and biases share a file, as in the checkpoint
        mx.save_safetensors(str(tmp_path / f"model-0000{f + 1}-of-00002.safetensors"),
                            {n: a for n, a in params.items() if owner[n] % 2 == f}, metadata={"format": "mlx"})
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: "tokenizer")
    monkeypatch.setenv("TF_FLASH_FUSED", "0")

    def logits(**options) -> list:
        loaded, _ = q4.load(tmp_path, **options)
        cache = loaded.make_cache()
        first = loaded(np.array([[11, 42, 5, 17, 42, 11, 60, 3, 5, 5, 90, 17]]), cache)
        return [loaded.layers[1].ple.ple_embedding.host, first, loaded(np.array([[33]]), cache)]

    monkeypatch.setenv("TF_NGRAM_HOST", "0")
    on_gpu = logits()
    monkeypatch.setenv("TF_NGRAM_HOST", "1")
    mapped = logits()
    monkeypatch.delenv("TF_NGRAM_HOST")
    on_ssd = logits(ple_on_ssd=True)
    assert on_gpu[0] is None and type(mapped[0].table) is HostTable and type(on_ssd[0].table) is SSDTable
    for a, b, c in zip(on_gpu[1:], mapped[1:], on_ssd[1:]):
        assert mx.array_equal(a, b).item() and mx.array_equal(a, c).item()
    on_ssd[0].close()
    monkeypatch.setenv("TF_NGRAM_HOST", "0")
    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("read weights before refusing"))
    with pytest.raises(ValueError, match="unset TF_NGRAM_HOST=0"):
        q4.load(tmp_path, ple_on_ssd=True)


def test_tables_on_ssd_take_the_host_path_without_asking_the_gpu(tmp_path, monkeypatch, mlx_cpu):
    from tensorfold.families.qwen4_exp import model as q4

    monkeypatch.setattr(mlx_cpu, "device_info", lambda: pytest.fail("asked the GPU"), raising=False)
    monkeypatch.delenv("TF_NGRAM_HOST", raising=False)
    assert q4.ngrams_on_host(tmp_path, ssd=True)
    monkeypatch.setenv("TF_NGRAM_HOST", "1")
    assert q4.ngrams_on_host(tmp_path, ssd=True)
    monkeypatch.setenv("TF_NGRAM_HOST", "0")
    with pytest.raises(ValueError, match="unset TF_NGRAM_HOST=0"):
        q4.ngrams_on_host(tmp_path, ssd=True)

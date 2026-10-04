"""SSD PLE layouts preserve mapped bf16 bytes without mapping or warming whole tables."""

from __future__ import annotations

import gc
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table, ssd_table
from tensorfold.families.qwen4_exp.host_table import SSDValueTable, open_table

LAYOUTS = ("bf16", "fp8", "nvfp4", "mlx")
COUNTS = (7, 3, 11)
WIDTH = 32


def _write(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> dict:
    """Write exact tensor bytes and explicit safetensors dtypes."""

    header, blobs, offset = {}, [], 0
    for name, (dtype, values) in tensors.items():
        blob = values.tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return header


def _checkpoint(folder: Path, layout: str, *, split: bool = False):
    """Create alternating-file shards; NVFP4 blocks can occupy their own indexed file."""

    rng = np.random.default_rng(17)
    by_file, shards, locations = {}, [], {}
    for index, rows in enumerate(COUNTS):
        key, filename = f"emb.shard_{index}", f"model-{index % 2}.safetensors"
        shards.append((filename, key))
        if layout == "bf16":
            parts = {"weight": ("BF16", rng.integers(0, 65536, (rows, WIDTH), dtype=np.uint16))}
        elif layout == "fp8":
            parts = {"weight": ("F8_E4M3", np.resize(np.arange(256, dtype=np.uint8), (rows, WIDTH)))}
        elif layout == "nvfp4":
            parts = {"weight": ("U8", np.resize(np.arange(256, dtype=np.uint8), (rows, WIDTH // 2))),
                     "weight_scale": ("F8_E4M3", rng.integers(0, 256, (rows, WIDTH // 16), dtype=np.uint8))}
        else:
            parts = {"weight": ("U32", rng.integers(0, 2**32, (rows, WIDTH // 8), dtype=np.uint32)),
                     "scales": ("BF16", rng.integers(0, 65536, (rows, WIDTH // 32), dtype=np.uint16)),
                     "biases": ("BF16", rng.integers(0, 65536, (rows, WIDTH // 32), dtype=np.uint16))}
        for suffix, tensor in parts.items():
            name = f"{key}.{suffix}"
            source = "model-blocks.safetensors" if split and suffix == "weight_scale" else filename
            by_file.setdefault(source, {})[name] = tensor
            locations[name] = source
    for filename, tensors in by_file.items():
        _write(folder / filename, tensors)
    return shards, locations


def _same(actual, expected) -> None:
    """Compare every output bit, including NaNs and signed zeros."""

    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("scale", [1.0, 0.731, -1.125, 2**-20])
def test_all_layouts_equal_mapped_bytes_at_boundaries_repeats_scalar_and_empty(tmp_path, layout, scale):
    shards, locations = _checkpoint(tmp_path, layout, split=layout == "nvfp4")
    mapped = open_table(tmp_path, shards, lambda _: scale, tensor_files=locations)
    disk = open_table(tmp_path, shards, lambda _: scale, ssd=True, tensor_files=locations)
    try:
        assert disk.rows == mapped.rows == sum(COUNTS)
        assert disk.nbytes == mapped.nbytes
        assert not hasattr(disk, "values") and not hasattr(disk, "files")
        for ids in ([0, 6, 7, 9, 10, 20], [8, 8, 1, 20, 8], np.array(7), [],
                    np.empty((0, 3), dtype=np.int64), np.arange(21)[::-1],
                    np.random.default_rng(5).integers(0, 21, (13, 16))):
            _same(disk.gather(ids), mapped.gather(ids))
        if layout != "mlx":
            assert disk.bits == 16 and disk.width == WIDTH and disk.lock() is False
    finally:
        disk.close()
        mapped._pool.shutdown(wait=True)


@pytest.mark.parametrize("layout", ["fp8", "nvfp4"])
@pytest.mark.parametrize("scale, expected", [(1 + 1 / 256, 0x3F80), (1 + 3 / 256, 0x3F82)])
def test_round_to_nearest_even_low_nibble_first_and_signed_zero(tmp_path, layout, scale, expected):
    codes = np.full((1, 16 if layout == "nvfp4" else 32), 0x22 if layout == "nvfp4" else 0x38,
                    dtype=np.uint8)
    codes[0, 0] = 0x82 if layout == "nvfp4" else 0x80
    tensors = {"emb.shard_0.weight": ("U8" if layout == "nvfp4" else "F8_E4M3", codes)}
    if layout == "nvfp4":
        tensors["emb.shard_0.weight_scale"] = ("F8_E4M3", np.full((1, 2), 0x38, dtype=np.uint8))
    _write(tmp_path / "model.safetensors", tensors)
    disk = open_table(tmp_path, [("model.safetensors", "emb.shard_0")], lambda _: scale, ssd=True)
    try:
        got = disk.gather([0])[0]
        assert int(got[-1]) == expected
        if layout == "nvfp4":
            assert got[:2].tolist() == [expected, 0x8000]
        else:
            assert got[0] == 0x8000
    finally:
        disk.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_gather_deduplicates_coalesces_and_finishes_short_reads(tmp_path, monkeypatch, layout):
    shards, _ = _checkpoint(tmp_path, layout)
    disk = open_table(tmp_path, shards, lambda _: 0.73, ssd=True)
    real, calls = os.pread, []
    try:
        expected = disk.gather([5, 3, 4, 5, 3])

        def read(fd: int, size: int, offset: int) -> bytes:
            calls.append((fd, size, offset))
            return real(fd, size, offset)

        monkeypatch.setattr(os, "pread", read)
        _same(disk.gather([5, 3, 4, 5, 3]), expected)
        assert len(calls) == (3 if layout == "mlx" else 2 if layout == "nvfp4" else 1)
        assert sorted(n for _, n, _ in calls) == sorted(3 * w for w in disk.widths)
        monkeypatch.setattr(os, "pread", lambda fd, size, offset: real(fd, min(size, 3), offset))
        _same(disk.gather([5, 3, 4, 5, 3]), expected)
    finally:
        disk.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_mapping_no_prefetch_and_no_io_for_empty_ids(tmp_path, monkeypatch, layout):
    shards, _ = _checkpoint(tmp_path, layout)
    monkeypatch.setattr(np, "memmap", lambda *a, **kw: pytest.fail("SSD path mapped a file"))
    disk = open_table(tmp_path, shards, lambda _: 1.0, ssd=True)
    try:
        monkeypatch.setattr(os, "pread", lambda *a, **kw: pytest.fail("unrequested table read"))
        assert disk.prefetch() == 0.0
        got = disk.gather([])
        assert all(a.shape[0] == 0 for a in (got if isinstance(got, tuple) else (got,)))
    finally:
        disk.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_concurrent_gathers_return_each_callers_bytes(tmp_path, layout):
    shards, _ = _checkpoint(tmp_path, layout)
    disk = open_table(tmp_path, shards, lambda _: 0.37, ssd=True)
    ids = [np.random.default_rng(seed).integers(0, 21, (16, 7)) for seed in range(8)]
    expected = [disk.gather(rows) for rows in ids]
    try:
        with ThreadPoolExecutor(8) as pool:
            for _ in range(10):
                for got, want in zip(pool.map(disk.gather, ids), expected, strict=True):
                    _same(got, want)
    finally:
        disk.close()


@pytest.mark.parametrize("layout", ["bf16", "fp8", "nvfp4"])
@pytest.mark.parametrize("ids, exception", [([-1], ValueError), ([21], ValueError), ([1.5], TypeError),
                                         ([True], TypeError), (np.array([2**64 - 1], dtype=np.uint64), ValueError)])
def test_bad_row_ids_are_refused(tmp_path, layout, ids, exception):
    shards, _ = _checkpoint(tmp_path, layout)
    disk = open_table(tmp_path, shards, lambda _: 1.0, ssd=True)
    try:
        with pytest.raises(exception):
            disk.gather(ids)
    finally:
        disk.close()


@pytest.mark.parametrize("layout", ["bf16", "fp8", "nvfp4"])
@pytest.mark.parametrize("field, value", [("dtype", "F16"), ("shape", [7]), ("shape", [True, 32]),
                                         ("data_offsets", [-1, 20]), ("data_offsets", [0, 2**40])])
def test_bad_component_metadata_is_refused_and_files_closed(tmp_path, monkeypatch, layout, field, value):
    shards, _ = _checkpoint(tmp_path, layout)
    path = tmp_path / shards[0][0]
    entry = host_table.read_header(path)["emb.shard_0.weight"]
    entry[field] = value
    opened, real = [], os.open
    monkeypatch.setattr(os, "open", lambda *a, **kw: opened.append(real(*a, **kw)) or opened[-1])
    with pytest.raises(ValueError):
        SSDValueTable([[(path, entry)]], "fp8" if layout == "fp8" else "bf16")
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("case", ["rows", "block_width", "shard_width", "missing", "mixed", "empty"])
def test_invalid_layouts_are_refused(tmp_path, case):
    shards, _ = _checkpoint(tmp_path, "nvfp4")
    path = tmp_path / shards[0][0]
    header = host_table.read_header(path)
    weight, block = header["emb.shard_0.weight"], header["emb.shard_0.weight_scale"]
    if case in ("rows", "block_width"):
        block = dict(block, shape=[6, 2] if case == "rows" else [7, 1])
        begin = block["data_offsets"][0]
        block["data_offsets"] = [begin, begin + np.prod(block["shape"]).item()]
        with pytest.raises(ValueError, match="row count|every 16"):
            SSDValueTable([[(path, weight), (path, block)]], "nvfp4")
    elif case == "shard_width":
        narrower = dict(weight, shape=[7, 8])
        narrower["data_offsets"] = [weight["data_offsets"][0], weight["data_offsets"][0] + 56]
        smaller = dict(block, shape=[7, 1])
        smaller["data_offsets"] = [block["data_offsets"][0], block["data_offsets"][0] + 7]
        with pytest.raises(ValueError, match="differ in row width"):
            SSDValueTable([[(path, weight), (path, block)], [(path, narrower), (path, smaller)]], "nvfp4")
    elif case == "missing":
        with pytest.raises(ValueError, match="missing tensor components"):
            SSDValueTable([[(path, weight)]], "nvfp4")
    elif case == "mixed":
        _write(tmp_path / "bf16.safetensors", {"other.weight": ("BF16", np.zeros((1, WIDTH), dtype=np.uint16))})
        with pytest.raises(ValueError, match="mix layouts"):
            open_table(tmp_path, shards + [("bf16.safetensors", "other")], lambda _: 1.0, ssd=True)
    else:
        with pytest.raises(ValueError, match="no shards"):
            SSDValueTable([], "bf16")


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf, 1e100, [1.0, 2.0]])
def test_non_scalar_or_non_finite_scales_fail_before_open(tmp_path, monkeypatch, scale):
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("invalid scale opened a file"))
    with pytest.raises(ValueError, match="finite fp32 scalar"):
        SSDValueTable([], "nvfp4", scale)


@pytest.mark.parametrize("layout", ["bf16", "fp8", "nvfp4"])
def test_truncation_at_open_and_after_open_and_explicit_cleanup(tmp_path, layout):
    shards, _ = _checkpoint(tmp_path, layout)
    path = tmp_path / shards[0][0]
    disk = open_table(tmp_path, shards, lambda _: 1.0, ssd=True)
    fds = list(disk._fds)
    os.truncate(path, 8 + struct.unpack("<Q", path.read_bytes()[:8])[0])
    try:
        with pytest.raises(OSError, match="short read"):
            disk.gather([0])
    finally:
        disk.close()
        disk.close()
    with pytest.raises(ValueError, match="closed"):
        disk.gather([0])
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(ValueError, match="file's end"):
        open_table(tmp_path, shards, lambda _: 1.0, ssd=True)


@pytest.mark.parametrize("raw", [b"", b"123", struct.pack("<Q", 100) + b"{}",
                                 struct.pack("<Q", 4) + b"nope", struct.pack("<Q", 2) + b"[]"])
def test_malformed_files_are_refused(tmp_path, raw):
    (tmp_path / "model.safetensors").write_bytes(raw)
    with pytest.raises(ValueError):
        open_table(tmp_path, [("model.safetensors", "emb")], lambda _: 1.0, ssd=True)


def test_collection_releases_every_file_and_worker(tmp_path):
    shards, locations = _checkpoint(tmp_path, "nvfp4", split=True)
    disk = open_table(tmp_path, shards, lambda _: 1.0, ssd=True, tensor_files=locations)
    disk.gather([0, 7])
    fds, pool = list(disk._fds), disk._pool
    del disk
    gc.collect()
    assert pool._shutdown
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_mixed_layouts_and_inconsistent_widths_are_refused(tmp_path):
    path = tmp_path / "mixed.safetensors"
    _write(path, {"a.weight": ("BF16", np.zeros((2, 32), np.uint16)),
                  "b.weight": ("F8_E4M3", np.zeros((2, 32), np.uint8))})
    with pytest.raises(ValueError, match="mix layouts"):
        open_table(tmp_path, [(path.name, "a"), (path.name, "b")], lambda _: 1.0, ssd=True)
    h = _write(path, {"a.weight": ("BF16", np.zeros((2, 32), np.uint16)),
                      "b.weight": ("BF16", np.zeros((2, 16), np.uint16))})
    with pytest.raises(ValueError, match="differ in row width"):
        SSDValueTable([[(path, h["a.weight"])], [(path, h["b.weight"])]], "bf16")
    with pytest.raises(ValueError, match="no shards"):
        SSDValueTable([], "bf16")
    with pytest.raises(ValueError, match="missing tensor components"):
        SSDValueTable([[(path, h["a.weight"])]], "nvfp4")


def test_every_nvfp4_nibble_and_sign_matches_exact_bf16_values(tmp_path):
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    _write(tmp_path / "model.safetensors", {
        "emb.weight": ("U8", codes),
        "emb.weight_scale": ("F8_E4M3", np.full((16, 2), 0x38, dtype=np.uint8)),
    })
    table = open_table(tmp_path, [("model.safetensors", "emb")], lambda _: 1.0, ssd=True)
    exact = np.array([0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                      0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0], dtype=np.uint16)
    want = np.empty((16, 32), dtype=np.uint16)
    want[:, 0::2], want[:, 1::2] = exact[codes & 15], exact[codes >> 4]
    try:
        _same(table.gather(np.arange(16)), want)
    finally:
        table.close()


def test_fp8_nan_codes_keep_canonical_nan_bits(tmp_path):
    _write(tmp_path / "model.safetensors", {
        "emb.weight": ("F8_E4M3", np.array([[0x7F, 0xFF, 0x00, 0x80]], dtype=np.uint8)),
    })
    table = open_table(tmp_path, [("model.safetensors", "emb")], lambda _: 0.37, ssd=True)
    try:
        assert table.gather([0]).tolist() == [[0x7FC0, 0x7FC0, 0x0000, 0x8000]]
    finally:
        table.close()


@pytest.mark.parametrize("record", [None, [], "BF16"])
def test_malformed_tensor_records_are_refused(tmp_path, record):
    raw = json.dumps({"emb.weight": record}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ValueError, match="invalid n-gram tensor metadata"):
        open_table(tmp_path, [("model.safetensors", "emb")], lambda _: 1.0, ssd=True)


def test_a_row_wider_than_read_bound_is_split(tmp_path, monkeypatch):
    path = tmp_path / "wide.safetensors"
    values = np.arange(3 * 160, dtype=np.uint16).reshape(3, 160)
    h = _write(path, {"weight": ("BF16", values)})
    disk = SSDValueTable([[(path, h["weight"])]], "bf16")
    real, sizes = os.pread, []
    monkeypatch.setattr(ssd_table, "MAX_READ", 37)

    def recorded(fd: int, size: int, offset: int) -> bytes:
        sizes.append(size)
        return real(fd, size, offset)

    monkeypatch.setattr(os, "pread", recorded)
    try:
        _same(disk.gather([2, 0, 2]), values[[2, 0, 2]])
        assert max(sizes) <= 37 and sum(sizes) == 2 * 320
    finally:
        disk.close()


def test_engine_close_stops_scheduler_then_closes_distinct_tables_once():
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    calls = []
    table = SimpleNamespace(close=lambda: calls.append("table"))
    other = SimpleNamespace(close=lambda: calls.append("other"))
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine.scheduler = SimpleNamespace(close=lambda: calls.append("scheduler"))
    engine.w = SimpleNamespace(layers=[
        SimpleNamespace(ple=SimpleNamespace(table=table)),
        SimpleNamespace(ple=SimpleNamespace(table=table)),
        SimpleNamespace(ple=SimpleNamespace(table=other)),
        SimpleNamespace(ple=SimpleNamespace(table=object())),
        SimpleNamespace(ple=None),
    ])
    engine.close()
    engine.close()
    assert calls == ["scheduler", "table", "other"]
    assert engine.scheduler is None


def test_engine_close_releases_wrapped_ssd_files_and_both_pools(tmp_path):
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    shards, _ = _checkpoint(tmp_path, "nvfp4")
    table = open_table(tmp_path, shards, lambda _: 1.0, ssd=True)
    ahead = host_table.ReadAhead(table)
    ahead.read_ahead(np.array([0, 1, 7]))
    fds, pools = list(table._fds), [table._pool, ahead._pool]
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine.scheduler = None
    engine.w = SimpleNamespace(layers=[SimpleNamespace(ple=SimpleNamespace(table=ahead))])
    engine.close()
    engine.close()
    assert not ahead._ahead
    assert all(pool._shutdown for pool in pools)
    assert all(not thread.is_alive() for pool in pools for thread in pool._threads)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(ValueError, match="closed"):
        ahead.gather(np.array([0]))
    with pytest.raises(ValueError, match="closed"):
        ahead.read_ahead(np.array([0]))


def test_read_ahead_close_finishes_active_reads_before_closing_reader():
    from threading import Event
    from types import SimpleNamespace

    entered, release, closed = Event(), Event(), Event()

    def gather(ids: np.ndarray) -> np.ndarray:
        entered.set()
        assert release.wait(timeout=5)
        assert not closed.is_set()
        return ids

    ahead = host_table.ReadAhead(SimpleNamespace(gather=gather, close=closed.set))
    ahead.read_ahead(np.array([0]))
    assert entered.wait(timeout=5)
    with ThreadPoolExecutor(1) as pool:
        finish = pool.submit(ahead.close)
        try:
            assert not closed.wait(timeout=0.03)
        finally:
            release.set()
        finish.result(timeout=5)
    assert closed.is_set() and not ahead._ahead
    ahead.close()


@pytest.mark.parametrize("dtype", ["U32", "I32"])
def test_affine_signed_and_unsigned_words_preserve_high_bits(tmp_path, dtype):
    path = tmp_path / "packed.safetensors"
    words = np.tile(np.array([0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x7FFFFFFF], dtype=np.uint32), (4, 1))
    words[1] ^= np.uint32(0xAAAAAAAA)
    scales = np.array([[0x3F80], [0x4000], [0xBF00], [0x0000]], dtype=np.uint16)
    biases = np.array([[0x8000], [0x3F00], [0xBF80], [0x4000]], dtype=np.uint16)
    stored = words.view(np.int32) if dtype == "I32" else words
    header = _write(path, {"emb.weight": (dtype, stored), "emb.scales": ("BF16", scales),
                           "emb.biases": ("BF16", biases)})
    files = [(path, *(header[f"emb.{part}"] for part in ("weight", "scales", "biases")))]
    mapped, disk = host_table.HostTable(files), ssd_table.SSDTable(files)
    try:
        ids = [3, 0, 3, 1, 2]
        _same(disk.gather(ids), (words[ids], scales[ids], biases[ids]))
        _same(disk.gather(ids), mapped.gather(ids))
        assert disk.gather(ids)[0].dtype == np.uint32
    finally:
        disk.close()
        mapped._pool.shutdown(wait=True)


@pytest.mark.parametrize("dtype", ["I8", "I16", "I64", "F32", "U8", "U64"])
def test_affine_packed_words_refuse_other_dtypes(tmp_path, dtype):
    path = tmp_path / "packed.safetensors"
    header = _write(path, {"weight": (dtype, np.zeros((2, 4), dtype=np.uint32)),
                           "scales": ("BF16", np.zeros((2, 1), dtype=np.uint16)),
                           "biases": ("BF16", np.zeros((2, 1), dtype=np.uint16))})
    with pytest.raises(ValueError, match="weight must be a U32 or I32"):
        ssd_table.SSDTable([(path, header["weight"], header["scales"], header["biases"])])


def test_nvfp4_inline_threshold_uses_unique_ids_and_preserves_bytes(tmp_path, monkeypatch):
    shards, _ = _checkpoint(tmp_path, "nvfp4")
    mapped = open_table(tmp_path, shards, lambda _: 0.0371)
    disk = open_table(tmp_path, shards, lambda _: 0.0371, ssd=True)
    submissions, original = [], disk._pool.map

    def recorded(function, jobs):
        submissions.append(len(jobs))
        return original(function, jobs)

    monkeypatch.setattr(disk._pool, "map", recorded)
    try:
        assert disk._inline_rows == 16
        for ids, parallel in ((np.arange(16), False), (np.arange(17), True),
                              (np.tile(np.arange(16), 4), False), (np.tile(np.arange(17), 4), True)):
            submissions.clear()
            _same(disk.gather(ids), mapped.gather(ids))
            assert bool(submissions) is parallel
    finally:
        disk.close()
        mapped._pool.shutdown(wait=True)


@pytest.mark.parametrize("layout", ["mlx", "fp8", "bf16"])
def test_nvfp4_inline_policy_does_not_change_other_formats(tmp_path, monkeypatch, layout):
    shards, _ = _checkpoint(tmp_path, layout)
    disk = open_table(tmp_path, shards, lambda _: 1.0, ssd=True)
    submissions, original = [], disk._pool.map

    def recorded(function, jobs):
        submissions.append(len(jobs))
        return original(function, jobs)

    monkeypatch.setattr(disk._pool, "map", recorded)
    try:
        assert disk._inline_rows == 0
        disk.gather(np.arange(16))
        assert submissions
    finally:
        disk.close()

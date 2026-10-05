"""The CUDA spill tier on the CPU: files round-trip bit-exactly, cut or foreign files are dropped, ranks agree."""

from __future__ import annotations

import dataclasses
import json
import os
import threading

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import spill, spill_format, spill_io  # noqa: E402


@dataclasses.dataclass
class Snap:
    ids: list
    rec: object
    conv: object
    pending: object
    mtp_len: int
    drafter_end: int
    rows: list | None = None
    nbytes: int = 0
    tail: list | None = None
    head: object = None


def _cfg(tmp_path, **kw):
    kw.setdefault("gib", 1.0)
    kw.setdefault("min_tokens", 8)
    kw.setdefault("min_free_gib", 0.0)
    kw.setdefault("direct", False)
    kw.setdefault("writers", 2)
    return spill.SpillConfig(root=str(tmp_path), **kw)


def _store(tmp_path, classes=None, **kw):
    return spill.SpillStore(_cfg(tmp_path, **kw), device="cpu", signature="test", model_id="m", quiet=True,
                            classes=(Snap,) if classes is None else classes)


def _snap(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Snap(list(range(n)), torch.randn(3, 5, generator=g), torch.randn(2, 4, generator=g).to(torch.bfloat16),
                torch.arange(6, dtype=torch.int32), 7, n,
                rows=[torch.randn(n, 8, generator=g), torch.randn(n // 4, 2, generator=g)],
                tail=[(torch.arange(3), torch.randn(2, 3, generator=g))], head=None)


def test_encode_decode_round_trip():
    s = _snap(32)
    items, layer = spill.encode(s, skip=("ids", "nbytes"))
    tensors = dict(items)
    back = spill.decode(json.loads(json.dumps(layer)), tensors, (Snap,))
    assert type(back) is Snap
    assert back.mtp_len == 7 and back.drafter_end == 32 and back.head is None
    assert torch.equal(back.rec, s.rec) and torch.equal(back.conv, s.conv)
    assert isinstance(back.tail[0], tuple) and torch.equal(back.tail[0][1], s.tail[0][1])
    assert back.nbytes == 0                      # skipped: the dataclass default
    assert not hasattr(back, "ids") or back.ids is None or isinstance(back.ids, list)


def test_transient_and_materialize():
    class Thing:
        transient = ("live",)

        def __init__(self):
            self.a = torch.ones(2)
            self.live = object()                 # not storable: left out by ``transient``

    seen = []
    items, layer = spill.encode(Thing(), materialize=lambda o: seen.append(o))
    assert seen and [n for n, _ in items] == ["0.a"] and "live" not in layer["fields"]


def _write_read(tmp_path, **kw):
    st = _store(tmp_path, **kw)
    s = _snap(64, seed=1)
    job = st.save(s.ids, s, skip=("ids", "nbytes"))
    assert job is not None
    job.done.wait(10)
    assert job.ok and st.stats.written == 1
    obj, meta = st.load(spill.ids_key(s.ids))
    assert torch.equal(obj.rec, s.rec) and torch.equal(obj.conv, s.conv) and int(meta["n"]) == 64
    assert all(torch.equal(a, b) for a, b in zip(obj.rows, s.rows)) and torch.equal(obj.tail[0][1], s.tail[0][1])
    return st, s


def test_write_read_buffered(tmp_path):
    _write_read(tmp_path)


def test_write_read_small_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(spill_io, "CHUNK", 4096)    # many staging buffers a file: the packing across buffers
    monkeypatch.setattr(spill_format, "BLOCK", 1024)    # and CRC blocks that span them
    monkeypatch.setattr(spill_io, "_PINNED", {})
    _write_read(tmp_path)


def test_restart_scan_and_size_check(tmp_path):
    _, s = _write_read(tmp_path)
    again = _store(tmp_path)
    assert again.has(s.ids)
    path = os.path.join(again.dir, again.index[spill.ids_key(s.ids)].name + ".safetensors")
    with open(path, "r+b") as f:                 # a cut file: dropped at the next start
        f.truncate(os.path.getsize(path) - 10)
    with open(path + ".9.partial", "wb") as f:     # a write cut short
        f.write(b"x")
    third = _store(tmp_path)
    assert not third.has(s.ids) and not os.path.exists(path) and not os.path.exists(path + ".9.partial")


def test_other_builds_pruned(tmp_path):
    _write_read(tmp_path)
    other = spill.SpillStore(_cfg(tmp_path, keep_builds=0), device="cpu", signature="another", model_id="m",
                             quiet=True)
    assert [d for d in os.listdir(tmp_path) if d.startswith("spill-")] == [os.path.basename(other.top)]


def test_floor_and_cap(tmp_path):
    st = _store(tmp_path, min_free_gib=1e9)      # no disk would be left: nothing written
    s = _snap(64)
    assert st.save(s.ids, s, skip=("ids", "nbytes")) is None and st.stats.no_space == 1
    small = _store(tmp_path / "b", gib=1e-9)     # a cap below one entry: no room
    assert small.save(s.ids, s, skip=("ids", "nbytes")) is None and small.stats.full == 1


def test_lru_prunes_oldest(tmp_path):
    st = _store(tmp_path)
    jobs = []
    for i in range(3):
        s = _snap(64, seed=i)
        s.ids = [i * 1000 + t for t in range(64)]
        jobs.append((s, st.save(s.ids, s, skip=("ids", "nbytes"))))
    for _, j in jobs:
        j.done.wait(10)
    one = max(e.size for e in st.index.values())
    st.cap = 2 * one + one // 2                  # room for two
    s = _snap(64, seed=9)
    s.ids = list(range(5000, 5064))
    st.save(s.ids, s, skip=("ids", "nbytes")).done.wait(10)
    assert not st.has(jobs[0][0].ids) and st.has(jobs[2][0].ids) and st.has(s.ids)


class _Two:
    """Two ranks as threads of one process: ``share`` hands rank 0's list to both, ``gather`` every rank's."""

    def __init__(self):
        self.box: dict = {}
        self.bar = threading.Barrier(2)

    def comm(self, rank):
        def share(v):
            if rank == 0:
                self.box["s"] = v
            self.bar.wait()
            out = self.box["s"]
            self.bar.wait()
            return out

        def gather(v):
            self.box[rank] = v
            self.bar.wait()
            out = [self.box[0], self.box[1]]
            self.bar.wait()
            return out

        return share, gather

    def run(self, fn):
        out: dict = {}
        err: list = []

        def go(r):
            try:
                out[r] = fn(r)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                err.append(exc)
                self.bar.abort()

        ts = [threading.Thread(target=go, args=(r,)) for r in (0, 1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        if err:
            raise err[0]
        return out[0], out[1]


def test_two_ranks_agree_and_reconcile(tmp_path):
    two = _Two()

    def open_store(r):
        share, gather = two.comm(r)
        return spill.SpillStore(_cfg(tmp_path), rank=r, world=2, device="cpu", signature="test", model_id="m",
                                share=share, gather=gather, quiet=True)

    a, b = two.run(open_store)
    assert a.compat == b.compat and a.dir != b.dir
    both, only_a = _snap(64, 1), _snap(64, 2)
    only_a.ids = list(range(100, 164))

    def save(r):
        st = (a, b)[r]
        j = st.save(both.ids, both, skip=("ids", "nbytes"))
        j.done.wait(10)
        j2 = st.save(only_a.ids, only_a, skip=("ids", "nbytes"))
        j2.done.wait(10)
        return st

    two.run(save)
    os.remove(os.path.join(b.dir, b.index[spill.ids_key(only_a.ids)].name + ".safetensors"))   # rank 1 lost one
    a2, b2 = two.run(open_store)
    assert a2.has(both.ids) and b2.has(both.ids)
    assert not a2.has(only_a.ids) and not b2.has(only_a.ids)
    assert [e.key for e in sorted(a2.index.values(), key=lambda e: e.last)] == \
        [e.key for e in sorted(b2.index.values(), key=lambda e: e.last)]
    assert {e.key: e.size for e in a2.index.values()} == {e.key: e.size for e in b2.index.values()}


def test_a_file_cut_after_start_is_refused_at_load(tmp_path):
    st, s = _write_read(tmp_path)
    path = os.path.join(st.dir, st.index[spill.ids_key(s.ids)].name + ".safetensors")
    with open(path, "r+b") as f:                 # the data loses its last bytes while the server runs
        f.truncate(os.path.getsize(path) - 1)
    with pytest.raises(EOFError):
        st.load(spill.ids_key(s.ids))
    with open(path, "ab") as f:                  # and a longer file is refused too
        f.write(b"\0" * 2)
    with pytest.raises(EOFError):
        st.load(spill.ids_key(s.ids))


def test_a_file_cut_inside_its_header_is_dropped_at_start(tmp_path):
    st, s = _write_read(tmp_path)
    path = os.path.join(st.dir, st.index[spill.ids_key(s.ids)].name + ".safetensors")
    with open(path, "r+b") as f:
        f.truncate(100)
    again = _store(tmp_path)
    assert not again.has(s.ids) and not os.path.exists(path)


def test_build_id_names_the_weights_path_version_and_sources(tmp_path, monkeypatch):
    def model(name):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
        return d

    a, b = model("rev-a"), model("rev-b")
    first = spill_format.build_id(a)
    assert first == spill_format.build_id(a)
    assert str(a.resolve()) in first and "tensorfold=" in first and "sources=" in first
    assert spill_format.build_id(b) != first                 # another snapshot of the weights: another directory
    import tensorfold
    monkeypatch.setattr(tensorfold, "__version__", "0.0.0-test")
    assert spill_format.build_id(a) != first                         # another TensorFold: nothing reused


def test_weights_id_follows_this_ranks_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"a" * 8)
    first = spill_format.weights_id(tmp_path)
    assert first == spill_format.weights_id(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"b" * 9)     # rewritten in place, same name
    assert spill_format.weights_id(tmp_path) != first


def test_stored_files_and_folders_are_private(tmp_path):
    st, s = _write_read(tmp_path)
    path = os.path.join(st.dir, st.index[spill.ids_key(s.ids)].name + ".safetensors")
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(st.dir).st_mode & 0o777 == 0o700 and os.stat(st.top).st_mode & 0o777 == 0o700


def test_a_rank_with_another_build_is_named(tmp_path, capsys):

    def gather(words):                             # rank 1 runs another build: its first four words differ
        return [words, [w + 1 for w in words[:4]] + words[4:]] if len(words) == 8 else [words, words]
    spill.SpillStore(_cfg(tmp_path), device="cpu", signature="test", model_id="m", world=2, gather=gather,
                     share=lambda v: v)
    assert "rank(s) [1] run another build" in capsys.readouterr().out


def test_a_flipped_data_bit_is_refused_at_load(tmp_path):
    st, s = _write_read(tmp_path)
    path = os.path.join(st.dir, st.index[spill.ids_key(s.ids)].name + ".safetensors")
    with open(path, "rb") as f:
        _, data_at = spill._read_header_file(f)
    with open(path, "r+b") as f:                 # one bit in the data; the size and the header untouched
        f.seek(data_at + 100)
        b = f.read(1)
        f.seek(data_at + 100)
        f.write(bytes([b[0] ^ 0x10]))
    with pytest.raises(ValueError, match="CRC-32"):
        st.load(spill.ids_key(s.ids))
    assert st.stats.crc_failed == 1


def test_the_header_holds_one_crc_a_block(tmp_path, monkeypatch):
    import zlib
    monkeypatch.setattr(spill_format, "BLOCK", 1024)
    st, s = _write_read(tmp_path)
    path = os.path.join(st.dir, st.index[spill.ids_key(s.ids)].name + ".safetensors")
    with open(path, "rb") as f:
        header, data_at = spill._read_header_file(f)
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        f.seek(data_at)
        data = f.read(end)
    crcs = [int(c, 16) for c in header["__metadata__"]["crc32"].split(",")]
    assert crcs == [zlib.crc32(data[i:i + 1024]) for i in range(0, end, 1024)]


def test_a_file_naming_another_class_is_refused_and_nothing_is_imported(tmp_path):
    with pytest.raises(ValueError, match="not a class"):
        spill.decode({"class": "tensorfold_test_absent:Thing", "fields": {}}, {}, (Snap,))
    _, s = _write_read(tmp_path)                 # read back as the Snap its store names
    other = _store(tmp_path, classes=(dict,))    # an engine that reads another class: the read fails
    with pytest.raises(ValueError, match="not a class"):
        other.load(spill.ids_key(s.ids))
    assert other.stats.loaded == 0


def test_metrics_carry_every_spill_count_under_the_servers_names(tmp_path):
    from types import SimpleNamespace

    from tensorfold.server import metrics

    st = _store(tmp_path)
    st.stats.crc_failed, st.stats.copy_wait_s, st.stats.restore_ms = 2, 1.5, [700.0, 780.0, 823.0]
    info = st.info()                             # /health's "spill"
    body = metrics.render(SimpleNamespace(engine=SimpleNamespace(spill=st)))
    kinds = dict(line.split()[2:4] for line in body.splitlines() if line.startswith(f"# TYPE {metrics.PREFIX}spill_"))
    counts = {k for k, v in info.items() if not isinstance(v, bool) and not k.endswith("_gib")}
    counts -= {"highwater", "writers"}           # settings, and GiB that repeat the bytes
    assert {key for key, *_ in metrics._SPILL} == counts and len(kinds) == len(counts)
    for name, kind in kinds.items():             # counters end in _total, times are seconds
        assert (kind == "counter") == name.endswith("_total") and "_ms" not in name and not name.endswith("_s")
    lines = body.splitlines()
    assert kinds[f"{metrics.PREFIX}spill_crc_failed_total"] == "counter"
    assert f"{metrics.PREFIX}spill_crc_failed_total 2" in lines
    assert f"{metrics.PREFIX}spill_copy_wait_seconds_total 1.5" in lines
    assert kinds[f"{metrics.PREFIX}spill_restore_p50_seconds"] == "gauge"
    assert f"{metrics.PREFIX}spill_restore_p50_seconds 0.78" in lines
    assert f"{metrics.PREFIX}spill_restore_p95_seconds 0.823" in lines


def test_the_shutdown_flush_writes_from_the_engine_the_app_serves_then():
    import functools
    from types import SimpleNamespace

    from tensorfold.cuda.turns import Turns

    flushed, turns = [], Turns()
    app = SimpleNamespace(engine=SimpleNamespace(flush_spill=lambda b: flushed.append(("old", b))),
                          _turns=lambda: turns)
    hook = functools.partial(spill.flush_on_exit, app, 5.0)
    app.engine = SimpleNamespace(flush_spill=lambda b: flushed.append(("new", b)))     # the server swapped engines
    hook()
    assert flushed == [("new", 5.0)] and not turns.busy


SKIP = ("ids", "nbytes")


def _slow(monkeypatch, key, rank=None):
    """Writes of prompt ``key`` (on ``rank``, or every rank) wait for the returned gate."""

    gate = threading.Event()
    real = spill.SpillStore._write

    def write(self, job, *a):
        if job.key == key and rank in (None, self.rank):
            gate.wait(20)
        return real(self, job, *a)

    monkeypatch.setattr(spill.SpillStore, "_write", write)
    return gate


def _three():
    snaps = [_snap(64, seed=i) for i in range(3)]
    for i, s in enumerate(snaps):
        s.ids = [i * 1000 + t for t in range(64)]
    return snaps


def test_the_cap_drops_by_the_index_even_while_a_write_runs(tmp_path, monkeypatch):
    first, second, third = _three()
    gate = _slow(monkeypatch, spill.ids_key(first.ids))
    st = _store(tmp_path)
    j1 = st.save(first.ids, first, skip=SKIP)
    j2 = st.save(second.ids, second, skip=SKIP)
    j2.done.wait(10)
    st.cap = 2 * j2.entry.size + j2.entry.size // 2                 # room for two
    j3 = st.save(third.ids, third, skip=SKIP)
    assert not st.has(first.ids) and st.has(second.ids) and st.has(third.ids)    # the oldest went, written or not
    gate.set()
    for j in (j1, j3):
        j.done.wait(10)
    assert not os.path.exists(j1.path) and os.path.exists(j3.path)               # its writer left no file
    assert not [n for n in os.listdir(st.dir) if n.endswith(".partial")]


def test_two_ranks_drop_the_same_entry_whatever_their_writers(tmp_path, monkeypatch):
    first, second, third = _three()
    gate = _slow(monkeypatch, spill.ids_key(first.ids), rank=0)       # rank 0 still writes it, rank 1 is done
    two = _Two()

    def run(r):
        share, gather = two.comm(r)
        st = spill.SpillStore(_cfg(tmp_path), rank=r, world=2, device="cpu", signature="test", model_id="m",
                              share=share, gather=gather, quiet=True, classes=(Snap,))
        j1 = st.save(first.ids, first, skip=SKIP)
        j2 = st.save(second.ids, second, skip=SKIP)
        j2.done.wait(10)
        if r == 1:
            j1.done.wait(10)
        st.cap = 2 * j2.entry.size + j2.entry.size // 2
        st.save(third.ids, third, skip=SKIP)
        return sorted(st.index)

    a, b = two.run(run)
    gate.set()
    assert a == b and spill.ids_key(first.ids) not in a


def test_a_rank_that_cannot_use_its_folder_stops_every_rank(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    two = _Two()

    def run(r):
        share, gather = two.comm(r)
        try:
            spill.SpillStore(_cfg(tmp_path if r == 0 else blocked), rank=r, world=2, device="cpu", signature="test",
                             model_id="m", share=share, gather=gather, quiet=True)
        except RuntimeError as exc:
            return str(exc)
        return "opened"

    a, b = two.run(run)
    assert "rank(s) [1]" in a and "rank(s) [1]" in b


def test_writes_queued_past_the_inflight_cap_wait_for_the_oldest(tmp_path, monkeypatch):
    first, second, _ = _three()
    gate = _slow(monkeypatch, spill.ids_key(first.ids))
    st = _store(tmp_path)
    j1 = st.save(first.ids, first, skip=SKIP)
    monkeypatch.setattr(spill, "INFLIGHT", j1.nbytes)               # room for one queued write
    done = threading.Event()
    threading.Thread(target=lambda: (st.save(second.ids, second, skip=SKIP), done.set())).start()
    assert not done.wait(0.5)                                      # the second waits for the first
    gate.set()
    assert done.wait(10) and j1.ok


def test_a_state_larger_than_the_cap_evicts_nothing(tmp_path):
    first, _, _ = _three()
    st = _store(tmp_path)
    st.save(first.ids, first, skip=SKIP).done.wait(10)
    big = _snap(256, seed=5)
    big.ids = list(range(9000, 9256))
    st.cap = st._total() + 1                          # room for what is stored, not for the larger state
    assert st.save(big.ids, big, skip=SKIP) is None and st.stats.full == 1
    assert st.has(first.ids)                          # nothing was evicted for a save that could never fit


def test_a_save_that_cannot_fit_beside_kept_copies_evicts_nothing(tmp_path):
    first, second, _ = _three()
    st = _store(tmp_path)
    for s in (first, second):
        st.save(s.ids, s, skip=SKIP).done.wait(10)
    big = _snap(256, seed=6)
    big.ids = list(range(7000, 7256))
    st.cap = st._total() + 4096                       # the larger state fits only if ``second`` goes too
    keep = frozenset({spill.ids_key(second.ids)})
    assert st.save(big.ids, big, skip=SKIP, keep=keep) is None and st.stats.full == 1
    assert st.has(first.ids) and st.has(second.ids)   # nothing went for a save that could not be made


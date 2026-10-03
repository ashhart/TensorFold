# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
# Modified for TensorFold dsv41-cuda: plain-class nodes (GroupedLinear), a storage-span guard, aligned pinned staging
# (O_DIRECT for real), prune before write with a free-space check, this engine's key, pointer-table rebind and counter
# reset after a read (weights.load), no timeline (engine.py prints one [boot] line).
"""Prepared per-rank folders: rank R's built weight tree (every tensor's storage bytes, dtype, shape, stride and
offset; the dataclasses around them) in one folder, read back with parallel O_DIRECT readers::

    $TF_DSV41_PREPARED/<model>/<key hash>/rank<R>/{manifest.json,data.bin}

A start without one reads the checkpoint, slices this rank's share and builds the tree (``weights._build``); with
one, the same bytes come back without slicing or conversion. The key covers the checkpoint (file names, sizes and
modification times; config.json's hash), the rank and world, torch's version, the GPU, the DSpark blocks, the knobs
that change the tree's bytes and the source of the code that builds it, so any change misses the old folder instead
of reading stale weights. A folder whose manifest does not match, or that fails to read, is ignored (the checkpoint
path runs). ``python -m tensorfold.families.deepseek_v41.cuda.fastboot prepare`` writes a rank's folder (the
deploy's ``make prepare``); ``check`` says whether it is current.

``data.bin`` is read in 64 MiB chunks by several threads with O_DIRECT (no page cache: on GB10 the page cache is the
GPU's memory) into page-aligned pinned buffers, each chunk copied to the device on its thread's stream. Each chunk has
a SHA-256 in the manifest: ``TF_DSV41_PREPARED_VERIFY=full`` checks all, ``sample`` (default) every 16th and the
last, ``off`` none; a mismatch fails the read (and the checkpoint path runs).

No torch at import: the key, manifest and folder logic run in host-only tests.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

FORMAT = 1
CHUNK = 64 << 20                 # reader / checksum chunk; every chunk offset is a multiple of it
ALIGN = 4096                     # storage offsets and the file size (O_DIRECT)
SAMPLE_EVERY = 16
ENV = "TF_DSV41_PREPARED"
# the code that decides the tree's bytes: whole files, or module:qualname where a file holds much else
CODE = ["tensorfold.families.deepseek_v41.cuda.weights", "tensorfold.families.deepseek_v41.cuda.reader",
        "tensorfold.families.deepseek_v41.split", "tensorfold.families.deepseek_v41.config",
        "tensorfold.cuda.exl3.format", "tensorfold.cuda.direct_read",
        "tensorfold.cuda.exl3.experts:prepare", "tensorfold.cuda.exl3.experts:Exl3RoutedExperts",
        "tensorfold.cuda.exl3.experts:k2_of", "tensorfold.cuda.exl3.experts:codebook_id",
        "tensorfold.cuda.exl3.linear:strips", "tensorfold.cuda.exl3.linear:plan",
        "tensorfold.cuda.exl3.linear:Exl3Linear", "tensorfold.cuda.exl3.linear:GroupedLinear",
        "tensorfold.families.deepseek_v41.cuda.fastboot:FORMAT"]
KNOBS = ("TF_FOLD_SHARED", "TF_GROUPED_WO_A")


# -- keys --------------------------------------------------------------------------------------------------------
def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def source_digest(names: list[str]) -> str:
    """SHA-256 over the source of ``module`` or ``module:qualname`` objects."""

    h = hashlib.sha256()
    for name in names:
        mod_name, _, qual = name.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            h.update(name.encode() + b"\0missing\0")
            continue
        obj: Any = mod
        for part in filter(None, qual.split(".")):
            obj = getattr(obj, part)
        if not qual:
            src = Path(mod.__file__).read_text()
        elif inspect.isclass(obj) or inspect.isfunction(obj):
            src = inspect.getsource(obj)
        else:
            src = repr(obj)
        h.update(name.encode() + b"\0" + src.encode() + b"\0")
    return h.hexdigest()


def files_ident(model_dir: str | Path) -> list:
    d = Path(model_dir)
    files = [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in sorted(d.glob("*.safetensors"))]
    cfg = d / "config.json"
    return [files, hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.exists() else ""]


def weights_key(model_dir: str | Path, rank: int, world: int, *, draft: bool, device: str = "cuda") -> dict:
    import torch

    gpu = torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() else "cpu"
    return {"format": FORMAT, "what": "weights", "model": Path(model_dir).name or "model",
            "files": files_ident(model_dir), "rank": int(rank), "world": int(world), "torch": torch.__version__,
            "gpu": gpu, "draft": bool(draft), "knobs": {k: os.environ.get(k, "") for k in KNOBS},
            "code": source_digest(CODE)}


def prepared_root(env: dict | None = None) -> Path | None:
    v = (os.environ if env is None else env).get(ENV, "").strip()
    return None if v.lower() in ("", "0", "off", "no", "false") else Path(v)


def folder(root: Path, key: dict) -> Path:
    return Path(root) / key["model"] / digest(key)[:16] / f"rank{key['rank']}"


def valid(d: Path, key: dict) -> tuple[bool, str]:
    man = d / "manifest.json"
    if not man.is_file():
        return False, "no folder"
    try:
        m = json.loads(man.read_text())
    except (OSError, ValueError) as exc:
        return False, f"unreadable manifest ({exc})"
    if m.get("format") != FORMAT:
        return False, f"format {m.get('format')} != {FORMAT}"
    if m.get("key_hash") != digest(key):
        diff = sorted(k for k in set(key) | set(m.get("key", {})) if key.get(k) != m.get("key", {}).get(k))
        return False, f"key differs ({', '.join(diff) or 'hash'})"
    data = d / "data.bin"
    if not data.is_file() or data.stat().st_size != m.get("file_size"):
        return False, "data file missing or truncated"
    return True, "ok"


# -- flatten / rebuild -------------------------------------------------------------------------------------------
class _Flat:
    def __init__(self) -> None:
        self.storages: list = []
        self.index: dict = {}
        self.used: list[int] = []         # bytes of each storage some tensor reaches (the span guard)

    def storage(self, t) -> int:
        st = t.untyped_storage()
        k = (t.device.type, t.device.index, st.data_ptr(), st.nbytes())
        if k not in self.index:
            self.index[k] = len(self.storages)
            self.storages.append((st, st.nbytes(), t.device.type))
            self.used.append(0)
        i = self.index[k]
        if t.numel():
            end = t.storage_offset() + sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
            self.used[i] = max(self.used[i], end * t.element_size())
        return i

    def enc(self, x: Any) -> Any:
        import dataclasses

        import torch

        if x is None or isinstance(x, (bool, int, float, str)):
            return {"P": x}
        if isinstance(x, torch.dtype):
            return {"Y": str(x).removeprefix("torch.")}
        if isinstance(x, torch.Tensor):
            return {"T": self.storage(x), "dtype": str(x.dtype).removeprefix("torch."), "shape": list(x.shape),
                    "stride": list(x.stride()), "off": int(x.storage_offset()), "dev": x.device.type}
        if isinstance(x, list):
            return {"L": [self.enc(v) for v in x]}
        if isinstance(x, tuple):
            return {"U": [self.enc(v) for v in x]}
        if isinstance(x, dict):
            if not all(isinstance(k, (str, int)) for k in x):
                raise TypeError("dict keys must be str or int")
            return {"D": [[k, self.enc(v)] for k, v in x.items()]}
        cls = type(x)
        if not cls.__module__.startswith("tensorfold.") or isinstance(x, type) or not hasattr(x, "__dict__"):
            raise TypeError(f"cannot store a {cls.__module__}.{cls.__qualname__} in a prepared folder")
        fields = vars(x) if not dataclasses.is_dataclass(x) else {f.name: getattr(x, f.name)
                                                                  for f in dataclasses.fields(x)} | vars(x)
        return {"O": f"{cls.__module__}:{cls.__qualname__}",
                "f": {k: self.enc(v) for k, v in fields.items() if not k.startswith("_")}}


def _find_class(path: str):
    mod, _, qual = path.partition(":")
    if not mod.startswith("tensorfold."):
        raise ValueError(f"refusing to build {path}")
    obj: Any = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _dec(x: dict, bufs: list) -> Any:
    import torch

    if "P" in x:
        return x["P"]
    if "Y" in x:
        return getattr(torch, x["Y"])
    if "T" in x:
        buf = bufs[x["T"]]
        t = torch.empty(0, dtype=getattr(torch, x["dtype"]), device=buf.device)
        t.set_(buf.untyped_storage(), x["off"], x["shape"], x["stride"])
        return t
    if "L" in x:
        return [_dec(v, bufs) for v in x["L"]]
    if "U" in x:
        return tuple(_dec(v, bufs) for v in x["U"])
    if "D" in x:
        return {k: _dec(v, bufs) for k, v in x["D"]}
    if "O" in x:
        cls = _find_class(x["O"])
        obj = cls.__new__(cls)
        for k, v in x["f"].items():
            object.__setattr__(obj, k, _dec(v, bufs))
        return obj
    raise ValueError(f"bad node {list(x)[:3]}")


# -- write -------------------------------------------------------------------------------------------------------
def _drop_cache(fd: int, offset: int = 0, length: int = 0) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def _pinned(n: int):
    """A page-aligned pinned uint8 buffer of n bytes (pinned blocks need not be page-aligned: over-allocate)."""

    import torch

    raw = torch.empty(n + ALIGN, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
    lead = -raw.data_ptr() % ALIGN
    return raw[lead:lead + n]


def save(obj: Any, d: Path, key: dict) -> dict:
    """``obj``'s tensors (storage bytes) and structure into ``d`` (written beside it, then renamed)."""

    import torch

    flat = _Flat()
    tree = flat.enc(obj)
    for (_, nbytes, _), used in zip(flat.storages, flat.used):
        if nbytes > used + ALIGN:            # a view into a bigger buffer would store the whole buffer
            raise ValueError(f"a storage of {nbytes} bytes of which tensors reach {used}: refusing to store it")
    d = Path(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{d.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    storages, offset = [], 0
    for _, nbytes, dev in flat.storages:
        storages.append({"offset": offset, "nbytes": nbytes, "dev": dev})
        offset += -(-nbytes // ALIGN) * ALIGN
    free = shutil.disk_usage(d.parent).free
    if free < offset + (2 << 30):
        shutil.rmtree(tmp, ignore_errors=True)
        raise OSError(f"{d.parent}: {free / 1e9:.1f} GB free, the folder needs {offset / 1e9:.1f} GB + 2 GiB")
    stage = _pinned(CHUNK)
    stage_np = stage.numpy()
    sums: list[str] = []
    h, filled = hashlib.sha256(), 0
    t0 = time.perf_counter()

    def hashed(mv) -> None:
        nonlocal h, filled
        pos = 0
        while pos < len(mv):
            n = min(len(mv) - pos, CHUNK - filled)
            h.update(mv[pos:pos + n])
            filled, pos = filled + n, pos + n
            if filled == CHUNK:
                sums.append(h.hexdigest())
                h, filled = hashlib.sha256(), 0

    zeros = bytes(ALIGN)
    with open(tmp / "data.bin", "wb", buffering=0) as f:
        fd, since_sync = f.fileno(), 0
        for st, nbytes, _ in flat.storages:
            src = torch.empty(0, dtype=torch.uint8, device=st.device)
            src.set_(st, 0, (nbytes,), (1,))
            pos = 0
            while pos < nbytes:
                n = min(CHUNK, nbytes - pos)
                stage[:n].copy_(src[pos:pos + n])
                mv = memoryview(stage_np)[:n]
                f.write(mv)
                hashed(mv)
                pos, since_sync = pos + n, since_sync + n
                if since_sync >= 1 << 30:                 # keep the page cache (GPU memory on GB10) small
                    os.fdatasync(fd)
                    _drop_cache(fd)
                    since_sync = 0
            pad = -nbytes % ALIGN
            if pad:
                f.write(zeros[:pad])
                hashed(memoryview(zeros)[:pad])
        os.fdatasync(fd)
        _drop_cache(fd)
    if filled:
        sums.append(h.hexdigest())
    del stage, stage_np
    _release_pinned()
    manifest = {"format": FORMAT, "key": key, "key_hash": digest(key), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "file_size": offset, "chunk": CHUNK, "chunk_sha256": sums, "storages": storages, "tree": tree}
    (tmp / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    print(f"[tensorfold] prepared folder written: {d} ({offset / 1e9:.1f} GB in {time.perf_counter() - t0:.1f}s)",
          flush=True)
    return manifest


# -- read --------------------------------------------------------------------------------------------------------
def _release_pinned() -> None:
    """Hand the staging buffers' pinned memory back (on GB10 it is GPU memory)."""

    import torch

    try:
        torch._C._host_emptyCache()
    except (AttributeError, RuntimeError):
        pass


def _plan(storages: list[dict], size: int) -> list[list[tuple[int, int, int, int]]]:
    """Per chunk k: (storage, byte offset in it, offset in the chunk, bytes)."""

    plan: list[list[tuple[int, int, int, int]]] = [[] for _ in range(-(-size // CHUNK))]
    for i, s in enumerate(storages):
        a, n, pos = s["offset"], s["nbytes"], 0
        while pos < n:
            k, within = divmod(a + pos, CHUNK)
            m = min(n - pos, CHUNK - within)
            plan[k].append((i, pos, within, m))
            pos += m
    return plan


def verify_mode() -> str:
    v = os.environ.get(f"{ENV}_VERIFY", "sample").strip().lower() or "sample"
    if v not in ("full", "sample", "off"):
        raise ValueError(f"{ENV}_VERIFY={v!r}: expected full, sample or off")
    return v


def _open(path: Path) -> tuple[int, bool]:
    if hasattr(os, "O_DIRECT") and os.environ.get(f"{ENV}_DIRECT", "1").strip() != "0":
        try:
            return os.open(path, os.O_RDONLY | os.O_DIRECT), True
        except OSError:
            pass
    return os.open(path, os.O_RDONLY), False


def read(d: Path, device: str = "cuda", threads: int | None = None) -> tuple[Any, dict]:
    """The object ``save`` stored in ``d`` with its tensors on ``device``, and read stats."""

    import mmap

    import torch

    d = Path(d)
    m = json.loads((d / "manifest.json").read_text())
    storages, size, sums = m["storages"], int(m["file_size"]), m["chunk_sha256"]
    if int(m.get("chunk", CHUNK)) != CHUNK:
        raise ValueError(f"{d}: chunk {m.get('chunk')} != {CHUNK}")
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    t0 = time.perf_counter()
    bufs = [torch.empty(s["nbytes"], dtype=torch.uint8, device=dev if s["dev"] != "cpu" else "cpu") for s in storages]
    plan = _plan(storages, size)
    mode = verify_mode()
    n_threads = max(1, min(int(threads or os.environ.get(f"{ENV}_THREADS", "") or 8), max(1, len(plan))))
    nxt, lock, errors, verified, direct_all = [0], threading.Lock(), [], [0], [True]

    def worker() -> None:
        try:
            stream = torch.cuda.Stream(device=dev) if cuda else None
            if cuda:
                stage = [_pinned(CHUNK) for _ in range(2)]
            else:
                stage = [torch.frombuffer(mmap.mmap(-1, CHUNK), dtype=torch.uint8) for _ in range(2)]
            fd, direct = _open(d / "data.bin")
            if direct and any(b.data_ptr() % ALIGN for b in stage):
                os.close(fd)
                fd, direct = os.open(d / "data.bin", os.O_RDONLY), False
            with lock:
                direct_all[0] &= direct
            events: list = [None, None]
            which = 0
            try:
                while True:
                    with lock:
                        k = nxt[0]
                        nxt[0] += 1
                    if k >= len(plan) or errors:
                        break
                    if events[which] is not None:
                        events[which].synchronize()
                    buf = stage[which]
                    mv = memoryview(buf.numpy())
                    off, want = k * CHUNK, min(CHUNK, size - k * CHUNK)
                    got = 0
                    while got < want:
                        r = os.preadv(fd, [mv[got:want]], off + got)
                        if r <= 0:
                            raise OSError(f"{d / 'data.bin'}: short read at {off + got}")
                        got += r
                    if not direct:
                        _drop_cache(fd, off, want)
                    if mode == "full" or (mode == "sample" and (k % SAMPLE_EVERY == 0 or k == len(plan) - 1)):
                        if hashlib.sha256(mv[:want]).hexdigest() != sums[k]:
                            raise ValueError(f"{d / 'data.bin'}: chunk {k} checksum mismatch")
                        with lock:
                            verified[0] += 1
                    if cuda:
                        with torch.cuda.stream(stream):
                            for i, a, s, n in plan[k]:
                                bufs[i][a:a + n].copy_(buf[s:s + n], non_blocking=bufs[i].is_cuda)
                            ev = torch.cuda.Event()
                            ev.record(stream)
                        events[which] = ev
                        which ^= 1
                    else:
                        for i, a, s, n in plan[k]:
                            bufs[i][a:a + n].copy_(buf[s:s + n])
                for ev in events:
                    if ev is not None:
                        ev.synchronize()
            finally:
                os.close(fd)
        except BaseException as exc:  # noqa: BLE001 - reported by the caller
            errors.append(exc)

    pool = [threading.Thread(target=worker, name=f"tf-dsv41-prepared-{i}", daemon=True) for i in range(n_threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if cuda:
        torch.cuda.synchronize(dev)
        _release_pinned()
    if errors:
        raise errors[0]
    obj = _dec(m["tree"], bufs)
    dt = time.perf_counter() - t0
    return obj, {"bytes": size, "seconds": dt, "gbps": size / 1e9 / max(dt, 1e-9), "direct": direct_all[0],
                 "verified": verified[0], "chunks": len(plan), "threads": n_threads}


def _kind(d: Path) -> tuple:
    try:
        k = json.loads((d / "manifest.json").read_text()).get("key", {})
    except (OSError, ValueError):
        return (None, None)
    return (k.get("what"), k.get("draft"))


def prune(d: Path, keep: int = 0) -> None:
    """Remove this rank's other key folders of the same model and kind beyond the ``keep`` newest (~95 GB each)."""

    kind = _kind(d) if (d / "manifest.json").exists() else None
    olds = sorted((p for p in d.parent.parent.glob(f"*/{d.name}") if p != d and (p / "manifest.json").exists()
                   and (kind is None or _kind(p)[0] == kind[0])),
                  key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)
    for p in olds[keep:]:
        shutil.rmtree(p, ignore_errors=True)
        try:
            p.parent.rmdir()
        except OSError:
            pass
        print(f"[tensorfold] removed an older prepared folder: {p}", flush=True)


def cached_tree(what: str, key: dict, build: Callable[[], Any], device: str = "cuda") -> Any:
    """``build()``'s result, from the prepared folder for ``key`` when a valid one exists; else built."""

    root = prepared_root()
    if root is None:
        return build()
    d = folder(root, key)
    ok, why = valid(d, key)
    if ok:
        try:
            obj, st = read(d, device)
            print(f"[tensorfold] {what}: prepared folder, {st['bytes'] / 1e9:.1f} GB in {st['seconds']:.1f}s "
                  f"({st['gbps']:.1f} GB/s, {st['threads']} threads, {'O_DIRECT' if st['direct'] else 'buffered'}, "
                  f"{st['verified']}/{st['chunks']} chunks verified)", flush=True)
            if not st["direct"]:
                print(f"[tensorfold] {what}: the prepared folder was read through the page cache (O_DIRECT refused)",
                      file=sys.stderr, flush=True)
            return obj
        except Exception as exc:  # noqa: BLE001 - any failure: build as without a folder
            print(f"[tensorfold] {what}: prepared folder {d} unusable ({type(exc).__name__}: {exc}); building from "
                  "the checkpoint", file=sys.stderr, flush=True)
    else:
        print(f"[tensorfold] {what}: no usable prepared folder ({why}): building from the checkpoint "
              "(make prepare writes one)", flush=True)
    return build()


# -- prepare ahead of a start ------------------------------------------------------------------------------------
def prepare(model_dir: Path, rank: int, world: int, root: Path, *, force: bool = False, draft: bool = True) -> int:
    """Build this rank's tree from the checkpoint and write its folder (on the node that serves the rank)."""

    from . import weights as W

    key = weights_key(model_dir, rank, world, draft=draft)
    d = folder(root, key)
    ok, why = valid(d, key)
    if ok and not force:
        print(f"[tensorfold] prepared folder current: {d}", flush=True)
        return 0
    prune(d)                                    # the old folder goes first: two would not fit beside the model
    t0 = time.perf_counter()
    tree = W.build(model_dir, rank=rank, draft=draft)
    print(f"[tensorfold] rank {rank}: built in {time.perf_counter() - t0:.0f}s ({why})", flush=True)
    save(W.storable(tree), d, key)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash prepared folders")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("prepare", "write this rank's prepared folder"),
                       ("check", "whether this rank's folder is current (exit 0) or why not (exit 1)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("model_dir")
        p.add_argument("--rank", type=int, required=True)
        p.add_argument("--world", type=int, default=2)
        p.add_argument("--root", default=os.environ.get(ENV, ""))
        p.add_argument("--no-draft", action="store_true")
        if name == "prepare":
            p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.root:
        raise SystemExit(f"--root or {ENV} names the prepared folders' root")
    if a.cmd == "prepare":
        return prepare(Path(a.model_dir), a.rank, a.world, Path(a.root), force=a.force, draft=not a.no_draft)
    key = weights_key(a.model_dir, a.rank, a.world, draft=not a.no_draft)
    ok, why = valid(folder(Path(a.root), key), key)
    print(f"{folder(Path(a.root), key)}: {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

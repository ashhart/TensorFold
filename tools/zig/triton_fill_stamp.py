"""triton_fill's stamps (ptxas's DWARF file entries), the stamped bwrap re-exec, and nvdisasm SASS."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


# ---- stamps: ptxas's DWARF file entries -------------------------------------------------------------------------

def elf_sections(b: bytes) -> dict:
    """{name: bytes} of an ELF64 little-endian file."""

    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    stro = hdrs[shstrndx][4]
    out = {}
    for h in hdrs:
        name = b[stro + h[0]:b.index(b"\0", stro + h[0])].decode()
        out[name] = b[h[4]:h[4] + h[5]]
    return out


def uleb(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        r |= (x & 0x7F) << s
        s += 7
        i += 1
        if x < 0x80:
            return r, i


def line_files(sec: bytes) -> dict:
    """{path: (mtime, size)} of a DWARF 2-4 .debug_line section's file tables."""

    out, off = {}, 0
    while off + 10 < len(sec):
        unit = struct.unpack_from("<I", sec, off)[0]
        ver = struct.unpack_from("<H", sec, off + 4)[0]
        p = off + 10
        p += 1 + (1 if ver >= 4 else 0) + 3
        opbase = sec[p]
        p += 1 + opbase - 1
        dirs = []
        while sec[p] != 0:
            e = sec.index(b"\0", p)
            dirs.append(sec[p:e].decode())
            p = e + 1
        p += 1
        while sec[p] != 0:
            e = sec.index(b"\0", p)
            name = sec[p:e].decode()
            d, p = uleb(sec, e + 1)
            mt, p = uleb(sec, p)
            sz, p = uleb(sec, p)
            path = name if name.startswith("/") or d == 0 else f"{dirs[d - 1]}/{name}"
            if mt or sz:
                out[path] = (mt, sz)
        off += 4 + unit
    return out


def stamps_of_cubins(dirs: list[Path]) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.cubin")):
            secs = elf_sections(f.read_bytes())
            for name in (".nv.merc.debug_line", ".debug_line"):
                if name in secs:
                    for path, st in line_files(secs[name]).items():
                        if out.setdefault(path, st) != st:
                            raise SystemExit(f"{path}: stamps {out[path]} and {st} in one set")
    return out


def stamped_root(stamps: dict, py: Path, served_py: str, served_triton: str) -> Path:
    """A scratch tree holding each stamped served path: the local file it maps to, with the image's mtime."""

    import triton

    pairs = [(served_py, str(py)), (served_triton, os.path.dirname(triton.__file__))]
    root = Path(tempfile.mkdtemp(prefix="triton-fill-root-"))
    for path, (mt, sz) in stamps.items():
        src = next((b + path[len(a):] for a, b in pairs if path.startswith(a)), None)
        if src is None or not os.path.exists(src):
            raise SystemExit(f"stamp {path}: no local file maps to it")
        if os.path.getsize(src) != sz:
            raise SystemExit(f"stamp {path}: local {src} is {os.path.getsize(src)} bytes, the image's {sz}: not the "
                             "served source")
        dst = root / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.utime(dst, (mt, mt))
    return root


def in_bwrap(root: Path) -> list[str]:
    """bwrap arguments: the host's top-level entries, then the stamped tree's over them (new tops created)."""

    args = ["bwrap", "--die-with-parent", "--dev", "/dev", "--proc", "/proc"]
    for e in sorted(os.listdir("/")):
        p = "/" + e
        if e in ("dev", "proc"):
            continue
        if os.path.islink(p):
            args += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            args += ["--bind", p, p]
    files = [Path(d) / f for top in sorted(os.listdir(root)) for d, _, fs in os.walk(root / top) for f in fs]
    # a served path whose directory the host lacks: bwrap gets a tmpfs on its deepest existing ancestor (hiding it)
    tmp = set()
    for full in files:
        parts = Path("/" + str(full.relative_to(root))).parts
        for i in range(2, len(parts)):
            if not os.path.isdir(os.path.join(*parts[:i])):
                if i > 2:
                    tmp.add(os.path.join(*parts[:i - 1]))
                break
    for t in sorted(tmp):
        print(f"[fill] sandbox: tmpfs over {t}", file=sys.stderr)
        args += ["--tmpfs", t]
    for full in files:
        args += ["--ro-bind", str(full), "/" + str(full.relative_to(root))]
    return args


def reexec_stamped(a) -> None:
    """Runs this command again inside bwrap with the stamped files at the served paths (once)."""

    if os.environ.get("TRITON_FILL_STAMPED") or not a.stamps:
        return
    stamps = {k: tuple(v) for k, v in json.loads(Path(a.stamps).read_text()).items()}
    import triton  # noqa: F401  (the local Triton's path for the map)

    root = stamped_root(stamps, Path(a.py).resolve(), a.served_py, a.served_triton)
    env = dict(os.environ, TRITON_FILL_STAMPED="1", TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="triton-fill-cache-"))
    cmd = in_bwrap(root) + ["--", sys.executable, *sys.argv]
    try:
        raise SystemExit(subprocess.run(cmd, env=env).returncode)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---- SASS -------------------------------------------------------------------------------------------------------

def sass(cubin: bytes) -> str:
    tool = os.environ.get("TRITON_NVDISASM_PATH")
    if not tool:
        import triton

        tool = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/nvdisasm")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        r = subprocess.run([tool, "-c", f.name], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvdisasm: {r.stderr.strip()}")
    return r.stdout

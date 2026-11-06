"""cuda_sass_identity.py BASE HEAD: cuobjdump -sass per function, base against head, of every image a change reaches."""
import hashlib
import os
import re
import subprocess
from pathlib import Path

SMS = (80, 86, 89, 120, 121)
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
CUOBJDUMP = os.environ.get("CUOBJDUMP", str(Path(NVCC).with_name("cuobjdump")))
GENERATED = ("qwen4_gdn", "qwen4_ple")  # regenerated from the production family owners before comparison
CONTROLS = (
    ("fp4pair constant", "common/fp_pairs.cuh", r'"r"\(0x7E807E80u\)', '"r"(0x7E817E80u)'),
    ("warp_sum order", "common/reduce.cuh", r"int o = 16; o > 0; o >>= 1\)", "int o = 1; o < 32; o <<= 1)"),
)

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def git(repo, *args):
    """Run git in the repository and return its stdout."""
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout


def export(repo, rev, dest):
    """The revision's zig/kernels and build files under dest."""
    dest.mkdir(parents=True)
    tar = git(repo, "archive", rev, "zig/kernels", "zig/build")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)


def catalog(tree):
    """Image name -> (unit, flags, arch_specific, min_sm): the catalog, then images other build files name (h3, ...)."""
    text = (tree / "zig/build/cuda_kernels.zig").read_text()
    consts = {m[1]: re.findall(r'"([^"]+)"', m[2]) for m in re.finditer(r"const (\w+) = &\[_\]\[\]const u8\{([^}]*)\}", text)}
    torch = re.findall(r'"([^"]+)"', re.search(r"torch_flags = \[_\]\[\]const u8\{([^}]*)\}", text)[1])
    images = {}
    others = sorted(p for p in (tree / "zig/build").glob("*.zig") if p.name != "cuda_kernels.zig")
    for f in [tree / "zig/build/cuda_kernels.zig", *others]:
        owned = dict(consts)
        owned.update({m[1]: re.findall(r'"([^"]+)"', m[2]) for m in re.finditer(r"const (\w+) = &\[_\]\[\]const u8\{([^}]*)\}", f.read_text())})
        for m in re.finditer(r'\.\{ \.name = "(\w+)",((?:[^{}]|\{[^{}]*\})*)\}', f.read_text()):
            name, rest = m[1], m[2]
            fl = re.search(r"\.flags = (&\.\{[^}]*\}|\w+)", rest)
            src = re.search(r'\.src = "([^"]+)"', rest)
            unit = src[1] if src else name
            if not fl or name in images or not (tree / "zig/kernels/cuda" / f"{unit}.cu").is_file():
                continue
            if fl[1] not in owned and not fl[1].startswith("&."):
                raise ValueError(f"unparsed CUDA compiler flags for {name}: {fl[1]}")
            flags = owned[fl[1]] if fl[1] in owned else re.findall(r'"([^"]+)"', fl[1])
            ms = re.search(r"\.min_sm = (\d+)", rest)
            images[name] = (unit, flags, ".arch_specific = true" in rest, int(ms[1]) if ms else 0)
    return torch, images


def includes(tree, unit):
    """Every file under zig/kernels the unit reaches through quoted includes (conditionals ignored: a superset)."""
    root = tree / "zig/kernels/cuda"
    seen, todo = set(), [root / f"{unit}.cu"]
    while todo:
        f = todo.pop().resolve()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for inc in INCLUDE.findall(f.read_text(errors="replace")):
            todo += [f.parent / inc, root / inc]
    return {p.relative_to(tree.resolve()).as_posix() for p in seen}


def build(tree, torch, img, out):
    """nvcc -fatbin as zig/build/cuda_fatbin.zig runs it, then cuobjdump -sass; returns the dump or an error line."""
    unit, flags, arch, min_sm = img
    gens = [f"-gencode=arch=compute_{s}{'a' if arch and s >= 120 else ''},code=sm_{s}{'a' if arch and s >= 120 else ''}"
            for s in SMS if s >= min_sm]
    if not gens:
        return ""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["nice", "-n", "19", NVCC, "-fatbin", *torch, *flags, f"-I{tree}/zig/kernels/cuda", *gens, "-o", str(out), f"{tree}/zig/kernels/cuda/{unit}.cu"]
    out.with_suffix(".command.json").write_text(__import__("json").dumps(cmd, indent=2) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out.with_suffix(".nvcc.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        return "BUILD FAIL " + r.stderr[-600:]
    dump = subprocess.run([CUOBJDUMP, "-sass", str(out)], capture_output=True, text=True, check=True).stdout
    out.with_suffix(".sass").write_text(dump)
    return dump


def functions(dump):
    """(arch, function) -> each instruction's line (any address width) and continuation (second word), collapsed."""
    res, cur, arch = {}, None, None
    for line in dump.splitlines():
        m = re.match(r"\s*arch = (sm_\w+)", line)
        if m:
            arch = m[1]
            continue
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = (arch, m[1])
            res[cur] = []
            continue
        if cur and re.match(r"\s*(/\*[0-9a-f]{4,}\*/|/\* 0x)", line, re.I):
            res[cur].append(re.sub(r"\s+", " ", line.strip()))
    return res


def digest(lines):
    """sha256 of a function's instruction lines."""
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def compare(fb, fh):
    """Changed, added and removed (arch, function) keys."""
    return [k for k in fb if k in fh and fb[k] != fh[k]], [k for k in fh if k not in fb], [k for k in fb if k not in fh]


def recheck(first, again):
    """First builds against rebuilds: (keys that differ again, keys whose tree disagreed with itself)."""
    (fb, fh), (fb2, fh2) = first, again
    unstable = sorted({k for k in set(fb) | set(fb2) if fb.get(k) != fb2.get(k)} | {k for k in set(fh) | set(fh2) if fh.get(k) != fh2.get(k)})
    ch, add, rem = compare(fb, fh)
    ch2, add2, rem2 = compare(fb2, fh2)
    return sorted((set(ch) & set(ch2)) | (set(add) & set(add2)) | (set(rem) & set(rem2))), unstable


def main():
    from cuda_sass_compare import main as compare_main
    return compare_main()


if __name__ == "__main__":
    raise SystemExit(main())

"""cuda_sass_identity.py BASE HEAD: cuobjdump -sass per function, base against head, of every image a change reaches."""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SMS = (80, 86, 89, 120, 121)
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
CUOBJDUMP = os.environ.get("CUOBJDUMP", str(Path(NVCC).with_name("cuobjdump")))
GENERATED = ("qwen4_gdn", "qwen4_ple")  # the build compiles a generated text; the checked-in unit stands in for it
CONTROLS = (
    ("fp4pair constant", "common/qmm_frag.cuh", r'"r"\(0x7E807E80u\)', '"r"(0x7E817E80u)'),
    ("warp_sum order", "common/reduce.cuh", r"int o = 16; o > 0; o >>= 1\)", "int o = 1; o < 32; o <<= 1)"),
)
USAGE = """BASE and HEAD are git revisions of this repository (a merge commit works). Every catalog image
(zig/build/cuda_kernels.zig) whose unit includes, directly or through other headers, a file under zig/kernels changed
between them is built from both trees with that tree's catalog flags, torch's nvcc flags and -I zig/kernels/cuda, for
sm_80/86/89/120/121 as the build passes them (min_sm, the "a" targets of arch-specific units), and compared function
by function. Two planted controls then prove the comparison can go red: a changed fp4pair constant
(common/qmm_frag.cuh) and a reversed warp_sum butterfly (common/reduce.cuh), each built against the head. Exit status
0 only when every function matches and both controls change functions."""
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
        for m in re.finditer(r'\.\{ \.name = "(\w+)",((?:[^{}]|\{[^{}]*\})*)\}', f.read_text()):
            name, rest = m[1], m[2]
            fl = re.search(r"\.flags = (&\.\{[^}]*\}|\w+)", rest)
            src = re.search(r'\.src = "([^"]+)"', rest)
            unit = src[1] if src else name
            if not fl or name in images or not (tree / "zig/kernels/cuda" / f"{unit}.cu").is_file():
                continue
            flags = consts[fl[1]] if fl[1] in consts else re.findall(r'"([^"]+)"', fl[1])
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
    cmd = [NVCC, "-fatbin", *torch, *flags, f"-I{tree}/zig/kernels/cuda", *gens, "-o", str(out), f"{tree}/zig/kernels/cuda/{unit}.cu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return "BUILD FAIL " + r.stderr[-600:]
    dump = subprocess.run([CUOBJDUMP, "-sass", str(out)], capture_output=True, text=True, check=True).stdout
    out.unlink()
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
        if cur and re.match(r"\s*(/\*[0-9a-f]{4,}\*/|/\* 0x)", line):
            res[cur].append(re.sub(r"\s+", " ", line.strip()))
    return res


def digest(lines):
    """sha256 of a function's instruction lines."""
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def compare(fb, fh):
    """Changed, added and removed (arch, function) keys."""
    return [k for k in fb if k in fh and fb[k] != fh[k]], [k for k in fh if k not in fb], [k for k in fb if k not in fh]


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE)
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--receipt", help="write image, arch, function and both SASS digests here")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("-j", type=int, default=4, help="parallel nvcc runs")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    repo = git(a.repo, "rev-parse", "--show-toplevel").decode().strip()
    revs = [git(repo, "rev-parse", r + "^{commit}").decode().strip() for r in (a.base, a.head)]
    names = git(repo, "diff", "--name-only", "--no-renames", revs[0], revs[1], "--", "zig/kernels").decode().split()
    changed = set(names)
    nvcc = subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"base {revs[0][:12]} head {revs[1][:12]}; {len(changed)} files changed under zig/kernels; {nvcc}")
    with tempfile.TemporaryDirectory(prefix="sassid-") as tmp:
        tmp = Path(tmp)
        trees = {"base": tmp / "base", "head": tmp / "head"}
        for tag, rev in zip(trees, revs):
            export(repo, rev, trees[tag])
        cats = {tag: catalog(t) for tag, t in trees.items()}
        both = sorted(set(cats["base"][1]) & set(cats["head"][1]))
        for gone in sorted(set(cats["base"][1]) ^ set(cats["head"][1])):
            print(f"{gone}: in one catalog only")
        reach = {tag: {n: includes(trees[tag], cats[tag][1][n][0]) for n in both} for tag in trees}
        imgs = [n for n in both if reach["base"][n] & changed or reach["head"][n] & changed or cats["base"][1][n] != cats["head"][1][n]]
        jobs = [(tag, n) for n in imgs for tag in trees]
        with cf.ThreadPoolExecutor(a.j) as ex:
            dumps = dict(zip(jobs, ex.map(lambda j: build(trees[j[0]], cats[j[0]][0], cats[j[0]][1][j[1]], tmp / "o" / j[0] / f"{j[1]}.fatbin"), jobs)))
        bad, total, heads, rows = 0, 0, {}, []
        for n in imgs:
            b, h = dumps[("base", n)], dumps[("head", n)]
            if b.startswith("BUILD FAIL") or h.startswith("BUILD FAIL"):
                bad += 1
                print(f"{n}: {b if b.startswith('BUILD FAIL') else h}")
                continue
            fb, fh = functions(b), functions(h)
            heads[n] = fh
            ch, add, rem = compare(fb, fh)
            total += len(fh)
            bad += len(ch) + len(add) + len(rem)
            note = " (checked-in unit; the build generates this one)" if n in GENERATED else ""
            print(f"{n}: {len(fh)} functions over {','.join(sorted({k[0] for k in fh}))}; changed {len(ch)} added {len(add)} removed {len(rem)}{note}")
            for k in (ch + add + rem)[:5]:
                print("   ", *k)
            rows += [(n, *k, digest(fb[k]) if k in fb else "-", digest(fh[k]) if k in fh else "-") for k in sorted(set(fb) | set(fh))]
        print(f"images {len(imgs)} functions {total} differing {bad}")
        if a.receipt:
            with open(a.receipt, "w") as f:
                f.write(f"# cuda_sass_identity base {revs[0]} head {revs[1]}; {nvcc}\n# image\tarch\tfunction\tbase sha256\thead sha256\n")
                f.writelines("\t".join(r) + "\n" for r in rows)
        red = True
        if not a.no_controls:
            for label, rel, pattern, repl in CONTROLS:
                ctl = tmp / "ctl"
                subprocess.run(["rm", "-rf", str(ctl)], check=True)
                export(repo, revs[1], ctl)
                f = ctl / "zig/kernels/cuda" / rel
                text, k = re.subn(pattern, repl, f.read_text(), count=1)
                hit = [n for n in imgs if f"zig/kernels/cuda/{rel}" in reach["head"][n] and n in heads]
                if not k or not hit:
                    print(f"control {label}: could not plant (pattern found {k}, images {len(hit)})")
                    red = False
                    continue
                f.write_text(text)
                with cf.ThreadPoolExecutor(a.j) as ex:
                    cd = dict(zip(hit, ex.map(lambda n: build(ctl, cats["head"][0], cats["head"][1][n], tmp / "o" / "ctl" / f"{n}.fatbin"), hit)))
                moved = {n: len(compare(heads[n], functions(cd[n]))[0]) for n in hit if not cd[n].startswith("BUILD FAIL")}
                fails = [n for n in hit if cd[n].startswith("BUILD FAIL")]
                print(f"control {label}: {sum(moved.values())} functions changed in {sum(1 for v in moved.values() if v)} of {len(hit)} images"
                      + (f", build failed: {' '.join(fails)}" if fails else ""))
                red = red and sum(moved.values()) > 0 and not fails
    return 0 if bad == 0 and red else 1


if __name__ == "__main__":
    sys.exit(main())

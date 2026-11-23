#!/usr/bin/env python3
"""Drama 3 codec before/after check on a synthetic codec: same bytes, speed, refusals; with its planted control.

No checkpoint needed: `make` writes random fp32 weights under codec_schema's names and sizes (1.62 GB), 600 frames of
codes (19 chunks: a first chunk of 4, then 32 each, past the 512-row ring) and a 3 s clip.

    drama3_codec_bytes.py make DATA
    drama3_codec_bytes.py run BIN DATA OUT [--times N]   # decode 600 frames + encode the clip, md5s and timings
    drama3_codec_bytes.py compare OUT_A OUT_B            # exit 0 when the waveform and the codes are the same bytes
    drama3_codec_bytes.py speed OUT_A OUT_B [OUT_A2 ...] # median decode/encode ms per binary (A-B-B-A rounds)
    drama3_codec_bytes.py plant ROOT --backend cuda|metal  # the control: layernorm eps 1e-6 -> 2e-6 in ROOT
    drama3_codec_bytes.py refusals ROOT [REV]            # each backend's codec errors and asserts at REV

The control: build ROOT after `plant`, `run` it, and `compare` against the unplanted run must exit 1 (the decode
moves). Encode codes are the weaker check: they did not move under the plant on these weights.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import struct
import subprocess
import sys
from pathlib import Path

FRAMES = 600
FIRST_CHUNK = 4
SEED = 20261104
D = 1024


def schema():
    """Tensor name -> float count, as zig/src/families/drama3/codec_schema.zig prepares them."""
    names = {}

    def tf(p, layers):
        for i in range(layers):
            for f, n in (("attn_norm", D), ("wqkv", 3 * D * D), ("wo", D * D), ("ffn_norm", D),
                         ("w13", 6 * D * D), ("w2", 3 * D * D), ("g_attn", D), ("g_ffn", D)):
                names[f"{p}.{i}.{f}"] = n
        names[f"{p}.norm"] = D

    names.update({"vq.table": 13 * D * D, "vq.cbn": 13 * D * 8, "vq.in_w": 10 * 8 * D, "vq.in_b": 80})
    tf("post", 8)
    tf("pre", 8)
    for s in ("up", "down"):
        for i in range(2):
            names[f"{s}.{i}.b"] = 2 * D if s == "up" else D
            names[f"{s}.{i}.{'wa' if s == 'up' else 'w'}"] = 2 * D * D
            for f, n in (("dw", 7 * D), ("pw1", 4 * D * D), ("pw2", 4 * D * D), ("pw1_b", 4 * D), ("dw_b", D),
                         ("ln_w", D), ("ln_b", D), ("pw2_b", D), ("gamma", D)):
                names[f"{s}.{i}.cn.{f}"] = n
    for dec in (True, False):
        p = "dec" if dec else "enc"
        names[p + ".conv0"] = 7 * 1536 * D if dec else 7 * 64
        names[p + ".conv0_b"] = 1536 if dec else 64
        names[p + ".final"] = 7 * 96 if dec else 3 * D * D
        names[p + ".final_b"] = 1 if dec else D
        names[p + ".final_alpha"] = 96 if dec else D
        strides = [8, 8, 4, 2] if dec else [2, 4, 8, 8]
        for bi in range(4):
            cin = (1536 >> bi) if dec else (64 << bi)
            cout = cin // 2 if dec else cin * 2
            names[f"{p}.{bi}.alpha"] = cin
            names[f"{p}.{bi}.b"] = strides[bi] * cout if dec else cout
            names[f"{p}.{bi}.wa"] = strides[bi] * cin * cout
            names[f"{p}.{bi}.wb"] = strides[bi] * cin * cout
            d = cout if dec else cin
            for r in range(3):
                for f, n in (("c7", 7 * d * d), ("c1", d * d), ("a1", d), ("a2", d), ("c7_b", d), ("c1_b", d)):
                    names[f"{p}.{bi}.ru{r}.{f}"] = n
        if not dec:
            tf("enc.3.tf", 4)
    return names


def values(rng, name, n):
    """Norm weights and snake alphas 1, layer scales 0.1, the rest uniform bytes with exponents of +-[0.0078, 2)."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("attn_norm", "ffn_norm", "norm", "ln_w", "a1", "a2") or "alpha" in leaf:
        return struct.pack("<f", 1.0) * n
    if leaf in ("g_attn", "g_ffn", "gamma"):
        return struct.pack("<f", 0.1) * n
    b = bytearray(rng.randbytes(4 * n))
    bias = leaf.endswith("_b") or leaf in ("b", "in_b")
    hi = (0x3A, 0xBA) if bias else (0x3C, 0xBC)
    b[3::4] = bytes(b[3::4]).translate(bytes(hi[0] if x < 128 else hi[1] for x in range(256)))
    return bytes(b)


def safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, bytes or a callable giving them)."""
    hdr, off = {}, 0
    for name, dtype, shape, data in tensors:
        n = 4 * math.prod(shape)
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for name, _, shape, data in tensors:
            raw = data() if callable(data) else data
            assert len(raw) == 4 * math.prod(shape), name
            f.write(raw)


def make(data):
    data.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tensors = [(k, "F32", [n], (lambda k=k, n=n: values(rng, k, n))) for k, n in schema().items()]
    safetensors(data / "codec.safetensors", tensors)
    rng = random.Random(7)
    with open(data / "frames.i32", "wb") as f:
        for _ in range(FRAMES):
            f.write(struct.pack("<11i", 0, rng.randrange(4096), *[rng.randrange(1024) for _ in range(9)]))
    n = 44100 * 3 + 1234
    pcm = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / 44100) + rng.randrange(-2000, 2000)))
                   for i in range(n))
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
    (data / "clip.wav").write_bytes(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + fmt + b"data" +
                                    struct.pack("<I", len(pcm)) + pcm)
    # a reference waveform makes --codec-check write its decode as codec_check.wav; its SNR is meaningless here
    ref = struct.pack("<f", 0.5) * (FRAMES * 2048)
    safetensors(data / "ref.safetensors", [("dec.audio", "F32", [FRAMES * 2048], ref)])
    sums = {p.name: md5(p) for p in sorted(data.iterdir()) if p.name != "inputs.json"}
    (data / "inputs.json").write_text(json.dumps(sums, indent=1) + "\n")
    print(json.dumps(sums, indent=1))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def run(binary, data, out, times):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = binary.resolve()
    data = data.resolve()
    check = [str(binary), "--codec-check", str(data / "ref.safetensors"), "--codec", str(data / "codec.safetensors"),
             "--frames", str(data / "frames.i32"), "--first-chunk", str(FIRST_CHUNK), "--ref-audio", str(data / "clip.wav")]
    decode_ms, encode_ms = [], []
    for i in range(times):
        p = subprocess.run(check, cwd=out, capture_output=True, text=True, timeout=1200)
        (out / f"check{i}.log").write_text(p.stdout + p.stderr)
        if "RESULT drama3 codec-check" not in p.stderr:
            sys.exit(f"{binary}: --codec-check did not finish (rc {p.returncode}), see {out}/check{i}.log")
        decode_ms.append(float(re.search(r"decode run 1: .* all in ([0-9.]+) ms", p.stderr).group(1)))
        encode_ms.append(float(re.search(r"encode: \d+ frames in ([0-9.]+) ms", p.stderr).group(1)))
        wav = md5(out / "codec_check.wav")
        if i == 0:
            first = wav
        elif wav != first:
            sys.exit(f"{binary}: the decode changed between runs of one binary ({first} then {wav})")
    enc = [str(binary), "--codec", str(data / "codec.safetensors"), "--encode", str(out / "codes.i32"),
           "--ref-audio", str(data / "clip.wav")]
    p = subprocess.run(enc, cwd=out, capture_output=True, text=True, timeout=1200)
    (out / "encode.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        sys.exit(f"{binary}: --encode failed (rc {p.returncode}), see {out}/encode.log")
    res = {"binary": str(binary), "binary_md5": md5(binary), "inputs": json.loads((data / "inputs.json").read_text()),
           "codec_check.wav": md5(out / "codec_check.wav"), "codes.i32": md5(out / "codes.i32"),
           "decode_ms": decode_ms, "encode_ms": encode_ms}
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("binary_md5", "codec_check.wav", "codes.i32", "decode_ms", "encode_ms")}))


def compare(a, b):
    ra, rb = (json.loads((p / "result.json").read_text()) for p in (a, b))
    if ra["inputs"] != rb["inputs"]:
        sys.exit("the two runs used different inputs")
    same = True
    for k in ("codec_check.wav", "codes.i32"):
        ok = ra[k] == rb[k]
        same &= ok
        print(f"{k}: {ra[k]} {'==' if ok else '!='} {rb[k]}")
    print("SAME" if same else "DIFF")
    return 0 if same else 1


def speed(outs):
    by = {}
    for p in outs:
        r = json.loads((p / "result.json").read_text())
        e = by.setdefault(r["binary_md5"], {"binary": r["binary"], "decode_ms": [], "encode_ms": []})
        e["decode_ms"] += r["decode_ms"]
        e["encode_ms"] += r["encode_ms"]
    for k, e in by.items():
        print(f"{k[:12]} {e['binary']}: decode median {statistics.median(e['decode_ms']):.1f} ms "
              f"(n={len(e['decode_ms'])}, min {min(e['decode_ms']):.1f}), encode median "
              f"{statistics.median(e['encode_ms']):.1f} ms (min {min(e['encode_ms']):.1f})")


PLANTS = {
    "cuda": ("zig/src/families/drama3/cuda/codec.zig", r"add\(@as\(f32, 1e-6\)\)", "add(@as(f32, 2e-6))"),
    "metal": ("zig/src/families/drama3/metal/codec.zig", r"\.eps = 1e-6", ".eps = 2e-6"),
}


def plant(root, backend):
    rel, pat, new = PLANTS[backend]
    path = root / rel
    text = path.read_text()
    n = len(re.findall(pat, text))
    if n != 1:
        sys.exit(f"{path}: expected one layernorm eps 1e-6, found {n}")
    path.write_text(re.sub(pat, new, text))
    print(f"planted: {rel} layernorm eps 1e-6 -> 2e-6")


SITE = re.compile(r"error\.\w+|std\.debug\.assert\(.*\)(?=;)|@panic\(.*\)(?=;)|unreachable")


def refusals(root, rev):
    """Each backend's codec refusals (error names, asserts, panics): the backend files plus the shared schedule."""
    fam = "zig/src/families/drama3/"
    out = {}
    for be in ("cuda", "metal"):
        sites = []
        for rel in (f"{be}/codec.zig", f"{be}/codec_encode.zig", "codec_net.zig"):
            p = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{fam}{rel}"], capture_output=True, text=True)
            if p.returncode == 0:
                sites += [m for line in p.stdout.splitlines() if not line.strip().startswith("//")
                          for m in SITE.findall(line)]
        out[be] = sorted(sites)
        print(f"{be}: " + " | ".join(out[be]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make").add_argument("data", type=Path)
    r = sub.add_parser("run")
    for k in ("binary", "data", "out"):
        r.add_argument(k, type=Path)
    r.add_argument("--times", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    sub.add_parser("speed").add_argument("outs", type=Path, nargs="+")
    p = sub.add_parser("plant")
    p.add_argument("root", type=Path)
    p.add_argument("--backend", choices=sorted(PLANTS), required=True)
    f = sub.add_parser("refusals")
    f.add_argument("root", type=Path)
    f.add_argument("rev", nargs="?", default="HEAD")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.data)
    elif a.cmd == "run":
        run(a.binary, a.data, a.out, a.times)
    elif a.cmd == "compare":
        return compare(a.a, a.b)
    elif a.cmd == "speed":
        speed(a.outs)
    elif a.cmd == "plant":
        plant(a.root, a.backend)
    else:
        refusals(a.root, a.rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())

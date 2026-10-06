"""Write Nemotron's decode and window kernels as mx.fast.metal_kernel builds them at the M5's call sites."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import py_kernels  # noqa: E402
from metal_source import PREAMBLE, Arg, kernel_text, strip_comments  # noqa: E402

ROOT = py_kernels.ROOT
OUT = ROOT / "zig/kernels/metal/nemotron"
NAMES = OUT / "kernels.zig"

# Nemotron 3.5 Lightning 30B-A3B
HIDDEN, EXPERT, SHARED, EXPERTS, TOP_K, GROUP = 2688, 1856, 3712, 128, 6, 64
HEADS, HEAD_DIM, NGROUPS, STATE, KC = 64, 64, 8, 128, 4
XD = HEADS * HEAD_DIM
CD = XD + 2 * NGROUPS * STATE
PROJ = XD + CD + HEADS
Q_HEADS, KV_HEADS, ATT_DIM, VOCAB = 32, 2, 128, 131072
QKV = (Q_HEADS + 2 * KV_HEADS) * ATT_DIM
NORM_THREADS = 896
# (key, N, K) of every lane-tiled 64-wide projection; out_proj and o_proj share a shape and so a kernel
DRAFT_IDS = Path(__file__).resolve().parents[2] / "zig/src/families/nemotron/draft_ids.txt"
DRAFT_VOCAB = len(DRAFT_IDS.read_text().split())  # the MTP head's draft vocabulary, as long as its list
PROJECTIONS = (("in", PROJ, HIDDEN), ("out", HIDDEN, XD), ("down", HIDDEN, SHARED), ("qkv", QKV, HIDDEN),
               ("head", VOCAB, HIDDEN), ("eh", HIDDEN, 2 * HIDDEN), ("draft", DRAFT_VOCAB, HIDDEN),
               ("up", SHARED, HIDDEN))
ROW_TILES = ((1, 0), (2, 0), (2, 1))      # (TMR, EDGE): 16 rows, 32-row blocks, a last block past MP
SPLIT_KEYS = ("in", "out", "eh")  # projections too narrow to fill the GPU: one threadgroup a K slice
PADDED_ROWS = tuple(range(16, 129, 16))   # MP values a window or prompt chunk of up to 128 rows pads to
ATTENTION_SG = tuple(range(1, 17))        # simdgroups a threadgroup: one variant each


@dataclass
class Spec:
    key: str
    name: str
    source: str
    inputs: list[Arg]
    outputs: list[Arg]
    header: str = ""
    template: list = field(default_factory=list)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _baked(key: str, base: str, body: str, consts: list, inputs: list[Arg], outputs: list[Arg], header: str) -> Spec:
    """lane_qmm._Baked: constants written into the source, the name hashed over header + source."""

    source = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in consts) + body
    return Spec(key, f"{base}_{_sha(header + source)}", source, inputs, outputs, header)


def _split(body: str) -> str:
    """lane_qmm's coop body with each K slice its own threadgroup, writing fp32 partials for tf_coop_combine."""

    whole = "const ushort slice = sg >> 1;"
    assert whole in body and "threadgroup float part[" in body
    body = body.replace(whole, "const ushort slice = threadgroup_position_in_grid.z;")
    body = "\n".join(line for line in body.split("\n") if "const ushort tip" not in line)
    cut = body.index("threadgroup float part[")
    return body[:body.rfind("\n", 0, cut) + 1] + (
        "  for (int i = 0; i < CAP; i++) {\n"
        "    const int m = rb + erow[i], n = n0 + ecol[i];\n"
        "    if (m < M) PART[((int64_t)slice * MP + m) * N + n] = C[i];\n"
        "  }\n")


def _chunk_guard(body: str) -> str:
    """The partial with threadgroups past the dims' chunk count returning first: they'd store over the next head's."""

    line = "const int L = dims[0], NCH = dims[1], NQ = dims[2], SGA = dims[4];"
    assert body.count(line) == 1
    i = body.index(line)
    end = body.index("\n", i)
    return body[:end + 1] + "  if (int(c) >= NCH) return;\n" + body[end + 1:]


def specs() -> list[Spec]:
    lq = py_kernels.load("kernels.qwen.dense.v1.lane_qmm")
    la = py_kernels.load("kernels.qwen.dense.v1.lane_attention")
    nk = py_kernels.load("kernels.nemotron.lightning.v1.kernels")
    lf = py_kernels.load("kernels.nemotron.lightning.v1.lane_fused")
    rows = py_kernels.load("kernels.nemotron.lightning.v1.rows")
    src = py_kernels.load("kernels.nemotron.lightning.v1.sources")
    threads = py_kernels.load("kernels.threads")

    bf, f32, u32, i32 = "bfloat16", "float32", "uint32", "int32"
    big = 64                      # any element count of 8 or more binds as device memory
    out: list[Spec] = []
    lane_in = [Arg("X", bf, big, 2), Arg("XS", f32, big, 2), Arg("Wq", u32, big, 2), Arg("SBt", bf, big, 3),
               Arg("mdims", i32, 8)]
    for k in (HIDDEN, XD, SHARED, 2 * HIDDEN):
        out.append(_baked(f"xsum_{k}", "lane_qmm_xsum", lq._XSUM, [("K", k), ("GS", GROUP)],
                          [Arg("X", bf, big, 2), Arg("mdims", i32, 8)], [Arg("XS", f32, big, 2)], lq._HEADER))
    for key, n, k in PROJECTIONS:
        for tmr, edge in ROW_TILES:
            consts = [("TMR", tmr), ("N", n), ("K", k), ("SK", lq.split_k(n, k)), ("GS", GROUP), ("EDGE", edge)]
            out.append(_baked(f"coop_{key}_{tmr}_{edge}", "lane_qmm_coop", lq._COOP, consts, lane_in,
                              [Arg("Y", bf, big, 2)], lq._HEADER))
            if key in SPLIT_KEYS:
                out.append(_baked(f"coop_{key}_{tmr}_{edge}_sk", "lane_qmm_coop_sk", _split(lq._COOP), consts,
                                  lane_in, [Arg("PART", f32, big, 2)], lq._HEADER))
    for tmr, edge in ROW_TILES:
        consts = [("TMR", tmr), ("N", SHARED), ("K", HIDDEN), ("SK", lq.split_k(SHARED, HIDDEN)), ("GS", GROUP),
                  ("EDGE", edge)]
        out.append(_baked(f"up_relu2_{tmr}_{edge}", "nemotron_lane_up_relu2", lf._COOP_RELU2_XS, consts, lane_in,
                          [Arg("Y", bf, big, 2), Arg("XSO", f32, big, 2)], lq._HEADER))
    for mp in PADDED_ROWS:
        out.append(_baked(f"group_norm_xs_{mp}", "nemotron_group_norm_xs", lf._GROUP_NORM_XS,
                          [("XD", XD), ("GS", XD // NGROUPS), ("MP", mp)],
                          [Arg("X", bf, big, 2), Arg("W", bf, big), Arg("eps", f32, 1), Arg("dims", i32, 8)],
                          [Arg("OUT", bf, big, 2), Arg("XS", f32, big, 2)], lq._HEADER))

    def templated(key: str, base: str, source: str, inputs: list[Arg], outputs: list[Arg], template: list) -> Spec:
        return Spec(key, f"{base}_{_sha(source)}", source, inputs, outputs, "", template)

    norm = nk._with_group_sums(src._ADD_NORM.replace("MIX", src._MIX_PLAIN))
    out.append(templated("add_norm_xs", "nemotron_add_norm_xs", norm,
                         [Arg("H", bf, big, 2), Arg("X", bf, big, 2), Arg("W", bf, big), Arg("eps", f32, 1),
                          Arg("dims", i32, 8)],
                         [Arg("HN", bf, big, 2), Arg("OUT", bf, big, 2), Arg("XS", f32, big, 2)],
                         [("D", HIDDEN), ("T", NORM_THREADS)]))
    moe = nk._with_group_sums(src._ADD_NORM.replace("MIX", src._MIX_MOE))
    out.append(templated("add_norm_moe_xs", "nemotron_add_norm_moe_xs", moe,
                         [Arg("H", bf, big, 2), Arg("Y", bf, big, 3), Arg("WE", f32, 8), Arg("SH", bf, big, 2),
                          Arg("W", bf, big), Arg("eps", f32, 1), Arg("dims", i32, 8)],
                         [Arg("HN", bf, big, 2), Arg("OUT", bf, big, 2), Arg("XS", f32, big, 2)],
                         [("D", HIDDEN), ("T", NORM_THREADS), ("E", TOP_K)]))
    out.append(templated("router", "nemotron_router", src._ROUTER,
                         [Arg("X", bf, big, 2), Arg("GW", bf, big, 2), Arg("rows", i32, 1)], [Arg("OUT", bf, big, 2)],
                         [("D", HIDDEN), ("NE", EXPERTS), ("SG", 8), ("MAXR", 16)]))
    out.append(templated("route", "nemotron_route", src._ROUTE,
                         [Arg("G", bf, big, 2), Arg("bias", f32, big), Arg("scaling", f32, 1)],
                         [Arg("IDX", u32, 8, 2), Arg("WT", f32, 8, 2)], [("NE", EXPERTS), ("K", TOP_K)]))
    table = [Arg(n, i32, 8) for n in ("SEG", "START", "SLOT", "STORE")]
    out.append(templated("mamba_conv", "nemotron_mamba_conv", src._MAMBA_CONV,
                         [Arg("P", bf, big, 2), Arg("CS_IN", bf, big, 3), Arg("CW", f32, big, 2), Arg("CB", f32, big),
                          *table],
                         [Arg("XBC", bf, big, 2), Arg("CS_OUT", bf, big, 3)],
                         [("XD", XD), ("NG", NGROUPS), ("DS", STATE), ("KC", KC), ("PROJ", PROJ), ("XOFF", XD)]))
    out.append(templated("mamba_scan", "nemotron_mamba_scan", src._MAMBA_SCAN,
                         [Arg("P", bf, big, 2), Arg("XBC", bf, big, 2), Arg("S_IN", f32, big, 4),
                          Arg("A_LOG", f32, big), Arg("DSKIP", f32, big), Arg("DT_BIAS", f32, big),
                          Arg("limits", f32, 2), Arg("dims", i32, 1), Arg("SEG", i32, 8), Arg("SLOT", i32, 8),
                          Arg("STORE", i32, 8)],
                         [Arg("Y", bf, big, 2), Arg("S_OUT", f32, big, 4)],
                         [("H", HEADS), ("DH", HEAD_DIM), ("NG", NGROUPS), ("DS", STATE), ("XD", XD), ("PROJ", PROJ),
                          ("DTOFF", XD + CD), ("SSZ", HEADS * HEAD_DIM * STATE)]))

    expert_in = [Arg("X", bf, big, 2), Arg("UIDS", u32, 8), Arg("START", i32, 8), Arg("COUNT", i32, 8),
                 Arg("MEMBERS", i32, 8), Arg("UCOUNT", i32, 1), Arg("W", u32, big, 3), Arg("S", bf, big, 3),
                 Arg("B", bf, big, 3)]
    for kind, body, k, n, extra in (("up", rows._EXPERT_UP, HIDDEN, EXPERT, [("TOPK", TOP_K)]),
                                    ("down", rows._EXPERT_DOWN, EXPERT, HIDDEN, [])):
        base = f"nemotron_rows_expert_{kind}"
        out.append(Spec(f"expert_{kind}", f"{base}_{_sha(rows._HEADER + body)}", body, expert_in,
                        [Arg("ACT" if kind == "up" else "Y", bf, big, 2)], rows._HEADER,
                        [("K", k), ("N", n), ("GS", GROUP), ("RPS", rows.RPS), ("SG", 2), *extra]))
    consts = (("NE", EXPERTS), ("K", TOP_K), ("T", rows.ROUTE_THREADS), ("MAXP", rows.MAX_GROUP_PAIRS))
    body = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + rows._ROUTE_GROUP
    header = rows._HEADER + threads.reserve(rows.ROUTE_THREADS)
    out.append(Spec("route_group", f"nemotron_rows_route_group_{EXPERTS}_{TOP_K}_{_sha(header + body)}", body,
                    [Arg("G", bf, big, 2), Arg("bias", f32, big), Arg("scaling", f32, 1), Arg("rows", i32, 1)],
                    [Arg("IDX", u32, big, 2), Arg("WT", f32, big, 2), Arg("UIDS", u32, big), Arg("START", i32, big),
                     Arg("COUNT", i32, big), Arg("MEMBERS", i32, big), Arg("UCOUNT", i32, 1)], header))

    gs = py_kernels.load("engine.gpu_sampling")
    for key, vocab, ids in (("sample", VOCAB, False), ("sample_ids", DRAFT_VOCAB, True)):
        source = f"  constexpr int V = {vocab};\n  constexpr int C = {gs.CANDIDATES};\n" + (gs._SOURCE_IDS if ids else gs._SOURCE)
        header = gs._HEADER + threads.reserve(1024)
        ins = [Arg("L", bf, big, 2), Arg("seeds", u32, 8), Arg("positions", u32, 8), Arg("cfg", f32, 8),
               Arg("kcap", u32, 8)] + ([Arg("IDS", u32, big)] if ids else [])
        out.append(Spec(key, f"tf_gpu_sample{'_ids' if ids else ''}_{_sha(header + source)}", source, ins,
                        [Arg("TOK", u32, 8)], header))

    partial = _chunk_guard(la._PARTIAL_DIRECT_128 if la.DIRECT_P else la._PARTIAL_128)
    pname = "partial_direct_128" if la.DIRECT_P else "partial_128"
    for sg in ATTENTION_SG:
        out.append(Spec(f"attn_partial_{sg}", f"lane_attention_{pname}_{_sha(la._HEADER + partial)}", partial,
                        [Arg("Qp", bf, big, 3), Arg("K", bf, big, 4), Arg("V", bf, big, 4), Arg("scale", f32, 1),
                         Arg("dims", i32, 5)],
                        [Arg("PO", f32, big), Arg("PM", f32, big), Arg("PL", f32, big)], la._HEADER,
                        [("G", Q_HEADS // KV_HEADS), ("D", ATT_DIM), ("SG", sg), ("CK", la.CHUNK), ("TK", la.TILE)]))
    out.append(Spec("attn_merge", f"lane_attention_merge_{_sha(la._HEADER + la._MERGE)}", la._MERGE,
                    [Arg("PO", f32, big), Arg("PM", f32, big), Arg("PL", f32, big), Arg("dims", i32, 5)],
                    [Arg("OUT", bf, big, 4)], la._HEADER, [("G", Q_HEADS // KV_HEADS), ("D", ATT_DIM)]))
    return out


def rendered(spec: Spec) -> tuple[str, str]:
    """(function name, file text) for a spec."""

    fname, text = kernel_text(spec.name, spec.inputs, spec.outputs, spec.source, spec.header, spec.template)
    return fname, "// Generated by tools/zig/gen_nemotron_kernels.py from our Python kernels; do not edit.\n" + PREAMBLE + strip_comments(text)


def names_zig(entries: list[tuple[str, str]]) -> str:
    lines = ["//! Generated by tools/zig/gen_nemotron_kernels.py: each kernel's function name and source; do not edit.",
             "", "pub const Kernel = struct { key: []const u8, function: [:0]const u8, source: []const u8 };", "",
             "pub const all = [_]Kernel{"]
    lines += [f'    .{{ .key = "{k}", .function = "{f}", .source = @embedFile("{k}.metal") }},' for k, f in entries]
    lines += ["};", ""]
    return "\n".join(lines)


def compare(entries: list[tuple[Spec, str, str]], directory: Path) -> int:
    """Each kernel text against the text after MLX's preamble in a compiled source of the same function."""

    found: dict[str, str] = {}
    for path in sorted(directory.glob("*.metal")):
        text = path.read_text(errors="replace")
        marker = text.rfind("/" * 79 + "\n")
        if marker >= 0:
            found[text[marker + 80:]] = path.name
    seen = mismatched = 0
    for spec, fname, text in entries:
        body = text[text.index(PREAMBLE) + len(PREAMBLE):]
        hits = [source for source in found if f"void {fname}(" in source]
        if not hits:
            continue
        seen += 1
        if all(strip_comments(source) != body for source in hits):
            mismatched += 1
            print(f"differs from MLX's: {spec.key} ({fname})")
    print(f"{seen} kernels found in {directory}, {mismatched} differ")
    return 1 if mismatched else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the written files are stale")
    ap.add_argument("--compare", type=Path, help="a directory of MLX-compiled kernel sources")
    args = ap.parse_args()
    entries = [(s, *rendered(s)) for s in specs()]
    if args.compare:
        return compare(entries, args.compare)
    files = {OUT / f"{s.key}.metal": text for s, _, text in entries}
    files[NAMES] = names_zig([(s.key, f) for s, f, _ in entries])
    stale = 0
    for path, text in files.items():
        if args.check:
            if not path.is_file() or path.read_text() != text:
                print(f"stale: {path.relative_to(ROOT)}")
                stale += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if args.check:
        extra = sorted(set(OUT.glob("*.metal")) - set(files))
        for path in extra:
            print(f"not generated: {path.relative_to(ROOT)}")
        return 1 if stale or extra else 0
    print(f"wrote {len(entries)} kernels to {OUT.relative_to(ROOT)} and {NAMES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

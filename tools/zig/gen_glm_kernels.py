"""Write GLM-5.3-Flash's decode kernels as mx.fast.metal_kernel builds them for the Python family's calls."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import py_kernels  # noqa: E402
from metal_source import PREAMBLE, Arg, _template_value, function_name, kernel_text, strip_comments  # noqa: E402

ROOT = py_kernels.ROOT
OUT = ROOT / "zig/kernels/metal/glm"
NAMES = OUT / "kernels.zig"

# GLM-5.3-Flash (TensorFold/GLM-5.3-Flash-MLX-4bit-MTP)
HIDDEN, VOCAB, EXPERTS, TOP_K, EXPERT, DENSE = 4096, 154880, 288, 8, 2048, 12288
KDA_HEADS, KDA_DIM, TAPS = 64, 128, 4
KDA_WIDTH = KDA_HEADS * KDA_DIM
KDA_PROJ = 3 * KDA_WIDTH + 2 * KDA_DIM + KDA_HEADS
MLA_HEADS, NOPE, RANK, Q_LORA, I_HEADS, I_DIM = 64, 256, 512, 1536, 32, 128
X_PROJ = Q_LORA + RANK + I_DIM + I_HEADS
QR_PROJ = MLA_HEADS * NOPE + I_HEADS * I_DIM
KDA_PROJ_TP = KDA_PROJ - KDA_WIDTH * 3 // 2 - KDA_HEADS // 2  # TP2: one Mac's heads' q, k, v and b; f_a and g_a whole
QR_PROJ_TP = QR_PROJ - MLA_HEADS // 2 * NOPE  # TP2: one Mac's heads' q_b; the indexer's queries whole
TOPK_KEYS = 2048 + 3                      # index_topk keys plus the always-selected tail of a partial block
MAX_ROWS = 16                             # the widest decode window
MAXU = MAX_ROWS * TOP_K                   # the shared expert's slot in the MoE kernels (never a routed one)
# (K, N) of every dense 4-bit projection a decode row runs through qmv_rows (KDA out = the MTP's eh_proj shape)
PROJECTIONS = ((HIDDEN, KDA_PROJ), (KDA_WIDTH, HIDDEN), (HIDDEN, X_PROJ), (Q_LORA, QR_PROJ),
               (MLA_HEADS * NOPE, HIDDEN), (HIDDEN, 2 * DENSE), (DENSE, HIDDEN), (HIDDEN, VOCAB), (HIDDEN, KDA_PROJ_TP),
               (HIDDEN, VOCAB // 2), (Q_LORA, QR_PROJ_TP), (HIDDEN, DENSE))
# TP2's row-split projections: one Mac's input columns, fp32 partials the pair sums in rank order
PARTIALS = ((KDA_WIDTH // 2, HIDDEN), (MLA_HEADS // 2 * NOPE, HIDDEN), (DENSE // 2, HIDDEN))
# MLX 0.32's one-row gemv / gemv_t tilings (kernels.gemv_params) at the shapes decode reaches
GEMV_T = {"igate": (1, 2, 8, 4, 4, 4), "values": (1, 4, 8, 4, 4, 4)}
GEMV = {"scores_lt4": (1, 8, 1, 32, 1, 4), "scores_le32": (1, 8, 1, 32, 4, 4), "scores": (4, 1, 1, 32, 4, 4)}


# bfloat abs/exp overloads for the family's kernels outside MLX: the float function in each math mode, rounded back to bfloat
BF16_MATH = """namespace metal {
inline bfloat16_t abs(bfloat16_t v) { return bfloat16_t(abs(float(v))); }
inline bfloat16_t exp(bfloat16_t v) { return bfloat16_t(exp(float(v))); }
namespace fast {
inline bfloat16_t abs(bfloat16_t v) { return bfloat16_t(metal::fast::abs(float(v))); }
inline bfloat16_t exp(bfloat16_t v) { return bfloat16_t(metal::fast::exp(float(v))); }
}
namespace precise {
inline bfloat16_t abs(bfloat16_t v) { return bfloat16_t(metal::precise::abs(float(v))); }
inline bfloat16_t exp(bfloat16_t v) { return bfloat16_t(metal::precise::exp(float(v))); }
}
}
"""


@dataclass
class Spec:
    key: str
    name: str
    source: str
    inputs: list[Arg]
    outputs: list[Arg]
    header: str = ""
    template: list = field(default_factory=list)
    extra: list = field(default_factory=list)   # more template instantiations of the same function


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def specs() -> list[Spec]:
    K = py_kernels.load("kernels.glm.flash.v1.kernels")
    F = py_kernels.load("kernels.glm.flash.v1.fused")
    H = py_kernels.load("kernels.glm.flash.v1.hc")
    KD = py_kernels.load("kernels.glm.flash.v1.kda")
    M = py_kernels.load("kernels.glm.flash.v1.moe")
    SA = py_kernels.load("kernels.glm.flash.v1.sparse_attention")
    assert F.SQ_FMA == 0 and KD.TY == 32 and H.HC_MIX_U == 8 and M.ROUTER_TG == 1024 and M.SPLIT_SHARED == 1

    bf, f32, u32, i32 = "bfloat16", "float32", "uint32", "int32"
    big = 64
    one = 1
    out: list[Spec] = []

    def plain(key: str, base: str, body: str, header: str, inputs: list[Arg], outputs: list[Arg], template: list,
              extra: list | None = None) -> None:
        out.append(Spec(key, f"{base}_{_digest(header + body)}", body, inputs, outputs, header, template, extra or []))

    qmv_in = [Arg("X", bf, big, 2), Arg("W", u32, big, 2), Arg("S", bf, big, 2), Arg("B", bf, big, 2)]
    for k, n in PROJECTIONS:
        plain(f"qmv_{k}_{n}", "tf_glm5_qmv_rows64", K._QMV_ROWS, K._HEADER, qmv_in, [Arg("OUT", bf, big, 2)],
              [("K", k), ("N", n), ("RPS", 4)])
    qmv_partial = K._QMV_ROWS.replace("OUT[r * N + row0 + j] = bfloat(v);", "OUT[r * N + row0 + j] = v;")
    assert qmv_partial != K._QMV_ROWS
    for k, n in PARTIALS:
        plain(f"qmvp_{k}_{n}", "tf_glm5_qmv_rows64_partial", qmv_partial, K._HEADER, qmv_in, [Arg("OUT", f32, big, 2)],
              [("K", k), ("N", n), ("RPS", 4)])
    for key, (bm, bn, sm, sn, tm, tn) in GEMV_T.items():
        plain(f"gemv_t_{key}", "tf_glm5_gemv_t_rows", K._GEMV_T_ROWS, "", [Arg("X", bf, big, 2), Arg("M", bf, big, 2)],
              [Arg("OUT", bf, big, 2)], [("T", bf), ("BM", bm), ("BN", bn), ("SM", sm), ("SN", sn), ("TM", tm),
                                         ("TN", tn)])
    for key, (bm, bn, sm, sn, tm, tn) in GEMV.items():
        plain(f"gemv_{key}", "tf_glm5_gemv_rows", K._GEMV_ROWS, "", [Arg("X", bf, big, 2), Arg("M", bf, big, 2)],
              [Arg("OUT", bf, big, 2)], [("T", bf), ("BM", bm), ("BN", bn), ("SM", sm), ("SN", sn), ("TM", tm),
                                         ("TN", tn)])

    hc_ex_in = [Arg("XOLD", bf, big, 3), Arg("BRANCH", bf, big, 2), Arg("POST", f32, big, 2), Arg("COMB", f32, big, 3),
                Arg("EPS", f32, one)]
    plain("hc_expand_10", "tf_glm5_fused_hc_expand", H._HC_EXPAND, F._HEADER, hc_ex_in,
          [Arg("XNEW", bf, big, 3), Arg("INV", f32, big), Arg("Z", f32, big, 2)],
          [("D", HIDDEN), ("EXPAND", 1), ("SPLIT", 0), ("ZOUT", 0), ("SQ_FMA", F.SQ_FMA)])

    for key, heads in (("kda_rows", KDA_HEADS), ("kda_rows_tp", KDA_HEADS // 2)):  # all heads, or one Mac's in TP2
        plain(key, "tf_glm5_kda_rows", KD._SOURCE, KD._HEADER,
              [Arg("P", bf, big, 2), Arg("CS", bf, big, 2), Arg("CW", f32, big, 2), Arg("FBW", u32, big, 2),
               Arg("FBS", bf, big, 2), Arg("FBB", bf, big, 2), Arg("GBW", u32, big, 2), Arg("GBS", bf, big, 2),
               Arg("GBB", bf, big, 2), Arg("A", f32, big), Arg("DTB", f32, big), Arg("ST", f32, big, 4),
               Arg("ONW", f32, big), Arg("LB", f32, one), Arg("EPS", f32, one)],
              [Arg("Y", bf, big, 2), Arg("ST_OUT", f32, big, 4), Arg("CS_OUT", bf, big, 2)],
              [("H", heads), ("D", KDA_DIM), ("TAPS", TAPS), ("TY", KD.TY), ("FB", 4), ("GB", 4)])

    def router(rr: int) -> list:
        return [("K", HIDDEN), ("NE", EXPERTS), ("RR", rr), ("C", 8 if rr <= 4 else 4), ("NT", M.ROUTER_TG)]

    plain("router", "tf_glm5_fused_router_tg", M._ROUTER_TG, F._HEADER, [Arg("X", f32, big, 2), Arg("RP", bf, big, 5)],
          [Arg("OUT", f32, big, 2)], router(1), [router(rr) for rr in range(2, MAX_ROWS + 1)])
    plain("moe_route", "tf_glm5_fused_moe_route", M._MOE_ROUTE, F._HEADER,
          [Arg("LOGITS", f32, big, 2), Arg("BIAS", f32, big), Arg("SCALE", f32, one)],
          [Arg("PICK", i32, big, 2), Arg("WTS", f32, big, 2), Arg("UIDS", i32, big), Arg("UMEM", i32, big, 2),
           Arg("UCOUNT", i32, 8)],
          [("NE", EXPERTS), ("TOPK", TOP_K), ("MAXR", MAX_ROWS), ("NT", max(32 * MAX_ROWS, -(-EXPERTS // 32) * 32))])
    group = [Arg("UIDS", i32, big), Arg("UMEM", i32, big, 2), Arg("UCOUNT", i32, 8)]
    for part in (1, 2):
        plain(f"moe_gateup_{part}", "tf_glm5_fused_moe_gateup", M._MOE_GATEUP, F._HEADER,
              [Arg("X", bf, big, 2), Arg("GW", u32, big, 3), Arg("GS", bf, big, 3), Arg("GB", bf, big, 3),
               Arg("UW", u32, big, 3), Arg("US", bf, big, 3), Arg("UB", bf, big, 3), Arg("SGU", u32, big, 2),
               Arg("SGUS", bf, big, 2), Arg("SGUB", bf, big, 2), *group, Arg("LIM", f32, one)],
              [Arg("ACT", bf, big, 3)],
              [("K", HIDDEN), ("N", EXPERT), ("RPS", 4), ("TOPK", TOP_K), ("MAXR", MAX_ROWS), ("MAXU", MAXU),
               ("PART", part), ("SB", 4), ("SV", 16), ("SLB", 8)])
        plain(f"moe_down_{part}", "tf_glm5_fused_moe_down", M._MOE_DOWN, F._HEADER,
              [Arg("ACT", bf, big, 3), Arg("DW", u32, big, 3), Arg("DS", bf, big, 3), Arg("DB", bf, big, 3),
               Arg("SDW", u32, big, 2), Arg("SDS", bf, big, 2), Arg("SDB", bf, big, 2), *group],
              [Arg("Y", bf, big, 3)],
              [("K", EXPERT), ("N", HIDDEN), ("RPS", 4), ("TOPK", TOP_K), ("MAXR", MAX_ROWS), ("MAXU", MAXU),
               ("PART", part), ("SB", 4), ("SV", 16), ("SLB", 8)])
    # expert parallel by rows: each Mac half of every routed expert (gate/up rows, down's inputs), fp32 down partials
    plain("moe_gateup_2h", "tf_glm5_fused_moe_gateup", M._MOE_GATEUP, F._HEADER,
          [Arg("X", bf, big, 2), Arg("GW", u32, big, 3), Arg("GS", bf, big, 3), Arg("GB", bf, big, 3),
           Arg("UW", u32, big, 3), Arg("US", bf, big, 3), Arg("UB", bf, big, 3), Arg("SGU", u32, big, 2),
           Arg("SGUS", bf, big, 2), Arg("SGUB", bf, big, 2), *group, Arg("LIM", f32, one)],
          [Arg("ACT", bf, big, 3)],
          [("K", HIDDEN), ("N", EXPERT // 2), ("RPS", 4), ("TOPK", TOP_K), ("MAXR", MAX_ROWS), ("MAXU", MAXU),
           ("PART", 2), ("SB", 4), ("SV", 16), ("SLB", 8)])
    down_partial = M._MOE_DOWN.replace("Y[(size_t(r) * SLOTS + slot) * N + row0 + j] = bfloat(v);",
                                       "Y[(size_t(r) * SLOTS + slot) * N + row0 + j] = v;")
    assert down_partial != M._MOE_DOWN
    plain("moe_down_2h", "tf_glm5_fused_moe_down_partial", down_partial, F._HEADER,
          [Arg("ACT", bf, big, 3), Arg("DW", u32, big, 3), Arg("DS", bf, big, 3), Arg("DB", bf, big, 3),
           Arg("SDW", u32, big, 2), Arg("SDS", bf, big, 2), Arg("SDB", bf, big, 2), *group],
          [Arg("Y", f32, big, 3)],
          [("K", EXPERT // 2), ("N", HIDDEN), ("RPS", 4), ("TOPK", TOP_K), ("MAXR", MAX_ROWS), ("MAXU", MAXU),
           ("PART", 2), ("SB", 4), ("SV", 16), ("SLB", 8)])
    plain("moe_combine", "tf_glm5_fused_moe_combine_split", M._MOE_COMBINE_SPLIT, F._HEADER,
          [Arg("YS", bf, big, 2), Arg("Y", bf, big, 3), Arg("WTS", f32, big, 2)], [Arg("OUT", bf, big, 2)],
          [("D", HIDDEN), ("TOPK", TOP_K)])

    for key, heads in (("sparse_attention", MLA_HEADS), ("sparse_attention_tp", MLA_HEADS // 2)):  # TP2: one Mac's heads
        out.append(Spec(key, f"tf_glm5_indexed_sparse_attention_{_digest(SA._SOURCE)}", SA._SOURCE,
                        [Arg("queries", bf, big, 3), Arg("keys", bf, big, 2), Arg("indices", i32, big, 2),
                         Arg("scale", f32, one), Arg("meta", i32, one)],
                        [Arg("out", bf, big, 3)], "", [("QK_DIM", RANK), ("TOPK", TOPK_KEYS), ("HEADS", heads)]))
    return out


def rendered(spec: Spec) -> tuple[str, str]:
    """(function name, file text) for a spec; extra instantiations get their own host names."""

    fname, text = kernel_text(spec.name, spec.inputs, spec.outputs, spec.source, spec.header, spec.template)
    for template in spec.extra:
        host = function_name(spec.name, spec.inputs, spec.outputs, template)
        args = fname + "<" + ", ".join(_template_value(v) for _, v in template) + ">"
        text += f'template [[host_name("{host}")]] [[kernel]] decltype({args}) {args};\n'
    return fname, ("// Generated by tools/zig/gen_glm_kernels.py from our Python kernels; do not edit.\n" + PREAMBLE
                   + BF16_MATH + strip_comments(text))


def names_zig(entries: list[tuple[Spec, str]]) -> str:
    lines = ["//! Generated by tools/zig/gen_glm_kernels.py: each kernel's function names and source; do not edit.",
             "", "pub const Kernel = struct { key: []const u8, functions: []const [:0]const u8, source: []const u8 };",
             "", "pub const all = [_]Kernel{"]
    for spec, fname in entries:
        names = [fname] + [function_name(spec.name, spec.inputs, spec.outputs, t) for t in spec.extra]
        listed = ", ".join(f'"{n}"' for n in names)
        lines.append(f'    .{{ .key = "{spec.key}", .functions = &.{{ {listed} }}, .source = @embedFile("{spec.key}.metal") }},')
    lines += ["};", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the written files are stale")
    args = ap.parse_args()
    py_kernels._stand_in()                     # the sources only: never import MLX here
    entries = [(s, *rendered(s)) for s in specs()]
    files = {OUT / f"{s.key}.metal": text for s, _, text in entries}
    files[NAMES] = names_zig([(s, f) for s, f, _ in entries])
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
        extra = sorted(p for p in OUT.glob("*.metal") if p not in files)
        for path in extra:
            print(f"not generated: {path.relative_to(ROOT)}")
        return 1 if stale or extra else 0
    print(f"wrote {len(entries)} kernels to {OUT.relative_to(ROOT)} and {NAMES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

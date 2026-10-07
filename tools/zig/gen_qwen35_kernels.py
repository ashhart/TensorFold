"""Generate the Qwen3.5-2B native operation kernels from the existing row decoder's arithmetic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import py_kernels
from metal_source import PREAMBLE, Arg, kernel_text, strip_comments

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "zig/kernels/metal/qwen3_5"
PROJECTIONS = ((6144, 2048), (2048, 2048), (16, 2048), (4096, 2048), (512, 2048), (2048, 6144), (248320, 2048))


def specs() -> list[dict]:
    """Fixed eight-row MMA tiles keep the same FMA chains for one row, windows and prompt chunks."""
    mm = py_kernels.load("kernels.qwen.dense.v1.simd_qmm")
    glue = py_kernels.load("kernels.qwen.dense.v1.row_glue")
    lane = py_kernels.load("kernels.qwen.dense.v1.lane_glue")
    attn = py_kernels.load("kernels.qwen.dense.v1.row_attention")
    bf, f32, u32, i32 = "bfloat16", "float32", "uint32", "int32"
    out = []

    def add(key, body, inputs, outputs, constants=(), header="", template=(), launch=None):
        source = "".join(f"constexpr int {k} = {v};\n" for k, v in constants) + body
        fn, text = kernel_text("qwen35_" + key, inputs, outputs, source, header, list(template))
        out.append(dict(key=key, function=fn, source=PREAMBLE + strip_comments(text), launch=launch,
                        body=source, header=header, inputs=inputs, outputs=outputs, template=template))

    for n, k in PROJECTIONS:
        s, nt, sgs = mm.splits(n, k), 4 if n % 32 == 0 else 2, 8
        constants = (("K", k), ("N", n), ("S", s), ("SGS", sgs), ("NT", nt), ("RT", 1), ("GS", 64))
        body = "#define LOAD8(r, j) (((const device uint4*)X)[size_t(r) * (K / 8) + (j)])\n" + mm._MMA
        add(f"qmm_{n}_{k}", body,
            [Arg("X", bf, 64, 2), Arg("W", u32), Arg("SC", bf), Arg("BI", bf), Arg("ONE", f32, 1)],
            [Arg("OUT", bf)], constants, mm._HEADER,
            launch=dict(columns=8 * nt, threads=sgs * 32, row_tile=8))
    add("norm", glue._SPECS["norm"][0],
        [Arg("H", bf), Arg("R", bf), Arg("Wt", bf), Arg("eps", f32, 1)],
        [Arg("HO", bf), Arg("XO", bf)], (("K", 2048),))
    add("norm_nores", glue._SPECS["norm_nores"][0],
        [Arg("H", bf), Arg("Wt", bf), Arg("eps", f32, 1)], [Arg("XO", bf)], (("K", 2048),))
    pre = lane._GDN_PRE.replace("ss / float(DK) + 1e-6f", "ss / float(DK) + (1e-6f / float(DK))") + """
    for (int r = 0; r < TAPS - 1; r++) {
      const int row = windows[w * TAPS + 1 + r];
      for (int j = 0; j < PER; j++) {
        const int c = c0 + int(lane) * PER + j;
        CO[(int(w) * (TAPS - 1) + r) * C + c] = row < TAPS - 1 ? CS[row * C + c] : QKV[(row - (TAPS - 1)) * C + c];
      }
    }
    """
    add("gdn_pre", pre,
        [Arg("QKV", bf), Arg("CS", bf), Arg("CW", bf), Arg("windows", i32), Arg("Ain", bf), Arg("Bin", bf),
         Arg("ALOG", f32), Arg("DT", bf)],
        [Arg("Q", bf), Arg("Kout", bf), Arg("Vout", bf), Arg("G", f32), Arg("BETA", bf), Arg("CO", bf)],
        (("NK", 16), ("NV", 16), ("DK", 128), ("DV", 128), ("TAPS", 4)))
    chain = glue._CHAIN.replace("        #pragma unroll\n", "")
    chain = chain.replace("        auto n =", "        const int W = dims[0];\n        auto n =", 1)
    tail = """        auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];"""
    assert tail in chain
    chain = chain.replace(tail, "")
    end = "          vc = vn; gc = gn; bc = bn;"
    assert end in chain
    chain = chain.replace(end, end + """
          if (dims[1] || node + 1 == W) {
            const int saved = dims[1] ? node : 0;
            auto o_state = state_out + ((saved * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
          }
    """)
    add("gdn_chain", chain,
        [Arg("q", bf), Arg("k", bf), Arg("v", bf), Arg("g", f32), Arg("beta", bf), Arg("state_in", f32),
         Arg("dims", i32, 2)], [Arg("y", bf), Arg("state_out", f32)],
        (("Dk", 128), ("Dv", 128), ("Hk", 16), ("Hv", 16)), template=(("InT", bf),))
    add("gdn_post", glue._GDN_POST,
        [Arg("Y", bf), Arg("Z", bf), Arg("NW", bf), Arg("eps", f32, 1)], [Arg("OUT", bf)],
        (("NV", 16), ("DV", 128), ("ZS", 2048), ("ZO", 0)))
    act = glue._MLP_ACT.replace("GU[m * 2 * N + i]", "GATE[m * N + i]")
    act = act.replace("GU[m * 2 * N + N + i]", "UP[m * N + i]")
    add("mlp_act", act, [Arg("GATE", bf), Arg("UP", bf)], [Arg("HOUT", bf)], (("N", 6144),))
    partial = attn._PARTIAL.replace("depth[w]", "w")
    partial = partial.replace("P + path[w * MAXD + (pos - P)]", "pos")
    partial = partial.replace("((qh * W) + w) * D", "((qh * dims[5]) + dims[6] + w) * D")
    add("attn_partial", partial, [Arg("Q", bf), Arg("K", bf), Arg("V", bf), Arg("scale", f32, 1), Arg("dims", i32)],
        [Arg("PM", f32), Arg("PL", f32), Arg("PO", f32)],
        (("D", 256), ("G", 4), ("CK", 128), ("SPLIT", 4), ("BLK", 4)))
    merge = attn._MERGE.replace("((qh * W) + w) * D", "((w * 8) + qh) * D")
    add("attn_merge", merge, [Arg("PM", f32), Arg("PL", f32), Arg("PO", f32), Arg("dims", i32)],
        [Arg("OUT", bf)], (("D", 256),))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    entries = specs()
    files = {f"{e['key']}.metal": "// Generated by tools/zig/gen_qwen35_kernels.py; do not edit.\n" + e["source"]
             for e in entries}
    index = "//! Generated Qwen3.5-2B kernels and fixed projection launch geometries.\n"
    index += "pub const Kernel = struct { key: []const u8, function: [:0]const u8, source: []const u8, columns: usize = 0, threads: usize = 0 };\n"
    index += "pub const all = [_]Kernel{\n"
    for e in entries:
        index += "    .{ .key = " + json.dumps(e["key"]) + ", .function = " + json.dumps(e["function"])
        index += ', .source = @embedFile("' + e["key"] + '.metal")'
        if e["launch"]:
            index += f", .columns = {e['launch']['columns']}, .threads = {e['launch']['threads']}"
        index += " },\n"
    files["kernels.zig"] = index + "};\n"
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    stale = 0
    for name, text in files.items():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                print("stale:", path.relative_to(ROOT))
                stale += 1
        else:
            path.write_text(text)
    return int(stale > 0)


if __name__ == "__main__":
    raise SystemExit(main())

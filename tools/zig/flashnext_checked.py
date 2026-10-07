"""Write checked-in Flash Next kernel sources and roles_gen.zig from a dump. CPU only, no compile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAXW = 16
# the widening lane kernels on GPUs without tensor units: the recorder's five rewrites (flashnext_dump.py SIMD_LANES)
SIMD_LANES = (
    ("  const short fn = ((qid & 2) | (lane & 1)) * 4;\n",
     "  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);   // simdgroup matrices: columns fn + {0, 1, 8, 9}\n"),
    ("  const device uint4* sbv = (const device uint4*)SBt;\n  bool colok[NF];\n"
     "  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;\n",
     "  const device uint2* sbv = (const device uint2*)SBt;\n  bool colok[NF][2];\n"
     "  for (int f = 0; f < NF; f++) for (int h = 0; h < 2; h++) colok[f][h] = n0 + f * 16 + fn + 8 * h < N;\n"),
    ("      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);\n",
     "      const uint4 q = uint4(colok[f][0] ? sbv[(g * N + n0 + f * 16 + fn) / 2] : uint2(0),\n"
     "                            colok[f][1] ? sbv[(g * N + n0 + f * 16 + fn + 8) / 2] : uint2(0));\n"),
    ("C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));",
     "C[t][i] = fma(s[f][j], P[t * NF * 8 + r * 8 + f * 4 + j], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));"),
    ("          if (m < M && nn < N)\n"
     "            for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);\n",
     "          if (m < M)\n            for (int j = 0; j < 4; j++)\n"
     "                if (colok[f][j >> 1])\n"
     "                  Y[m * N + nn + (j & 1) + 8 * (j >> 1)] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);\n"),
)


def role_key(site: str) -> str:
    """The engine's role name for a launch site (tools/zig/flashnext_roles.py)."""
    name, _, rows = site.partition("|")
    for a, b in (("_row@", "@"), ("_mma@", "@"), ("q4_gdn_pipe@", "q4_gdn@"), ("q4_gdn_step@", "q4_gdn@")):
        name = name.replace(a, b)
    if "#[" in name:
        head, shape = name.split("#[", 1)
        dims, r = shape.rstrip("]").split(", "), int(rows or 1)
        lead = [] if int(dims[0]) == r else ["4R"] if int(dims[0]) == 4 * r else [dims[0]]
        name = head + "#[" + ", ".join(lead + dims[1:]) + "]"
    return f"{name}|{int(rows or 1)}"


def zig_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("repo", type=Path, help="the worktree root")
    ap.add_argument("--maxw", type=int, default=MAXW)
    args = ap.parse_args()

    plan = json.loads((args.dump / "plan.json").read_text())
    variants = plan["variants"]

    # the roles, one per (stem, rows): the recorded grid and threadgroup, plus the variant's file and argument lists
    roles: dict[str, dict] = {}
    for site, v in plan["sites"].items():
        k = role_key(site)
        assert k not in roles or roles[k] == v, f"two sites map to {k}"
        roles[k] = v
    for k, v in roles.items():
        var = variants[v["function"]]
        v["file"] = var["file"]
        v["inputs"] = var["inputs"]
        v["outputs"] = var["outputs"]
        v["meta"] = var["meta"]

    # Unrecorded widths follow callRows. One width fills as a constant. The indexer trio is not launched from the table.
    stems: dict[str, dict[int, dict]] = {}
    for k, v in roles.items():
        stem, _, rows = k.rpartition("|")
        stems.setdefault(stem, {})[int(rows)] = v
    for stem, at in stems.items():
        groups: dict[str, dict[int, dict]] = {}
        for w, v in at.items():
            groups.setdefault(v["function"], {})[w] = v
        for gw in groups.values():
            ws = sorted(gw)
            if len(ws) < 2:
                # A single recorded width fills the other widths as a constant.
                only = gw[ws[0]]
                for w in range(1, args.maxw + 1):
                    if w not in at:
                        at[w] = {**only, "derived": True}
                continue
            w0, w1 = ws[0], ws[1]
            axis = next((a for a in (0, 1, 2) if gw[w1]["grid"][a] != gw[w0]["grid"][a]), None)
            for w in range(1, args.maxw + 1):
                if w in at:
                    continue
                grid = list(gw[w0]["grid"])
                if axis is not None:
                    grid[axis] = gw[w0]["grid"][axis] * w // w0
                at[w] = {**gw[w0], "grid": grid, "derived": True}

    # the checked-in sources: every recorded kernel file, verbatim, plus the no-tensor-unit lane variants
    out_k = args.repo / "zig" / "kernels" / "metal" / "flashnext"
    out_k.mkdir(parents=True, exist_ok=True)
    files = {}
    for var in variants.values():
        src = args.dump / var["file"]
        text = src.read_text()
        name = Path(var["file"]).name
        (out_k / name).write_text(text)
        files[var["file"]] = name
        if name.startswith("lane_qmm_bytes_grouped"):
            lanes = text
            for old, new in SIMD_LANES:
                assert lanes.count(old) == 1, f"{name}: rewrite pattern not found once: {old[:50]!r}"
                lanes = lanes.replace(old, new)
            (out_k / name.replace(".metal", "-lanes.metal")).write_text(lanes)
            files["lanes:" + var["file"]] = name.replace(".metal", "-lanes.metal")

    # the embedded sources live beside their files (the kernel_sources module's path), the roles table in the family
    src_lines = [
        "//! The Flash Next kernel sources serve launches, generated by tools/zig/flashnext_checked.py from a",
        "//! recorded dump: the recorded texts verbatim, and the -lanes files the no-tensor-unit recorder writes",
        "//! (flashnext_dump.py's SIMD_LANES rewrites of the same lane kernels).",
        "",
        "pub const Source = struct { name: []const u8, text: []const u8 };",
        "",
        "pub const sources = [_]Source{",
    ]
    for file in sorted(files.values()):
        src_lines.append(f"    .{{ .name = {zig_str(file)}, .text = @embedFile({zig_str(file)}) }},")
    src_lines += ["};", ""]
    (args.repo / "zig" / "kernels" / "metal" / "flashnext" / "sources_gen.zig").write_text("\n".join(src_lines))
    lines = [
        "//! The Flash Next role table, generated by tools/zig/flashnext_checked.py from a recorded dump.",
        "//! Every site the engine launches: its kernel function, the checked-in file that holds it, the",
        "//! argument order the recorded variant binds, and the grid and threadgroup per width 1..16.",
        "//! The texts ride kernel_sources.flashnext_gen.sources (the embeds must stay in that module's path).",
        "",
        "pub const Entry = struct {",
        "    site: []const u8,",
        "    function: []const u8,",
        "    file: []const u8,",
        "    inputs: []const []const u8,",
        "    outputs: []const []const u8,",
        "    meta: []const []const u8,",
        "    grid: [3]u32,",
        "    tg: [3]u32,",
        "};",
        "",
        "pub const entries = [_]Entry{",
    ]
    count = 0
    for stem in sorted(stems):
        for w in sorted(stems[stem]):
            v = stems[stem][w]
            site = f"{stem}|{w}"
            ins = ", ".join(zig_str(s) for s in v["inputs"])
            outs = ", ".join(zig_str(s) for s in v["outputs"])
            meta = ", ".join(zig_str(s) for s in v["meta"])
            g = ", ".join(str(x) for x in v["grid"])
            t = ", ".join(str(x) for x in v["threadgroup"])
            lines.append(f"    .{{ .site = {zig_str(site)}, .function = {zig_str(v['function'])}, "
                         f".file = {zig_str(files[v['file']])}, .inputs = &.{{ {ins} }}, .outputs = &.{{ {outs} }}, "
                         f".meta = &.{{ {meta} }}, .grid = .{{ {g} }}, .tg = .{{ {t} }} }},")
            count += 1
    lines += ["};", ""]
    out = args.repo / "zig" / "src" / "families" / "flashnext" / "roles_gen.zig"
    out.write_text("\n".join(lines))
    print(f"{len(files)} kernel files written to {out_k.relative_to(args.repo)}")
    print(f"{count} role entries over {len(stems)} stems written to {out.relative_to(args.repo)}")
    derived = sum(1 for stem in stems.values() for v in stem.values() if v.get("derived"))
    print(f"{derived} widths filled by the callRows rule, the rest recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())

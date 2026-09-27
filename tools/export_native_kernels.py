"""Export TensorFold's Metal source verbatim for the Zig/MLX-C host.

Development-only generator. The native executable embeds the generated sources;
it neither imports Python nor calls a Python subprocess. Run after kernel edits.
"""
import argparse
import ast
import json
from pathlib import Path

import mlx.core as mx
from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_glue, lane_tree, lane_attention
from tensorfold.kernels.qwen.dense.v1 import row_attention, simd_qmm
from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_gdn, row_forward, row_qmv
from tensorfold.kernels.nemotron.lightning.v1 import kernels as nemotron, rows as nemotron_rows
from tensorfold.kernels.qwen.flash_next.v1 import kernels as flash
from tensorfold.engine import gpu_sampling, topk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "native" / "metal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed sources differ from Python")
    args = parser.parse_args()
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)

    def emit(path, content):
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale native kernel: {path}; rerun this tool without --check")
        else:
            path.write_text(content)

    original = mx.fast.metal_kernel
    mx.fast.metal_kernel = lambda **kwargs: kwargs
    definitions = []
    def export(key, spec):
        emit(OUT / f"{key}.metal", spec["source"])
        emit(OUT / f"{key}.h", spec.get("header", ""))
        ins = ", ".join(json.dumps(x) for x in spec["input_names"])
        outs = ", ".join(json.dumps(x) for x in spec["output_names"])
        # Match Zig 0.17's formatter for multi-element array literals.
        if len(spec["input_names"]) > 1:
            ins = f" {ins} "
        if len(spec["output_names"]) > 1:
            outs = f" {outs} "
        contiguous = str(spec.get("ensure_row_contiguous", True)).lower()
        definitions.append(
            f'pub const {key} = Spec{{ .name = "{key}", '
            f'.inputs = &.{{{ins}}}, .outputs = &.{{{outs}}}, '
            f'.source = @embedFile("metal/{key}.metal"), '
            f'.header = @embedFile("metal/{key}.h"), .contiguous = {contiguous} }};'
        )
    try:
        gpu_sampling._kernel = None
        gpu_sampling._kernel_ids = None
        topk._kernels.clear()
        export("gpu_sample", gpu_sampling._get_kernel())
        export("gpu_sample_ids", gpu_sampling._get_kernel_ids())
        export("radix_topk", topk._kernel())
        for module, names in [
            (lane_qmm, ["xsum", "main", "main_tiled"]),
            (lane_glue, ["norm", "norm_nores", "gdn_pre", "gdn_post", "mlp_act"]),
            (lane_tree, ["tree", "replay"]),
            (lane_attention, ["tail", "tree_merge", "partial", "partial_direct", "partial_128", "partial_direct_128", "merge"]),
            (row_attention, ["partial", "merge"]),
            (lane_fuse, list(lane_fuse._variant_sources())),
            (row_forward, list(row_forward._SPECS)),
        ]:
            (module._variants if module is lane_fuse else module._kernels).clear()
            for name in names:
                spec = module._kernel(name)
                key = module.__name__.rsplit(".", 1)[-1] + "_" + name
                export(key, spec)
        simd_qmm._kernels.clear()
        # The custom load prologue is the exact example exercised by upstream tests.
        # It is a diagnostic specialization, not an arbitrary runtime shader API.
        test_tree = ast.parse((ROOT / "tests/test_simd_qmm.py").read_text())
        prologue_test = next(node for node in test_tree.body if isinstance(node, ast.FunctionDef)
                             and node.name == "test_prologue_gives_the_unfused_bits")
        scale_header = next(ast.literal_eval(node.value) for node in prologue_test.body
                            if isinstance(node, ast.Assign) and node.targets[0].id == "header")
        scale = simd_qmm.Prologue("scale", "scale8(X, E, (r), (j), K)", ("E",), scale_header)
        for kind in ("mma", "scalar"):
            export(f"simd_qmm_{kind}", simd_qmm._compiled(kind, ()))
            export(f"simd_qmm_{kind}_dep", simd_qmm._compiled(kind, (), dep=True))
            export(f"simd_qmm_{kind}_scale", simd_qmm._compiled(kind, (), prologue=scale))
        row_qmv._kernel = None
        export("row_qmv", row_qmv._compiled())
        row_forward._variants.clear()
        for norm in (False, True):
            for epilogue in ("plain", "act", "residual"):
                export(f"row_forward_qmv_{int(norm)}_{epilogue}", row_forward._variant_kernel(norm, epilogue))
        export("row_forward_gate_up_act", dict(source=row_forward._gate_up_act_source(), header=row_qmv._HEADER,
               input_names=["X", "W", "S", "B"], output_names=["OUT"]))
        for name, source in (("step", lane_gdn._STEP_SOURCE), ("step_kh", lane_gdn._STEP_SOURCE_KH)):
            export("lane_gdn_" + name, dict(source=source,
                   input_names=["q", "k", "v", "log_g", "beta", "s0kq", "log_prev", "k_hist", "d_hist", "lg_hist", "tlen"],
                   output_names=["y", "delta_out", "log_out"]))
        # Extract static kernel declarations directly, preserving each module's header.
        # Dynamic source variants are enumerated below, so no model weights are needed.
        for module in (nemotron, nemotron_rows, flash):
            module._kernels.clear()
            tree = ast.parse(Path(module.__file__).read_text())
            for call in ast.walk(tree):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "_kernel" and call.args
                        and isinstance(call.args[0], ast.Constant)):
                    continue
                try:
                    spec = eval(compile(ast.Expression(call), module.__file__, "eval"), vars(module))
                except NameError:  # locally constructed add_norm variants below
                    continue
                export(call.args[0].value, spec)
        for kind, mix, inputs in [
            ("plain", nemotron._MIX_PLAIN, ["H", "X", "W", "eps"]),
            ("moe", nemotron._MIX_MOE, ["H", "Y", "WE", "SH", "W", "eps"]),
            ("experts", nemotron._MIX_EXPERTS, ["H", "Y", "WE", "W", "eps"]),
        ]:
            name = "nemotron_add_norm_" + kind
            export(name, nemotron._kernel(name, nemotron._ADD_NORM.replace("MIX", mix), inputs, ["HN", "OUT"]))
        for kind, branch, writeback, names in [
            ("none", "", "", ["H"]),
            ("plain", flash._BRANCH_PLAIN, flash._WRITEBACK, ["H", "INJ", "BR"]),
            ("grouped", flash._BRANCH_GROUPED, flash._WRITEBACK, ["H", "INJ", "Y", "WTS", "LG"]),
        ]:
            name = "q4_hc_norm_" + kind
            source = flash._HC_NORM.replace("BRANCH", branch).replace("WRITEBACK", writeback)
            export(name, flash._kernel(name, source, names, ["HN", "SSP"]))
        export("q4_router_float", flash._kernel("q4_router_float", flash._ROUTER.replace("OUT_T", "float"), ["X", "GW", "rows"], ["OUT"]))
        export("q4_router_bfloat", flash._kernel("q4_router_bfloat", flash._ROUTER.replace("OUT_T", "bfloat"), ["X", "GW", "rows"], ["OUT"]))
        export("q4_ple_lookup", flash._kernel("q4_ple_lookup", flash._PLE_LOOKUP,
               ["IDS", "GSTART"] + [f"{kind}{g}" for g in range(8) for kind in ("W", "S", "B")], ["OUT"]))
    finally:
        mx.fast.metal_kernel = original
    emit(ROOT / "native" / "kernel_sources.zig",
        '// Generated by tools/export_native_kernels.py; do not edit.\n'
        'pub const Spec = struct { name: [:0]const u8, inputs: []const [:0]const u8, '
        'outputs: []const [:0]const u8, source: [:0]const u8, header: [:0]const u8, contiguous: bool };\n'
        + "\n".join(definitions) + "\n"
    )
    print(f"{'Checked' if args.check else 'Exported'} {len(definitions)} kernels in {OUT}")


if __name__ == "__main__":
    main()

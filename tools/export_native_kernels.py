"""Export TensorFold's Metal kernels for the Zig/MLX-C host.

Development-only generator. The native executable embeds the generated sources;
it neither imports Python nor calls a Python subprocess. Run after kernel edits.
Integer constants become templates and Metal math precision is explicit. Retired
kernel interfaces come from the versioned independent oracles in native_legacy.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_glue, lane_tree, lane_attention
from tensorfold.kernels.qwen.dense.v1 import row_attention, simd_qmm, affine_rows
from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_gdn
from tools.native_legacy import row_forward, row_qmv, tree_attention, nemotron as legacy_nemotron
from tensorfold.kernels.nemotron.lightning.v1 import kernels as nemotron
from tools.native_legacy import nemotron_rows
from tools.native_legacy import flash
from tensorfold.kernels.qwen.flash_next.v1 import attention, base
from tensorfold.engine import gpu_sampling, topk
from tensorfold.kernels.qwen.prism.v1 import rotate
from tools.native_runtime import require_mlx

OUT = ROOT / "native" / "metal"


def main():
    require_mlx()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed sources differ from Python")
    args = parser.parse_args()
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)

    files = {}
    def emit(path, content):
        if path in files:
            raise ValueError(f"Duplicate kernel export: {path}")
        files[path] = content

    def write_or_check(path, content):
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale native kernel: {path}; rerun this tool without --check")
        else:
            path.write_text(content)

    original = mx.fast.metal_kernel
    mx.fast.metal_kernel = lambda **kwargs: kwargs
    definitions = []
    def export(key, spec):
        if not isinstance(spec, dict):
            # Current lane projections bake their integer templates at launch;
            # Zig supplies the same constants as Metal template arguments.
            spec = dict(source=spec.body, header=lane_qmm._HEADER,
                        input_names=spec.inputs, output_names=spec.outputs)
        # Undo Python's launch-specific constexpr/thread reservation wrappers.
        spec = dict(spec)
        spec["source"] = re.sub(r"\A(?:  constexpr int \w+ = -?\d+;\n)+", "", spec["source"])
        spec["header"] = re.sub(r"\n\[\[max_total_threads_per_threadgroup\(\d+\)\]\]\n$", "", spec.get("header", ""))
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
        export("affine_rows", dict(source=affine_rows._SOURCE, header=affine_rows._HEADER,
               input_names=["X", "W", "SC", "BI"], output_names=["OUT"]))
        for name, body, inputs in (("rotate", rotate._ROTATE, ["X", "SG"]),
                                   ("embed", rotate._EMBED, ["IDS", "W", "SC", "BI", "SG"]),
                                   ("dense", rotate._DENSE, ["X", "WT"])):
            export("prism_" + name, dict(source=body, input_names=inputs, output_names=["OUT"]))
        topk._kernels.clear()
        for mapped in (False, True):
            export("gpu_sample_ids" if mapped else "gpu_sample", dict(
                source=gpu_sampling._SOURCE_IDS if mapped else gpu_sampling._SOURCE,
                header=gpu_sampling._HEADER, input_names=["L", "seeds", "positions", "cfg", "kcap"] + (["IDS"] if mapped else []),
                output_names=["TOK"]))
        export("radix_topk", dict(source=topk._SOURCE, input_names=["X", "dims"], output_names=["IDX", "VAL"]))
        for module, names in [
            (lane_qmm, ["xsum", "main", "main_tiled", "lowbit", "bytes"]),
            (lane_glue, ["norm", "norm_nores", "gdn_pre", "gdn_post", "mlp_act"]),
            (lane_tree, ["tree", "replay"]),
            (lane_attention, ["partial", "partial_direct", "partial_128", "partial_direct_128", "merge"]),
            (row_attention, ["partial", "merge"]),
            (lane_fuse, list(lane_fuse._variant_sources())),
            (row_forward, list(row_forward._SPECS)),
        ]:
            (module._variants if module is lane_fuse else module._kernels).clear()
            for name in names:
                spec = module._kernel(name)
                key = module.__name__.rsplit(".", 1)[-1] + "_" + name
                export(key, spec)
        export("lane_attention_tail", dict(source=tree_attention._TAIL, header=lane_attention._HEADER,
               input_names=["QB", "K", "V", "scale", "dims", "paths", "depths", "POA", "PMA", "PLA"],
               output_names=["PO", "PM", "PL"], ensure_row_contiguous=False))
        export("lane_attention_tree_merge", dict(source=tree_attention._TREE_MERGE, header=lane_attention._HEADER,
               input_names=["POA", "PMA", "PLA", "POB", "PMB", "PLB", "dims"], output_names=["OUT"]))
        pre = lane_glue._GDN_PRE.replace("float(Ain[w * NV + hv])", "float(Ain[w * ZS + AO + hv])")
        pre = pre.replace("float(Bin[w * NV + hv])", "float(Bin[w * ZS + BO + hv])")
        export("lane_fuse_gdn_pre", dict(source=pre,
               input_names=["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT"],
               output_names=["Q", "Kout", "Vout", "G", "BETA"]))
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
        export("nemotron_mamba_step", dict(source=legacy_nemotron._MAMBA_STEP,
               input_names=["P", "CS_IN", "S_IN", "CW", "CB", "A_LOG", "DSKIP", "DT_BIAS", "limits", "dims"],
               output_names=["Y", "CS_OUT", "S_OUT"]))
        for name, source, ins, outs in (
            ("qmv", nemotron_rows._QMV, ["X", "W", "S", "B"], ["OUT"]),
            ("expert_up", nemotron_rows._EXPERT_UP, ["X", "IDS", "W", "S", "B"], ["ACT"]),
            ("expert_down", nemotron_rows._EXPERT_DOWN, ["X", "IDS", "W", "S", "B"], ["Y"]),
        ):
            export("nemotron_rows_" + name, dict(source=source, header=nemotron_rows._HEADER,
                   input_names=ins, output_names=outs))
        for module in (nemotron, flash):
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
                key = call.args[0].value
                if key in ("nemotron_mamba_conv", "nemotron_mamba_scan"):
                    continue
                if key == "nemotron_route":
                    spec["source"] = legacy_nemotron._ROUTE
                if module is flash:
                    names = [node.id for node in ast.walk(call.args[1]) if isinstance(node, ast.Name)]
                    if names and all(hasattr(attention, name) for name in names):
                        spec["source"] = eval(compile(ast.Expression(call.args[1]), attention.__file__, "eval"), vars(attention))
                        if spec.get("header") == flash._QDOT_HEADER:
                            spec["header"] = base.QDOT_HEADER
                export(key, spec)
        for kind, mix, inputs in [
            ("plain", nemotron._MIX_PLAIN, ["H", "X", "W", "eps"]),
            ("moe", nemotron._MIX_MOE, ["H", "Y", "WE", "SH", "W", "eps"]),
            ("experts", legacy_nemotron._MIX_EXPERTS, ["H", "Y", "WE", "W", "eps"]),
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
    if len(definitions) != 92:
        raise ValueError(f"Native catalog must retain all 92 kernels; found {len(definitions)}")
    emit(ROOT / "native" / "kernel_sources.zig",
        '// Generated by tools/export_native_kernels.py; do not edit.\n'
        'pub const Spec = struct { name: [:0]const u8, inputs: []const [:0]const u8, '
        'outputs: []const [:0]const u8, source: [:0]const u8, header: [:0]const u8, contiguous: bool };\n'
        + "\n".join(definitions) + "\n"
    )
    for path, content in files.items():
        write_or_check(path, content)
    print(f"{'Checked' if args.check else 'Exported'} {len(definitions)} kernels in {OUT}")


if __name__ == "__main__":
    main()

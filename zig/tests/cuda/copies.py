"""Checks (or rewrites) zig/kernels/cuda copies of the Python engine's CUDA device code: same lines, same bits."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

GDN_FOOTER = (
    "// The instantiations the Python wrappers launch (gdn_replay_cuda, dispatch_tree for bf16 keys).\n"
    "#define TF_REPLAY(QK) template __global__ void tf_gdn::replay_kernel<QK, 8, 4>(const long long*, int, const int*, int, \\\n"
    "    const int*, int, float*, int, int, int);\n"
    "TF_REPLAY(__nv_bfloat16)\n"
    "TF_REPLAY(float)\n"
    "#define TF_TREE(S, R, W, C) template __global__ void tf_gdn::tree_kernel<__nv_bfloat16, S, R, W, C>( \\\n"
    "    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, \\\n"
    "    const long long*, const int*, const int*, int, __nv_bfloat16*, int, int, int, tf_gdn::Pending<__nv_bfloat16>, \\\n"
    "    float*, const long long*, bool);\n"
    "TF_TREE(0, 8, 4, true)\n"
    "TF_TREE(1, 8, 4, false)\n"
    "TF_TREE(2, 8, 2, false)\n"
    "TF_TREE(2, 4, 4, false)\n"
    "TF_TREE(4, 2, 4, false)\n"
    "TF_TREE(8, 2, 4, false)\n"
    "TF_TREE(16, 2, 2, false)\n"
    "TF_TREE(32, 2, 1, false)\n"
)

# name: source, first and last line taken, lines dropped (ATen includes), (namespace line, new name), instantiations
COPIES = {
    "gdn.cu": ("src/tensorfold/cuda/kernels/gdn.cu", 1, 319, (3, 4), (10, "tf_gdn"), GDN_FOOTER),
    "qmm_frag.cuh": ("src/tensorfold/cuda/kernels/qmm_frag.cuh", 1, 118, (), None, ""),
    "experts.cuh": ("src/tensorfold/cuda/experts.cuh", 1, 105, (), None, ""),
    "qmm_group.cu": (
        "src/tensorfold/cuda/kernels/qmm_group.cu", 1, 348, (3, 5), (14, "tf_qmm_group"),
        "// The instantiations Nemotron's windows launch on sm_121 (tile 2: rows <= 16, bf16 out).\n"
        "template __global__ void tf_qmm_group::group_kernel<64, 16, 64, 1, 4, 8, false, false, false, false>(\n"
        "    const __nv_bfloat16*, const float*, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);\n",
    ),
    "qmm_prefill.cu": (
        "src/tensorfold/cuda/kernels/qmm_prefill.cu", 1, 151, (3, 5), (11, "tf_qmm_prefill"),
        "// The instantiation prefill_matmul launches (tile 0: 128x128 on 2x4 warps, 3 stages), bf16 out.\n"
        "template __global__ void tf_qmm_prefill::prefill_kernel<64, 128, 128, 2, 4, 3, false>(\n"
        "    const __nv_bfloat16*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int,\n"
        "    int, int, int);\n",
    ),
    "experts.cu": (
        "src/tensorfold/cuda/experts.cu", 1, 286, (3, 4, 7), (11, "tf_experts"),
        "// Decode form for groups of 64: up with relu^2 (epilogue 1), down to fp32 (epilogue 0).\n"
        "#define TF_EXPERT(EPI) template __global__ void tf_experts::expert_kernel<64, 1, EPI, 2, 4>(const __nv_bfloat16*, \\\n"
        "    int, int, const uint4*, int, int, const int*, const int*, const int*, void*, int, float);\n"
        "TF_EXPERT(1)\n"
        "TF_EXPERT(0)\n",
    ),
    "experts_prefill.cu": (
        "src/tensorfold/cuda/experts_prefill.cu", 1, 153, (3, 4, 7), (11, "tf_experts_prefill"),
        "// Prefill form for groups of 64: up with relu^2 (epilogue 1), down to bf16 (epilogue 3).\n"
        "#define TF_PREFILL(EPI) template __global__ void tf_experts_prefill::prefill_kernel<64, 1, EPI, 2, 2, 4>( \\\n"
        "    const __nv_bfloat16*, int, int, const uint4*, int, int, const int*, const int*, const int*, void*, int, float);\n"
        "TF_PREFILL(1)\n"
        "TF_PREFILL(3)\n",
    ),
    "experts_pack.cu": (
        "src/tensorfold/cuda/experts_pack.cu", 1, 42, (3, 4, 6), (8, "tf_experts_pack"),
        "// Groups of 64 inputs.\n"
        "template __global__ void tf_experts_pack::pack_kernel<2>(const uint32_t*, const uint16_t*, const uint16_t*,\n"
        "    uint32_t*, int, int, int, int);\n",
    ),
    "prefill_attention.cu": (
        "src/tensorfold/cuda/kernels/prefill_attention.cu", 1, 250, (3, 6), (11, "tf_prefill_attention"),
        "// Head dim 128, eight warps, eight query heads a block, eight staging slots (Nemotron's 16 heads a KV head).\n"
        "template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 8, 8>(const __nv_bfloat16*,\n"
        "    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float);\n",
    ),
    "scan_rows.cu": ("src/tensorfold/families/nemotron_h/cuda/scan_rows.cu", 1, 90, (3, 4), (8, "tf_scan_rows"), ""),
}


def render(name: str) -> tuple[str, str]:
    """The copy's expected text and the sha256 of the exact source lines it came from."""

    src, first, last, drop, ns, footer = COPIES[name]
    lines = (ROOT / src).read_text().splitlines(keepends=True)
    if last > len(lines):
        raise SystemExit(f"{src} has {len(lines)} lines, fewer than the copy's {last}; update copies.py")
    taken = lines[first - 1:last]
    digest = hashlib.sha256("".join(taken).encode()).hexdigest()
    note = ", comments and ATen includes dropped" if drop else ", comments dropped"
    out = [f"// Device code of {src} (lines {first}-{last}{note}), checked by zig/tests/cuda/copies.py.\n"]
    for number, line in enumerate(taken, start=first):
        if line.lstrip().startswith("//"):
            continue                # comments stay in the source; the copy keeps code only (same SASS)
        if number in drop:
            if "ATen" not in line and "c10" not in line and "torch/" not in line:
                raise SystemExit(f"{src}:{number} is no longer an ATen include; update copies.py")
            continue
        if ns is not None and number == ns[0]:
            if line != "namespace {\n":
                raise SystemExit(f"{src}:{number} is no longer the anonymous namespace; update copies.py")
            line = f"namespace {ns[1]} {{\n"
        out.append(line)
    if ns is not None:
        out.append(f"}} // namespace {ns[1]}\n\n")
    out.append(footer)
    return "".join(out), digest


def main() -> int:
    write = "--write" in sys.argv
    bad = 0
    for name in COPIES:
        text, digest = render(name)
        path = ROOT / "zig/kernels/cuda" / name
        if write:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} from source lines sha256 {digest}")
        elif not path.exists() or path.read_text() != text:
            print(f"DRIFT {path.relative_to(ROOT)}: differs from its source; rerun with --write and re-prove its bits")
            bad += 1
        else:
            print(f"ok {name} (source lines sha256 {digest})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

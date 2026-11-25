"""Drop-in CLI replacement for `python -m mlx_lm.convert` + Native Metal GPU TFQ3 quantizer (Zero Apple MLX)."""
from __future__ import annotations

import argparse
import sys

import mlx_tf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mlx-tf-convert",
        description="TensorFold mlx_tf bare-metal model quantizer: MLX affine (2..8-bit) & Native TFQ3 (3/3.5/4/5-bit, zero Apple MLX).",
    )
    p.add_argument("--hf-path", required=True, help="Local directory or HuggingFace repo ID of raw BF16/FP16/F32 model.")
    p.add_argument("--mlx-path", default="mlx_model", help="Destination directory for quantized checkpoint.")
    p.add_argument("--format", choices=("affine", "tfq3"), default="affine", help="Output format: 'affine' (MLX safetensors) or 'tfq3' (native .tffpw).")
    p.add_argument("-q", "--quantize", action="store_true", default=True, help="Quantize weights (default: True).")
    p.add_argument("--q-bits", "--bits", dest="q_bits", type=int, default=4, help="Bit width: 2..8 for affine, 3/4/5 for tfq3 (default: 4).")
    p.add_argument("--q-group-size", "--group-size", dest="q_group_size", type=int, default=64, help="Group size for affine: 32, 64, or 128 (default: 64).")
    p.add_argument("--profile", default="", help="TFQ3 bit-allocation profile: '', '3b', '3.5b', '4b', or '5b'.")
    p.add_argument("--head-bits", type=int, default=0, help="TFQ3 lm_head bit width (0 = auto).")
    p.add_argument("--keep-head-bf16", action="store_true", help="Keep lm_head in lossless BF16 for TFQ3.")
    args = p.parse_args(argv)

    out_dir = mlx_tf.convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        format=args.format,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        profile=args.profile,
        head_bits=args.head_bits,
        quantize_head=not args.keep_head_bf16,
    )
    fmt_label = f"TFQ3 ({args.profile or f'{args.q_bits}b'})" if args.format == "tfq3" else f"{args.q_bits}-bit g{args.q_group_size} affine"
    print(f"Saved TensorFold mlx_tf {fmt_label} checkpoint to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

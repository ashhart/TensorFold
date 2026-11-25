"""CLI drop-in for `python3 -m mlx_lm.server` backed by TensorFold `mlx_tf` (Zero Apple MLX)."""
import argparse
from . import load


def main():
    ap = argparse.ArgumentParser(description="TensorFold mlx_tf.server — Bare-Metal OpenAI + VLM + /v1/stream-parallel server")
    ap.add_argument("--model", required=True, help="Local model directory or HuggingFace repo ID")
    ap.add_argument("--draft", default=None, help="Optional DFlash2 drafter directory or HF repo ID")
    ap.add_argument("--cross-draft", default=None, help="Optional smaller draft model")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--max-streams", type=int, default=16)
    ap.add_argument("--bf16", action="store_true", help="Keep raw BF16/FP16 weights lossless")
    ap.add_argument("--force-simd", action="store_true", help="Force Metal 3.1 SIMD kernels on M5+")
    args = ap.parse_args()

    model, _ = load(
        args.model,
        draft_model=args.draft,
        cross_draft_model=args.cross_draft,
        max_context=args.max_context,
        max_streams=args.max_streams,
        force_simd=args.force_simd,
        keep_bf16=args.bf16,
    )
    try:
        model.serve(port=args.port, host=args.host, model_name=args.name)
    finally:
        model.close()


if __name__ == "__main__":
    main()

"""CLI drop-in for `python3 -m mlx_lm.generate` backed by TensorFold `mlx_tf` (Zero Apple MLX)."""
import argparse
from . import generate, load


def main():
    ap = argparse.ArgumentParser(description="TensorFold mlx_tf.generate — Bare-Metal Apple Silicon LLM & VLM CLI")
    ap.add_argument("--model", required=True, help="Local model directory or HuggingFace repo ID")
    ap.add_argument("--draft", default=None, help="Optional DFlash2 drafter directory or HF repo ID")
    ap.add_argument("--cross-draft", default=None, help="Optional smaller draft model (e.g. Qwen3.5-0.8B)")
    ap.add_argument("--prompt", "-p", required=True)
    ap.add_argument("--image", "-i", action="append", default=None, help="Image file path, file:// URI, or data: URI (repeatable)")
    ap.add_argument("--max-tokens", "-m", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--stop", action="append", default=None, help="Stop sequence (repeatable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Keep raw BF16/FP16 weights lossless instead of on-the-fly 4-bit")
    ap.add_argument("--force-simd", action="store_true", help="Force Metal 3.1 SIMD kernels on M5+")
    args = ap.parse_args()

    with load(
        args.model,
        draft_model=args.draft,
        cross_draft_model=args.cross_draft,
        max_context=args.max_context,
        force_simd=args.force_simd,
        keep_bf16=args.bf16,
    )[0] as model:
        tok = load_tokenizer(model.model_dir)
        if args.image:
            generate(
                model,
                tok,
                prompt=args.prompt,
                images=args.image,
                use_chat_template=not args.no_chat,
                enable_thinking=args.think,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=args.stop,
                verbose=True,
            )
        else:
            prompt = (
                args.prompt
                if args.no_chat
                else tok.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    add_generation_prompt=True,
                    enable_thinking=args.think,
                )
            )
            generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=args.stop,
                verbose=True,
            )


def load_tokenizer(model_dir: str):
    from . import MlxTfTokenizer
    return MlxTfTokenizer(model_dir)


if __name__ == "__main__":
    main()

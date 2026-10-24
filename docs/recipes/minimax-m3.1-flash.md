# MiniMax-M3.1-Flash on Metal (one 256 GB M5 Ultra)

The `minimax_m3` family serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a
428B-A23B mixture of experts with block-sparse attention (a lightning indexer picks each row's key blocks) and a
CLIP vision tower, on the lane host: tools, structured output, thinking on or off, images, the prompt cache, two
streams, and drafts from the release's DSpark head or incoai's DFlash2.

MiniMax's release stores its experts in NVFP4. The family reads MLX-style affine groups instead, from two public
conversions:

| checkpoint | experts | on disk | window on 256 GB |
| --- | --- | ---: | ---: |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34) | 4-bit (g128) on 26 of 60 layers, 3-bit (g64) on the rest | 203.7 GiB | 65,536 |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k) | 4-bit on 3 layers, 3-bit on the rest | 189.2 GiB | 262,144 |

Attention, the shared expert, the dense layers and the head are 8-bit in both; both carry the DSpark and DFlash2
weights. The conversion and its checks are in the
[serving recipe](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra).

## Run

```sh
hf download BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34 --revision 136e775c89e996cac20c05befbcb26886ad24782 \
  --local-dir ~/models/MiniMax-M3.1-Flash-MLX-mix34
MINIMAX_LOAD_LIMIT_GB=226 tensorfold serve ~/models/MiniMax-M3.1-Flash-MLX-mix34 \
  --context 65536 --parallel 2 --prompt-cache-gib 4 --vision
```

For mix256k (revision `c7defa69b0e8b57426a9945eb3f9eb91cfef9f8c`), `--context 262144`.

- `MINIMAX_LOAD_LIMIT_GB` bounds weights, scratch and caches together. The default (75% of RAM) holds the weights
  but no caches, so set it; 226 leaves about 4 GiB free at a full 262,144-token prompt on mix256k.
- `MINIMAX_KV_CACHE`: `int8` (the default; a scale a position and head, 77 KiB a token across the 60 layers) or
  `bf16` (135 KiB a token).
- `MINIMAX_DRAFTER`: `dflash2` (the default) or `dspark`; `--no-drafts` decodes plainly.
- Thinking off ends the prompt with `</mm:think>`, as the model does when it chooses not to think. Tool calls use
  MiniMax's `]<]minimax[>[` markup, parsed into OpenAI `tool_calls`; `tool_choice: required` or a named tool is
  constrained in that markup (xgrammar's `minimax_m3_xml`).
- Images: at most 576 tokens an image (672 x 672, the release processor's `max_pixels`); larger images are scaled
  down. Each image is its own tower pass. No video.

## Checks (Apple M5 Ultra, 256 GB, macOS 27)

- Four layers against transformers' own model: KL at most 0.005. The full model against an MLX reference at the same
  weights: KL 0.015.
- 1,024 rows teacher-forced against the NVFP4 release, int8 cache: mix34 KL 0.250 (80.0% top-1), mix256k 0.284
  (78.2%).
- Drafted replies equal plain ones (DFlash2 and DSpark); windows equal one-row steps bit for bit; two streams'
  replies equal their runs alone; replies are equal with the prompt cache on and off.
- A needle at 251,713 tokens on mix256k (545 s cold). The tower against transformers': relative error 0.030 to fp32
  (torch bf16: 0.039).
- Decode, one stream: 43-44 tokens a second plain, 42-66 drafted (prose to code). Cold prompts: about 1,200 tokens
  a second at 8k, 1,000 at 32k, 700 at 128k.

## Not covered

- Only the M5's tensor units were checked; nothing smaller than 256 GB holds the weights.
- The family's Metal kernels are not registered with the kernel contract yet.

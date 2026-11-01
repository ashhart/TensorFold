# MiniMax-M3.1-Flash on Metal (one 256 GB M5 Ultra)

The `minimax_m3` family serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a
428B-A23B mixture of experts with block-sparse attention (a lightning indexer picks each row's key blocks) and a
CLIP vision tower, on the lane host: tools, structured output, thinking on or off, images, the prompt cache, two
streams, and drafts from the release's DSpark head or incoai's DFlash2.

MiniMax's release stores its experts in NVFP4 and the rest in MXFP8; as published it takes about 248 GB, more than a
256 GB Mac can give a model. The family reads MLX-style affine groups instead, from two community checkpoints:
lossy conversions of the release, published with its own `config.json` (`minimax_m3_vl`, the language model under
`text_config`), the MLX `quantization` block in place of the release's, and the drafters' weights:

| checkpoint | experts | on disk | window on 256 GB |
| --- | --- | ---: | ---: |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34) | 4-bit (g128) on 26 of 60 layers, 3-bit (g64) on the rest | 203.7 GiB | 32,768 |
| [`BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k`](https://huggingface.co/BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k) | 4-bit on 3 layers, 3-bit on the rest | 189.2 GiB | 147,456 |

Attention, the shared expert, the dense layers and the head are 8-bit in both; both carry the DSpark and DFlash2
weights. Against the release's own weights (1,024 positions teacher-forced, bf16 cache) the conversion costs a mean KL
of 0.244 (80.7% top-1) on mix34 and 0.295 (78.4%) on mix256k. Everything the engine does from the converted weights on
is exact. The key/value cache is bf16: under the 226 GB limit below its caches fit 150,400 tokens on mix256k and 37,376
on mix34. The
[serving recipe](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra) has the conversion's studies.

TensorFold's code is Apache-2.0. MiniMax-M3.1-Flash's weights, and these checkpoints made from them, have the MiniMax
Community License: free for non-commercial use; commercial use must show "Built with MiniMax M3.1-Flash" and notify
MiniMax, with its written authorization above US$20 million yearly revenue; its appendix lists prohibited uses.
incoai's DFlash2 drafter, carried in both checkpoints, is Apache-2.0, as its
[model card](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2/blob/ac07a7c1ecaae26a7ace595756124dca974b432d/README.md)
states at the revision converted (`ac07a7c1`); the release's DSpark head is under MiniMax's licence with the rest.

## Run

```sh
hf download BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34 --revision 9524999057e0d277d26438606e033e2191fb465c \
  --local-dir ~/models/MiniMax-M3.1-Flash-MLX-mix34
MINIMAX_LOAD_LIMIT_GB=226 tensorfold serve ~/models/MiniMax-M3.1-Flash-MLX-mix34 \
  --context 32768 --parallel 2 --prompt-cache-gib 4 --vision
```

For mix256k (revision `916e7ca1044f1c3e7566163096243c940327224d`), `--context 147456`.

- `MINIMAX_LOAD_LIMIT_GB` bounds weights, scratch and caches together. The default (75% of RAM) holds the weights
  but no caches, so set it; at 226 a 145,203-token prompt on mix256k peaked at 210 GiB wired.
- The cache takes 135 KiB a token across the 60 layers; `--parallel 2` streams share it.
- `MINIMAX_DRAFTER`: `dflash2` (the default) or `dspark`; any other value is refused. `--no-drafts` decodes plainly.
- Thinking off ends the prompt with `</mm:think>`, as the model does when it chooses not to think. Tool calls use
  MiniMax's `]<]minimax[>[` markup, parsed into OpenAI `tool_calls`; `tool_choice: required` or a named tool is
  constrained in that markup (xgrammar's `minimax_m3_xml`).
- Images: at most 576 tokens an image (672 x 672, the release processor's `max_pixels`); larger images are scaled
  down. Each image is its own tower pass. No video.

## Checks (Apple M5 Ultra, 256 GB, macOS 27)

- Four layers against transformers' own model: KL at most 0.005. The full model against an MLX reference at the same
  weights: KL 0.015.
- 1,024 rows teacher-forced against the release's NVFP4 weights: mix34 KL 0.244 (80.7% top-1), mix256k 0.295 (78.4%).
- Drafted replies equal plain ones (DFlash2 and DSpark); windows equal one-row steps bit for bit; two streams'
  replies equal their runs alone; replies are equal with the prompt cache on and off.
- A needle at 145,203 tokens on mix256k (115 s cold; 210 GiB wired at the peak). The tower against transformers':
  relative error 0.030 to fp32 (torch bf16: 0.039).
- Decode with DFlash2, one stream (prose / code / structured / list): mix256k 44.0 / 55.8 / 47.8 / 50.9 tokens a
  second, mix34 41.2 / 61.3 / 49.3 / 55.3. Cold prompts on mix256k: 1,478 tokens a second at 8,403 tokens, 1,486 at
  16,598 and 1,484 at 30,929 (4,096-row chunks); 1,379 at 128k before the kernel fixes.
- The family's 50 Metal kernels are registered with the kernel contract: 50 pass gates 1-4.

## Not covered

- Only the M5's tensor units were checked; nothing smaller than 256 GB holds the weights.
- The release itself: serving it exactly needs native NVFP4/MXFP8 kernels on Metal and a Mac that holds about 248 GB.
- Video input.

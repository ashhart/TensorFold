# Qwen3.6-35B-A3B

The `qwen3_5_moe` family serves Qwen3.6-35B-A3B on Apple Silicon and on one NVIDIA GPU. Its layers are the 27B's
(Gated DeltaNet and gated full attention, every fourth layer attention) with routed experts in place of the dense
MLP, and on both backends it drafts with the checkpoint's own MTP layer.

## Checkpoint

```bash
tensorfold pull TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --name bench
```

Tested revision: `81169a9bc511a27c1b4eedb77a2cd98ced431847` (20.9 GB). Its weights are
`mlx-community/Qwen3.6-35B-A3B-4bit` (MLX affine 4-bit in groups of 64, routers and the shared-expert gate at
8 bits), converted from `Qwen/Qwen3.6-35B-A3B`; that conversion drops the MTP layer, which ships beside it as
`mtp-4bit.safetensors` (converted by the same rules: experts split into gate and up projections, norms
shifted by one, projections 4-bit, gates 8-bit). The mlx-community folder serves the same way once that file
is placed in it. After the weights, one Spark keeps about 75 GB for caches: attention holds 20 KB a token and
DeltaNet 63 MB a stream.

`--no-drafts` or request field `"draft": false` selects serial decoding, the reference drafted output equals.

## Mac execution

Macs verify on the dense family's row decoder (row-exact matmuls and attention, routed experts through MLX's
grouped matmul), every window checked at load against one-row steps. Drafts come from the MTP layer:

- The head reads each kept row's final normed state with the next token's embedding, and draws its drafts with
  the target's keyed rule over a 79,616-id draft vocabulary. Drafts only choose the rows a round verifies, so
  every reply equals its `"draft": false` run.
- It absorbs every prompt row into its own attention cache; a long prompt's first token comes no later.
- Each round's depth comes from this Mac's costs: the verify windows timed at load, then the stream's measured
  rounds, against each depth's landing rate in the stream. Where no depth beats a plain round by 5%, a round
  verifies the pending token alone; a probe draft follows 8 plain rounds, then 16, 32 and on to 128 while drafts
  keep losing. `--mtp-drafts N` caps the chain (default 4).
- Concurrent streams absorb their kept rows in one head pass and draft level by level, each by its own rule.
- `mlx-community/Qwen3.6-35B-A3B-4bit` has the same weights without the MTP file. It drafts with the file from
  `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` once that file is in the Hugging Face cache
  (`hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP mtp-4bit.safetensors`, 0.5 GB) or named by
  `TF_QWEN36_MTP=<file>`; without it, it decodes one token a round and says so at startup.
  `--drafter z-lab/Qwen3.6-35B-A3B-DFlash` drafts with DFlash v1 instead.

On an M3 Ultra (60-core GPU, MLX 0.32.3), greedy, median of three after a warm-up. mlx-vlm 0.7.4 serves the same
weights with `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` (two drafts a round). The short prompts are about 85 tokens
with 384 generated; the long one is 28,400 tokens of Python's `typing.py` with 128 generated.

| tok/s | Short prose | Short code | Short, thinking on | 28,400-token prompt | First token at 28,400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold, MTP (default) | 173.4 | 244.7 | 196.4 | 124.0 | 15.9 s |
| TensorFold, `--no-drafts` | 118.5 | 119.2 | 119.3 | 92.7 | 15.9 s |
| TensorFold, `--drafter z-lab/Qwen3.6-35B-A3B-DFlash` | 156.1 | 288.2 | 183.1 | 87.5 | 15.7 s |
| mlx-vlm 0.7.4, MTP | 111.9 | 141.8 | 120.7 | 85.6 | 14.8 s |
| mlx-vlm 0.7.4 | 106.4 | 105.7 | 107.5 | 88.0 | 14.7 s |

Each TensorFold reply has the same token SHA-256 in all three modes. MTP rounds verify 4 to 5 rows and keep about
three tokens. DFlash's 16-row blocks keep more on short code, and fall behind plain decoding at the long prompt.

The chat template thinks by default, as on vLLM: replies reason in `reasoning_content` before the answer in
`content`, and `max_tokens` counts both. The server says so at startup, and prints a warning when a reply reaches
`max_tokens` before it leaves its think block (its `content` is empty). `--no-thinking`, or
`"chat_template_kwargs": {"enable_thinking": false}` in a request, turns thinking off.

## CUDA execution

CUDA reads this model's MLX 4-bit checkpoint only; NVFP4 and EXL3 exports of it are not read yet. Prompts take bf16
activations by default: 0.90-0.96x the FP8 prompt path from 2k to 128k, and 1.23-1.89x vLLM on NVIDIA's NVFP4 export
from 2k to 64k. `--prefill-fp8` restores the FP8 path ([prompt precision](cuda.md#prompt-precision)).

Verify windows run the 27B's shared kernels (4-bit matmul, DeltaNet tree and replay, tree attention) with
routed experts from `tensorfold/cuda/experts.py`: the router's top 8 of 256 by fp32 logit (ties to the lower
id), weights renormalized over the eight, the shared expert as expert 256 with a sigmoid gate, and the slots
summed in slot order. Each (row, expert) pair gets the same bits in any window, so a drafted row equals the
serial step.

Each round first verifies a copied continuation when the context repeats eight or more tokens, and otherwise
a chain of up to three MTP drafts: the head reads the target's final normed state and the next token's
embedding, drafts with the target's keyed sampling rule over a draft vocabulary, and a chain stops after a
draft it gives under 30%. Decoding runs in buffers the engine keeps between requests, so verify chains and
head steps replay CUDA graphs captured once per width and context bucket.
Prompts prefill in chunks; the head absorbs every prompt row but the last. States are kept at the second
message's start, the last assistant turn's start and prompt ends, so a prompt sharing a system block or
extending a conversation resumes there with a fresh prefill's bits.

### Concurrent requests

```bash
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --parallel 8 --name bench
```

`--parallel N` decodes up to N requests in shared rounds, and every reply equals the same request served alone
and its `"draft": false` run. A round verifies every stream's MTP chain or copied continuation in one forward;
the head absorbs every stream's kept rows in one call, then the chains advance a step at a time for all streams
by the one-stream rule, so a stream drafts what it drafts alone. A new prompt prefills 1,024 tokens a round
while the others decode, states are kept at message starts and prompt ends, and each stream's caches are sized
once, at admission. Startup admits N full prompt/reply windows, three kept prompt ends and the graph buffers
before loading; an explicit `--context` that does not fit is refused with the window that does. Rounds over
several streams run eagerly, as the 27B's do; a stream decoding alone replays the one-stream CUDA graphs, so a
lone request runs as fast as without `--parallel`.

On one RTX PRO 6000 Blackwell Max-Q (NGC 26.07, torch 2.13, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
checkpoint revision 81169a9), with 8 client threads over HTTP:

| | `--parallel 1` | `--parallel 4` | `--parallel 8` |
| --- | ---: | ---: | ---: |
| Label JSON: 32 requests, public-domain passages, up to 1,410 tokens, default sampling | 904 tok/s | 1,255 tok/s | 1,476 tok/s |
| p50 / p95 latency | 8.6 / 9.1 s | 5.9 / 8.1 s | 5.1 / 7.7 s |
| Chat: 48 requests, up to 256 tokens, half greedy, half seeded | 427 tok/s | 859 tok/s | 1,094 tok/s |
| p50 / p95 latency | 3.3 / 3.9 s | 1.6 / 2.5 s | 1.2 / 2.0 s |
| Peak memory (nvidia-smi) | 22.3 GiB | 22.1 GiB | 22.9 GiB |

Every reply's token SHA-256 is the same at each N and with `"draft": false`. The label replies repeat their JSON,
so copied continuations keep 9.3 tokens a round per stream; chats keep 2.9. One client at a time gets 912 tok/s
on the label requests at `--parallel 8`, as a lone stream replays the graphs.

## Measurements

One DGX Spark (GB10) in NVIDIA's `pytorch:26.07-py3` container, checkpoint revision 81169a9, against vLLM serving
`nvidia/Qwen3.6-35B-A3B-NVFP4` with MTP=3 on the same Spark (`vllm/vllm-openai`, prefix caching, chunked prefill,
`--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.60`).

Decode with the [public benchmark command](README.md#measurements); drafted replies equal `"draft": false` ones:

| | Code sampled | Chat sampled | Code greedy | Chat greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold | 179.4 tok/s | 141.3 tok/s | 166.6 tok/s | 162.8 tok/s |
| vLLM, MTP=3 | 120.6 tok/s | 100.6 tok/s | 122.1 tok/s | 117.0 tok/s |

Serial decoding (`--no-drafts`) runs at 86-88 tok/s. A round verifies up to four rows, and each row brings its
own eight experts, so a round reads about twice the bytes of one serial step; longer drafts pay off only when
most of their rows are kept.

Cold prefill with `tools/prefill_cold.py` (chat prompts from the Python standard library at exact rendered
lengths, a unique first line each so nothing resumes; median time to first token of three):

| Prompt tokens | 2,048 | 8,192 | 16,384 | 32,768 | 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold | 7,161 tok/s | 7,353 tok/s | 6,541 tok/s | 5,257 tok/s | 3,688 tok/s |
| vLLM, MTP=3 | 5,907 tok/s | 5,881 tok/s | 5,090 tok/s | 3,951 tok/s | 2,693 tok/s |

The server process peaks at 31.3 GiB (nvidia-smi) during the 65,536-token prompts; vLLM holds its memory
reservation, 72.9 GB at 0.60.

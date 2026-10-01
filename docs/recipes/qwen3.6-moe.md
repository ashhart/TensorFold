# Qwen3.6-35B-A3B

The `qwen3_5_moe` family serves Qwen3.6-35B-A3B on one NVIDIA GPU. Its layers are the 27B's (Gated DeltaNet
and gated full attention, every fourth layer attention) with routed experts in place of the dense MLP, and it
drafts with the checkpoint's own MTP layer.

## Checkpoint

```bash
tensorfold pull Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP
tensorfold serve Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP --name bench
```

Tested revision: `81169a9bc511a27c1b4eedb77a2cd98ced431847` (20.9 GB). Its weights are
`mlx-community/Qwen3.6-35B-A3B-4bit` (MLX affine 4-bit in groups of 64, routers and the shared-expert gate at
8 bits), converted from `Qwen/Qwen3.6-35B-A3B`; that conversion drops the MTP layer, which ships beside it as
`mtp-4bit.safetensors` (converted by the same rules: experts split into gate and up projections, norms
shifted by one, projections 4-bit, gates 8-bit). The mlx-community folder serves the same way once that file
is placed in it. After the weights, one Spark keeps about 75 GB for caches: attention holds 20 KB a token and
DeltaNet 63 MB a stream.

`--no-drafts` or request field `"draft": false` selects serial decoding, the reference drafted output equals.

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
tensorfold serve Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP --parallel 8 --name bench
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

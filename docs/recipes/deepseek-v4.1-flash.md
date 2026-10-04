# DeepSeek-V4.1-Flash

The `deepseek_v41` family serves DeepSeek-V4.1-Flash on CUDA over two DGX Sparks (GB10, 128 GB each), one rank a
machine, from the EXL3 export `sfxnz/DeepSeek-V4.1-Flash-EXL3` at revision
`982b70452f399814f56b46272fd30394ae10d58c`. That revision keeps DeepSeek's own bytes everywhere but the routed
experts: the non-routed weights in DeepSeek's FP8 (e4m3 with a power-of-two scale per 32x32 block), the LM head in
MXFP8, the three DSpark draft stages' experts in DeepSeek's MXFP4, and the Engram tables. The 384 routed experts of
the 40 layers are EXL3 at 2 bits (mcg codebook). Packages: `src/tensorfold/families/deepseek_v41/` (host side, no
torch) and its `cuda/` (the engine, Triton kernels on the shared `cuda/exl3`, `Mx8Linear`, NVFP4 experts and comm).

## Download, on both machines

```bash
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 982b70452f399814f56b46272fd30394ae10d58c
```

The command prints the snapshot directory, `MODEL_DIR` below: 48 shards, 357,486,881,279 bytes (333 GiB).
`tensorfold pull` takes the repository's `main`, which holds only the model card, so name the revision. Both
machines need the whole revision on local disk, shards 47 and 48 included: they hold the two Engram tables
(94.6 GiB each), and each rank reads its own half of every row (hash columns 0-11 on rank 0, 12-23 on rank 1)
from its own disk with `pread`, through the page cache. Engram rows never cross the link and are never resident.

## Serve

Start the [two-rank container](../../RUNBOOK.md#nvidia-gpus) on each machine, with `NCCL_SOCKET_IFNAME` and
`NCCL_IB_HCA` for the link between them, then rank 1 first:

```bash
tensorfold serve MODEL_DIR --tp 2 --rank 1 --master 192.0.2.1 --no-thinking
tensorfold serve MODEL_DIR --tp 2 --rank 0 --master 192.0.2.1 --no-thinking \
    --name deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --port 8080
```

`192.0.2.1` stands for rank 0's address on the link. The ranks compare their settings at startup (context,
drafting, the Engram files' sizes and headers) and refuse to serve, naming both values, when they differ. The first
start builds four CUDA extensions (about two minutes); later starts load in about 40 s from a warm page cache.

| Flag | Effect |
| --- | --- |
| `--tp 2 --rank R --master ADDR` | Required: one GPU a machine. One rank, or a separate `--drafter`, is refused. |
| `--context N` | The prompt-plus-reply window. Omitted, the whole 1,048,576-token window when it fits. |
| `--mtp-drafts D` | DSpark drafts a round, 1 to 5 (default 3); 0 decodes one token a round. |
| `--mtp-confidence P` | Draft fewer when unsure: stop before the first draft whose product of DSpark's confidence scores falls under `P` (at least one). |
| `--no-drafts` | The serial reference: DSpark is not loaded. |
| `--no-thinking`, `--reasoning-effort` | Request defaults, as on every family; a request's own switch wins. |
| `--parallel N` | Accepted and ignored with a note: requests run one at a time. |
| `--prefill-fp8` | Refused: prompt matmuls take bf16 activations. |

`TF_DSV41_CACHE_GIB` (default 3) sizes the device arena that keeps other conversations' prompt states and
`TF_DSV41_CACHE_ENTRIES` (default 8) how many it keeps; the startup log says when the window leaves less. Give both
ranks the same values.

## Memory

A rank holds 73.2 GiB of weights: its half of the routed experts (TP over each expert's intermediate dim, 1,152 a
rank), of the attention heads (32), of the shared expert, of the LM head's vocabulary and of DSpark; the embedding
and the indexers are whole on both. The caches, the RoPE tables and a prompt chunk's and a decode window's scratch
are sized at startup for the window and allocated once, as are the kept-prompt arena and every transient the
kernels take (a warm-up prompt and decode rounds run before serving), so serving allocates no device memory.

| Window | Startup line, rank 0 (rank 1 alike) | `free -h` available after a request, rank 0 / rank 1 |
| --- | --- | --- |
| 65,538 | `startup estimate 78.46 GiB within 99.59 GiB; ... allocated prompt/reply window 65538` | 28 / 31 GiB |
| default | `startup estimate 79.09 GiB within 99.58 GiB; native 1048576, allocated prompt/reply window 1048576` | 24 / 26 GiB |

The Engram reads fill the page cache, which admission counts as available. Across a cold 64k prompt and three
resumes of it, `torch.cuda.memory_reserved` stayed at 80.7 GiB on rank 0.

## Requests

- Prompts render through DeepSeek's own encoder (vendored, unmodified; the checkpoint has no Jinja template).
  Developer messages render as system messages, and tools join the first system message.
- Thinking follows `chat_template_kwargs.thinking` or `enable_thinking`, else the server default. An effort name
  turns thinking on, as on every family, so a client that sends `reasoning_effort: "low"` to a `--no-thinking`
  server gets thinking; send `chat_template_kwargs.thinking: false` (or `reasoning_effort: "none"`) to keep it off.
- Efforts: `low`, `high` and `max` are the encoder's budgets 50, 75 and 100 (`medium` is heard as `high`, `xhigh`
  as `max`), and an integer from 1 to 100 is the budget itself ([API](../api.md#reasoning)). vLLM's DeepSeek
  encoder maps the names to 25, 50 and 75, so a thinking prompt with a named effort renders differently there.
- Tool calls are DeepSeek's DSML blocks, parsed into OpenAI `tool_calls` with `tool_choice` `auto` or `none`. The
  calls arrive in the final chunk only: a streamed reply never shows the DSML block, and its calls come whole at
  the end, with `finish_reason: "tool_calls"`.
- Sampling defaults to DeepSeek's recommendation: temperature 1.0, `top_p` 0.95 and no top-k (`--top-k` or a
  request's `top_k` sets one). Every draw is TensorFold's keyed rule over both ranks' vocabulary halves. With top-k
  off, a row whose nucleus runs past each rank's 1,024 best candidates reads the rank's whole vocabulary half to
  the host, and `top_p` 1.0 does so for every row: exact, and slow until that draw runs on the device. On the
  same prompt at temperature 1, sampling took 69 ms a round with the default and 2 ms with `top_k` 20, and the
  reply decoded at 24.3 tok/s against 50.7.
- `ignore_eos`, `stop`, `seed`, `min_p` and `"draft": false` work as on the other CUDA families.
- Refused with HTTP 400: images, `response_format` and the `guided_*` and `structured_outputs` fields, `logprobs`,
  `tool_choice: "required"` or a named function, `thinking_budget` (and every thinking request on a server started
  with `--thinking-budget`), and `n` above 1. Both ranks decode a request to its end, so a budget's cut would decode
  the whole first reply before the budgeted one.

## Precision

Where TensorFold's sums differ from DeepSeek's reference (`inference/model.py` and `inference/kernel.py` of
deepseek-ai/DeepSeek-V4.1-Flash at `dba1be0`; the EXL3 export does not ship them, and source comments cite their
lines as `M:n` and `K:n`). None trades precision for speed; each is equal or higher precision, or a reorder, and each
changes bits.

| Operation | DeepSeek's reference | TensorFold |
| --- | --- | --- |
| FP8 projections | Activations quantized to FP8 (1x32, power-of-two scales) before each FP8 matmul | bf16 activations against the exact FP8 weights (`Mx8Linear`, the 32x32 scales repeated per row): higher |
| Row-parallel partials (`wo_b`) | Each rank's partial rounded to bf16, then an fp32 all-reduce | fp32 partials gathered and added in rank order: higher |
| Indexer scores | bf16, heads split over the ranks, a bf16 all-reduce | fp32, every head on both ranks: higher, and a row can keep a different 512th entry |
| Routed experts | FP8 activations x FP4 weights; the routing weight applied before `w2`, rounded to bf16 | fp16 activations x the EXL3 weights; the routing weight applied after `w2` in the fp32 combine: a reorder at equal or higher precision |
| Shared expert | Computed whole on every rank; its bf16 `w2` output added to the fp32 sum after the routed all-reduce | Inner dim split over the ranks; each rank's fp32 `w2` partial added into its fp32 MoE share, then the shares gathered and added in rank order: higher, and a reorder |
| LM head | fp32 logits of bf16 activations | bf16 activations against the MXFP8 weights, fp32 logits: equal |
| Window, compressed and index keys | FP8 / FP4 quantize-dequantize, trained with it | The same quantize-dequantize, stored as bf16 values (the same numbers packed storage would hold): equal |
| Decode indexer at even positions | Layers 2, 8 and 14 (and the layers that reuse their lists) score against layer 20's index keys when their own compressor emitted no entry | Every layer scores its own source's index keys, as the reference's prompt path does |

The last row is a defect of the reference's decode path that TensorFold does not reproduce: reproducing it would
make a decoded token's bits differ from the same prompt prefilled.

## Exactness

A drafted reply equals the same server's `"draft": false` reply, a resumed prompt equals a fresh one, and both ranks
emit the same tokens. Every kernel computes a row alone in an order fixed by the row's absolute position, so a row
gets the same bits alone or in a verify window of up to 6 rows (the target token and 5 drafts), eager or in a CUDA
graph, at any context. The prompt path and decode path take different kernels, chosen by the call site, never by
the row count, so a prompt's rows never depend on its chunking; their bits differ from decode's, so each request
keeps its prompt's state one token before its end, and a follow-up turn resumes there. DSpark drafts with the same
keyed draws (over each rank's 1,024 best candidates when top-k is off), and the target keeps a draft only when it
equals its own draw. Each row is drawn with its absolute position's key from candidates both ranks gather, so the
ranks pick the same tokens, keep the same rows and commit alike without a broadcast, and both decode every request
to `max_tokens` or an end token. Fp32 partials are gathered and added in rank order, and every collective, the
request header included, runs on both ranks in the same order.

Receipts on two Sparks at `--context 65538`: `tools/bench_concurrent.py --alone --serial` found every reply equal
to its solo run (48 of 48) and every solo run equal to its `"draft": false` run (10 of 10), at temperatures 1 and 0; a
request with tools, sent twice with the second `"draft": false`, returned the same `token_sha` and the same two calls
streamed and not, greedy and sampled; three prompts drafted and serial at temperatures 0 and 1, and the family's
default sampling drafted, serial and resent, gave equal `token_sha`; a cold 64k prompt and three resumes of it gave
equal replies. The two-rank check script (`tests/cuda/dsv41_tp_check.py`) gives the same logits hashes, step by
step, over NCCL on two Sparks as two threads on one GPU.

Quality against the vLLM recipe on the same weights, with vLLM's own repeat runs as the noise floor: teacher-forced
NLL 0.1371 over 40 passages from ten Project Gutenberg books (19,828 positions), against vLLM's 0.1389-0.1391;
TensorFold's repeat is bit-identical. Top-1 disagreement with vLLM is 1.08% and the median |dlogprob| 1.0e-4.
GSM8K 97/100 (vLLM 96 in its run on these weights, 97 in the recipe's baseline), MMLU 204/228 (vLLM 204 and 205),
tool calls 22/22 with exact arguments, needles 9/9 up to about 130k tokens.

## Speed

Two DGX Sparks over their direct cable, `--context 65538`, default drafting (3 drafts a round), against the vLLM
recipe for this checkpoint on the same machines and weights, measured in a separate session (not interleaved; vLLM's
sampled fibonacci-raw runs ranged from 40 to 65 tok/s). `tools/bench_openai.py` (64 tokens, median tok/s of 5):

| Prompt | Temperature | TensorFold | vLLM |
| --- | --- | ---: | ---: |
| fibonacci-raw | 1 | 41.8 | 47.3 |
| gpu-chat-no-think | 1 | 38.1 | 44.3 |
| fibonacci-raw | 0 | 56.4 | (its reply is end tokens) |
| gpu-chat-no-think | 0 | 41.3 | 43.1 |

The vLLM recipe's own decode cells (greedy, thinking off, 200 tokens, one stream), with TensorFold's decode tokens a
round (the 199 after the prompt's first token over its rounds) and vLLM's mean acceptance length, both with 3 drafts
a round:

| Cell | TensorFold | Tokens a round | vLLM | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| prose | 45.8 | 2.80 | 50.7 | 2.45 |
| structured | 64.2 | 3.90 | 84.3 | 4.00 |
| prose_long | 34.4 | 2.12 | 43.8 | 2.17 |

Drafting is not the gap: TensorFold keeps more tokens a round on prose and slightly fewer on structured and
prose_long. The gap is the round: about 61 ms against vLLM's 48, of which the host's Engram reads take 3.5-6 ms and
target sampling 2 ms (both on the critical path), and a serial token 35 ms of device time. The prose cell stops
naturally at 74 tokens, so most of its 200 measured tokens come after the end token.

Cold prompts, `tools/prefill_cold.py` (median of 3; vLLM ran the same messages):

| Prompt | 2k | 8k | 16k | 32k | 64k |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold, tok/s | 524 | 546 | 562 | 561 | 563 |
| vLLM, tok/s | 778 | 769 | 772 | 777 | 774 |

In a prompt chunk the routed experts decode each EXL3 tile again for every 16 rows.

## Not yet

Images (DeepSeek-V4.1's vision encoder), grammars and `response_format`, `logprobs`, concurrent streams, one GPU, a
Mac, and other exports of the model: the check refuses a BF16 LM head and EXL3 outside the routed experts, so
`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (EXL3 throughout) is refused.

# Qwen3.8 Flash Next

The `qwen4_exp` family has Gated DeltaNet, sparse attention, MoE, hyper-connections and hashed n-gram
embeddings. The supported checkpoint uses MLX affine 4-bit weights in groups of 32 and includes an MTP head.

```bash
tensorfold pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --name bench
```

On MLX, a supported conversion without the head runs without MTP drafting. On CUDA, pass `--no-drafts`
for such a conversion; the default positive draft depth otherwise refuses the missing head.

## MLX execution

N-gram tables stay in host file mappings when the checkpoint's model files exceed 75% of the GPU's
recommended working set. `TF_NGRAM_HOST=1` forces this mode; `TF_NGRAM_HOST=0` keeps the tables in MLX.
Startup admission uses the same choice as the loader and subtracts the mapped weights, scales and biases
from the checkpoint's size. For the named 4-bit checkpoint, about 29.8 GiB of its 105.4 GiB is mapped,
leaving a conservative 75.6 GiB resident-weight estimate. The default command selects host mode on an
M4 Max with 128 GiB, whose process budget is 89.6 GiB, including the 3 GiB process reserve.

Mapped pages still use RAM while cached. The loader prefetches them after its initial forwards; macOS
can reclaim them, and subsequent lookups may read from disk. The remaining weights must fit the MLX
budget, and the server sizes context from runtime cache and workspace needs. `TENSORFOLD_MEMORY_LIMIT_GB`
can lower or raise the default budget, capped by physical RAM and the GPU's recommended working set.
For example, `TENSORFOLD_MEMORY_LIMIT_GB=110` gives a 128 GiB M4 Max a 110 GiB process budget and
107 GiB for MLX. An explicit context must still fit the startup estimate; a larger budget does not
establish full-window inference or keep every mapped page resident.
After measuring shared rounds, the runtime releases the probes' rollback buffers before sizing prompt
memory, so those unused states do not reduce the available context.

Fused kernels handle hyper-connections, routing, experts, recurrence and sparse attention. Row-exact
projections and stable routing ties keep each verify row independent of the other rows. M5 GPUs use
the lane matmul; M1 through M4 use the per-row projection and hyper-connection kernels by default. Rejected tails
restore recurrent state, n-gram history and attention state, including incomplete pooled blocks.
The load-time row check disables drafting when windows do not reproduce serial steps.

The prefill path uses sparse selected-key attention, fused hyper-connections and n-gram lookups,
stacked DeltaNet projections and sorted expert rows. It submits bounded groups of layers to limit live
workspace. Compatible prefill matmul kernels check against MLX; unsupported paths use MLX's kernels.
`TF_FLASH_PREFILL=0` selects the reference prefill path for comparison.

Prefill and decode can round differently. The chunk planner uses detected assistant-message starts and
the second message when at least 256 tokens follow the previous chunk start, otherwise cutting after
the chunk chosen at startup: the largest of 8,192 (with tensor units), 4,096 and 2,048 tokens whose
working memory leaves room for 128K tokens of context, or the model's window if smaller. Where none
fits, the prompt path queues one layer at a time instead of two. Cold and resumed prompts use the same
rendered-token boundaries, and reuse starts only at these cuts. Templates without detected markers use
the same fixed chunks. A chunk's sorted expert rows reach MLX's gather in slices of at most 32,768
(its M5 kernel keeps row offsets in 16 bits through MLX 0.32.2). Snapshots include the prefill
path, resolved matmul route and GPU identity; changing arithmetic requires a fresh cache.

Load-time shared-forward checks compare each stream with its own call. A failed check limits forwards
to one stream, while successful checks allow the lane engine to combine requests.

## CUDA

On CUDA Flash Next serves NVFP4 and EXL3 checkpoints, and the MLX 4-bit checkpoint as the portable option: the same
files a Mac serves, and the only format two ranks and `--ple-on-ssd` read. `tensorfold serve` loads the checkpoint
you name; it picks none by itself. Use the [container setup](../../RUNBOOK.md#nvidia-gpus) for any of them. Prompts
take bf16 activations by default; what that costs against the FP8 prompt path (`--prefill-fp8`) depends on the
format ([prompt precision](cuda.md#prompt-precision)):

| Checkpoint | Weights | bf16 prompts against `--prefill-fp8` |
| --- | --- | --- |
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (`925d7be6`), a mirror of local-inference-lab's export | NVFP4 routed experts, MXFP8 elsewhere and in the n-gram table | the next row's kernels; not timed on its own |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (`7c4f1bc1`) | NVFP4 routed experts, MXFP8 elsewhere | 0.94-1.03x from 2k to 64k |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`7b719225`) | NVFP4 routed experts, bf16 elsewhere | unchanged: no FP8 prompt kernel |
| `turboderp/Qwen3.8-Flash-Next-exl3` (`3.05bpw_h5_ng5`) | EXL3 | unchanged: EXL3 prompts never took FP8 activations |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | MLX affine 4-bit | unchanged: its prompts were already bf16 |

TensorFold finds Mia-AiLab's export by its `model_type` (`qwen3_8_flash_next`) and serves it like
local-inference-lab's.

### NVFP4 checkpoints

The CUDA engine reads published ModelOpt NVFP4 exports as they ship, MTP head included, on one GPU:

```bash
tensorfold serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8080
```

| Checkpoint (revision) | Routed experts | DeltaNet, attention, shared expert | n-gram table |
| --- | --- | --- | --- |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (`7c4f1bc1`) | NVFP4 | MXFP8 | NVFP4 rows |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`7b719225`) | NVFP4 | bf16 | FP8 rows |
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (`925d7be6`), a mirror of local-inference-lab's export | NVFP4 | MXFP8 | MXFP8 rows |

The loader reads each linear by its tensors. An NVFP4 weight is an E2M1 code times its e4m3 scale (a block of 16
inputs) times the tensor's fp32 `weight_scale_2`; an MXFP8 weight is an e4m3 byte times a power of two (a block of
32). Both products fit bf16 exactly, so decode multiplies them in bf16 MMAs, adds each block's products times its
scale in block order and applies the tensor's scale once; the K split depends on the shape alone, so drafted
windows keep serial decoding's bits. Prompts run the MXFP8 linears on bf16 rows and the stored bytes, each byte
times its power of two exact in bf16 and one fp32 sum over the inputs (`--prefill-fp8`: the FP8 prompt matmul).
Tests check the kernels against an fp64 reference built by an independent numpy dequantizer
(`tensorfold/cuda/nvfp4/format.py`). Both exports store their RMSNorm weights centred (gamma - 1), and the loader
tells centred from uncentred norms by their stored values. An n-gram table's shards must share one layout, or the
load stops.

Block-scaled FP8 linears (ModelOpt `FP8_PB_WO`, the DeepSeek-style layout: e4m3 bytes and an fp32 `weight_scale_inv`
per 128x128 block) are read too. Decode keeps the e4m3 bytes in the FP8 GEMM's fragment order and each (64 inputs,
column)'s block scale as fp32; the lane matmul multiplies a 64-input stage in bf16 MMAs (e4m3 fits bf16 exactly) and
adds the stage's products times its scale in stage order, so the stored weight is exact and rows stay independent of
the row count, as for the other formats. Prompts take the same lane matmul (bf16 activations, the stored bytes, fp32
sums); `--prefill-fp8` runs the FP8 prompt matmul over the same bytes with the block scales as bf16 group scales. A
projection stack that mixes block FP8 with bf16 (`in_proj_b` and `in_proj_a` beside `in_proj_qkv` and `in_proj_z`;
the indexer's projection beside q/k/v) runs each part on its own kernel into its columns. A block-FP8 `lm_head`
stays on the lane matmul with its stored bytes, for decode rows and a prompt's head rows; the draft head's rows are
dequantized and requantized to 4 bits, as for every NVFP4 checkpoint (drafts only). Checked on a local ModelOpt
export with NVFP4 experts, block-FP8 DeltaNet and attention projections, an FP8 n-gram table and NVFP4 MTP experts:
drafted replies equal `"draft": false` ones (six pairs, 2k-16k-token prompts, greedy and sampled), resumed prompts
equal fresh ones, and each reply of 2 and 4 concurrent requests equals the same request alone. On one Spark, one
request, it decodes 6-27% faster than the same weights dequantized to bf16 on the bf16 path (code 59.2 against 49.1
tok/s greedy, chat 36.6 against 34.6 greedy and 41.6 against 32.8 sampled).

The routed experts run on a grouped NVFP4 kernel that reads the step's routing plan on the GPU, so a decode graph
captured for one step's experts replays another step's. A test decodes a checkpoint whose expert picks change every
step with graphs on and off and compares the tokens; on local-inference-lab's and RadixArk's exports, replies with
the decode graphs equal replies without them.

local-inference-lab's and RadixArk's exports were served with `tensorfold serve` and checked: drafted replies equal
`"draft": false` ones (nine pairs, 2k-16k-token prompts, greedy and sampled), resumed prompts equal fresh ones, and
each reply of concurrent requests (`--parallel 4`) equals the same request alone. Mia-AiLab's mirror loads and serves
through the same code; its replies have not been checked on their own.

Not supported here:
- `--tp 2` and `--ple-on-ssd` on an NVFP4 checkpoint stop at startup: two ranks read the MLX checkpoint, and the
  NVFP4 exports' tables stay memory-mapped.
- `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` (186 GB; bf16 linears and n-gram table) has the RadixArk layout apart
  from its bf16 table, which the loader reads, but the whole checkpoint has not been loaded or served here, so it
  is not listed as tested.

### EXL3 checkpoints (experimental)

The CUDA engine also serves EXL3 packs of Flash Next (`quant_method: exl3`) on one GPU through the shared EXL3
module ([EXL3 weights](exl3.md)): any codebook, and a width per tensor, so mixed-K packs whose experts range from
2 to 8 bits within a layer load as they are. turboderp publishes a 3.05 bpw pack on the branch `3.05bpw_h5_ng5`.
Download that branch, then serve the folder:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-Flash-Next-exl3', revision='3.05bpw_h5_ng5', local_dir='flashnext-exl3-3.05bpw')"
python -m tensorfold.cuda.exl3.inspect flashnext-exl3-3.05bpw   # bits per tensor, from the headers
tensorfold serve flashnext-exl3-3.05bpw --host 0.0.0.0 --port 8080
```

A pack maps its n-gram table from its own file and runs on one GPU: `--tp 2` and `--ple-on-ssd` are for the MLX
checkpoint, and an EXL3 pack refuses both.

- Dense projections (attention, DeltaNet, the MTP head's fc layers, the head) run on the row-invariant EXL3
  linear (`cuda/exl3/linear.py`). Routed experts and the shared expert (as expert 512 of the same table) run on
  the grouped EXL3 expert kernel (`cuda/exl3/experts.py`), each expert matrix at its own width. The fp32 router,
  its top-k with id tie-break, and the write-back in slot order are the MLX path's.
- The tensors a pack leaves unquantized (hyper-connection down / inject / up, DeltaNet `in_proj_a/b`, the n-gram
  key and value) run on an fp16 Triton matmul whose tiles and K split depend only on the shape. No cuBLAS.
- The packs store every centred RMSNorm weight as gamma - 1: the hyper-connection norms, q/k norms, the
  indexer's, the n-gram branch's three and the MTP head's two input norms. The loader detects this and adds 1
  in fp32. Checked tensor by tensor against the MLX checkpoint: after the offset the largest difference is
  0.031 (hyper-connection norms) and 0.004 (n-gram norms). DeltaNet's gated norm, `A_log`, `dt_bias`, the conv weights and the router equal the MLX values.
- The 128 `ple.ngram_embedding.shard_*` tensors are named `.trellis` but are not EXL3 tiles. They use
  ExLlamaV3's n-gram row codec: a row is one fp16 scale followed by 160 values of K bits in a tail-biting mul1
  trellis, and each head has an fp16 bias. The engine decodes a window's rows as ExLlamaV3's `ngram_dequant`
  does, bit for bit (tested at 2 to 8 bits). The rows stay in the checkpoint, memory-mapped.
- turboderp's pack keeps the MTP head's final mixer in `mtp_hyper_connection_mixer_patch.safetensors`,
  outside the index, and the reader loads it from there. A pack that carries the MTP layer drafts with it.
  For the draft head, the head's rows for the draft vocabulary are decoded through the EXL3 linear and
  requantized to 4-bit groups of 32. This affects only which drafts are proposed; verification uses the full
  EXL3 head.
- Prompt chunks decode each dense projection's weights once a chunk and multiply with a fixed-tile GEMM, so a
  prompt's rows do not depend on its chunking; routed experts run the grouped kernel in 1,024-row windows.

Measured on one DGX Spark (GB10) through `tensorfold serve`: the 3.05 bpw pack against the MLX 4-bit checkpoint
on the same engine and box, the [public benchmark command](README.md#measurements), medians of 10 runs a cell
(two sessions of each, alternated):

| Cell | EXL3 3.05 bpw | MLX 4-bit | vLLM MTP=3 |
| --- | ---: | ---: | ---: |
| Code, sampled | 80.8 tok/s | 76.5 tok/s | 42.4 tok/s |
| Chat, sampled | 59.4 tok/s | 62.7 tok/s | 33.2 tok/s |
| Code, greedy | 77.7 tok/s | 74.2 tok/s | 40.9 tok/s |
| Chat, greedy | 69.2 tok/s | 72.4 tok/s | 37.6 tok/s |

Drafted replies equal `"draft": false` ones (9 of 9), resumed prompts equal fresh ones, and four concurrent
streams equal their solo runs. Cold prefill runs 930-970 tok/s from 2k to 64k, about 0.4x the MLX checkpoint's
2,300-2,500: the routed experts decode their tiles again for every 16 rows. The pack's weights take 52 GB of GPU
memory against the MLX checkpoint's 81 GB, so on a Spark the default window is the full 262,144 tokens (57.6 GiB
allocated after loading) where the MLX checkpoint's is about 74,000; the 32.6 GB n-gram table stays in the page
cache.

#### Against ExLlamaV3 on the same pack

These comparisons were made when the path was written, on the 0.3.4.1 engine, with two packs: turboderp's
3.05 bpw and SAGE, a mixed-K 4.15 bpw pack the contributor built (2 to 8 bits a matrix; not published).

The first comparison is teacher-forced over 2,048 positions: the first 2,049 tokens of ExLlamaV3's
`eval/eval_texts`, all fed in one pass. TensorFold decodes in 64-row windows. ExLlamaV3 is the fork at 249f22a
with its mixed-K expert path. The token ids are the same for all three checkpoints, which share one
tokenizer.

| | top-1 agreement | TensorFold NLL | ExLlamaV3 NLL | TensorFold top-1 acc | ExLlamaV3 top-1 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGE 4.15 bpw (mixed-K, 2-8 bits) | 0.954 | 1.0640 | 1.0642 | 0.718 | 0.720 |
| turboderp 3.05 bpw | 0.959 | 0.8510 | 0.8506 | 0.774 | 0.779 |
| TensorFold, MLX 4-bit (same tokens) | | 1.0574 | | 0.726 | |

The two engines' mean NLLs agree within 0.0005. The positions where they pick different tokens are close
calls. Where ExLlamaV3's top two logits are at least 0.5 apart, the engines agree on 98.8% (SAGE) and 99.1%
(3.05) of positions; at a gap of at least 1.0 they agree on 99.7%. ExLlamaV3 is no closer to itself when the
batch shape changes: rerunning the first 1,025 tokens alone, instead of as part of all 2,049, it agrees with its
own full pass on 96.4% (SAGE) and 97.2% (3.05) of positions. At the same positions TensorFold agrees with
ExLlamaV3's full pass on 95.5% and 97.0%. The per-position NLL difference has a median of 0.018 and 0.011.

On TensorFold's side, `tests/cuda/test_qwen4_exp_exl3.py` checks the following against the pack that
`TENSORFOLD_EXL3_FLASHNEXT` names, cut to 4 layers plus the head and the MTP head:

- a window of 1 to 128 rows gives one-row steps' logits bit for bit;
- MTP-drafted decoding gives serial decoding's tokens, eager and in CUDA graphs, greedy and sampled;
- a real layer's experts match an fp64 reference and are row-invariant.

Through `tensorfold serve` on the full packs, each reply was requested with `"draft": false` and again with
drafts. The prompts were the benchmark's two, the raw prompt as chat, and two 256-token chats, each greedy and
with seeds 1234 to 1238. On both packs, 30 of 30 replies had identical token-id SHA-256s. A drafted round
emitted 2.93 tokens on average on SAGE and 3.16 on 3.05.

```bash
TENSORFOLD_EXL3_FLASHNEXT=flashnext-exl3-3.05bpw pytest -q tests/cuda/test_qwen4_exp_exl3.py
```

#### Serial speed against ExLlamaV3

These are decode tokens per second after the first token, on one Spark. Each cell is one stream, 64-token replies,
and the median of seeds 1234 to 1238 after one warm-up. Every engine went through the same client:
`python tools/bench_openai.py http://127.0.0.1:PORT MODEL --tokens 64 --reps 5`. All TensorFold rows ran in
one exclusive GPU session, with every pack's page cache dropped before each server started. ExLlamaV3 ran behind a
minimal OpenAI shim around its `Generator`, serially: batch size 1, an 8,192-token cache, n-gram tables in host
RAM, no drafts. The ExLlamaV3 SAGE row comes from a second session. In that session it first failed to load
("Insufficient VRAM in split") while the previous server's pack was still in the page cache. The second session
also re-measured TensorFold on SAGE serially at 34.6 / 34.4 / 34.8 / 34.6, within 3.5% of the table.

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| ExLlamaV3, SAGE 4.15 bpw, serial | 22.3 | 22.2 | 22.3 | 22.2 |
| TensorFold, SAGE 4.15 bpw, serial | 34.3 | 34.0 | 34.5 | 33.5 |
| ExLlamaV3, turboderp 3.05 bpw, serial | 37.0 | 36.9 | 37.1 | 36.9 |
| TensorFold, turboderp 3.05 bpw, serial | 34.9 | 34.9 | 35.1 | 35.0 |
| TensorFold, MLX 4-bit, serial | 36.8 | 36.7 | 37.0 | 36.8 |

Serially on the mixed-K pack, TensorFold is 1.51-1.55x ExLlamaV3. On the uniform 3.05 bpw pack
TensorFold's serial path is 5-6% behind ExLlamaV3's, and on both EXL3 packs it is 5-9% behind its own MLX path.
With drafts on the current engine (the table above), the 3.05 bpw pack decodes 1.6-2.2x ExLlamaV3's serial speed.

### MLX 4-bit, one or two ranks

For two ranks, pull the checkpoint on both and start rank 1 first:

```bash
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

### Serving

The default CUDA cap is six MTP drafts, with chains stopping below the configured confidence threshold.
`--mtp-drafts N` changes the cap; `--no-drafts` or `"draft": false` selects serial decoding.
Single-request serving uses CUDA graphs for verify windows and draft steps. Two-rank reductions add
gathered partials in rank order.

With one GPU, `--parallel N` enables eager shared forwards for up to N requests; CUDA
`--parallel auto` selects one request. Two ranks serve one request at a time and reject `--parallel N`
when N exceeds one. For prefix reuse, the single-request engine and the concurrent decoder keep prompt
states; a follow-up prefills the reply again. A kept state stops one token before its prompt's end, so the
same prompt sent again resumes, and so does a next chat turn that renders the generation prompt's `<think>`
and newline as `<think>` and two newlines. Cache capacity is allocated at startup; inspect the reported
capacity rather than assuming an older fixed token limit.

With `--parallel N`, a prompt prefills inside the rounds: each round runs the live replies' windows and the next
prompt pass (up to 2,048 rows, several prompts packed) in one forward, and each layer's experts once for both. Every
reply still equals its solo run. The cost is decode speed while prompts fill: a pass of more than ~500 rows reads
most of the experts, so live replies decode at about a tenth of their usual rate during a burst's prefill (2.6-3.2
of 32-41 tok/s on one Spark) instead of stopping. `--decode-share S` sizes the passes so a round's decoding takes
that share of the pass's time: 0.25 about doubles decode during a prefill and roughly halves prompt speed. The
default, 0, keeps whole passes.

N-gram tables are file-backed host data. On unified-memory GPUs they compete with weights and cache
allocations for RAM, so a checkpoint's GPU allocation alone does not describe its memory requirement. An explicit
`--context` that leaves them no room is reported at startup; their lookups then page from disk, a cost of about 1.3x
on prompts.

`--ple-on-ssd` leaves the 29.8 GiB of n-gram tables in the checkpoint and reads each lookup's rows from SSD,
so a 128 GB Mac can hold Flash Next. It is an opt-in trade. On an M3 Ultra, replies were the same tokens,
decode was 3.5-8% slower across the four cells, prefill was unchanged, the peak footprint fell by 40 GiB
(135.1 to 95.4 GiB) and start-up halved (35.7 s to 17.8 s).

`--ssd-experts GIB` also leaves the routed experts (70.3 GiB) in the checkpoint and streams them into a GPU pool
of that many GiB; with `--ple-on-ssd` as well, a 64 GB Mac can hold Flash Next (`python -m pip install
"tensorfold[ssd]"` first: the pool's host side is a small MLX extension built on first use). The GPU hands each
MoE layer's picks to the host and waits while missing experts are read into the pool; the expert kernels are the
resident ones with only the weight address changed, so replies are the resident model's tokens.

On an M3 Ultra, each flag was held to a smaller Mac's budget and compared on the same machine:
- `--ple-on-ssd` at a 128 GB Mac's budget (89.6 GiB) peaked at 85.6 GiB. Decode was 0.91-1.03x the resident
  run and prefill 0.84-0.91x.
- Adding `--ssd-experts 24` at a 64 GB Mac's budget (44.8 GiB) peaked at 39.5 GiB. Replies were the same tokens as
  with the experts resident: 36 of 36 requests, drafted and serial. Decode ran at 36.8-42.5 tok/s against
  107.6-129.7 (0.31-0.39x), and prefill at 332-354 against 1,014-1,081 tok/s.
- A smaller budget also halves the prompt chunk, to 2,048 tokens. Prompts past that length then reply
  differently from a 256 GB Mac's run, whichever flags are set.

### KV cache

`--kv-dtype bf16` is the default. `--kv-dtype int8` and `--kv-dtype int4` store each attention layer's keys and
values as codes with one fp16 scale per 32 values, the arithmetic of ExLlamaV3's `-cq 8` and `-cq 4` (the
non-companded grid): each group of 32 is rotated by a 32-point Hadamard, its absmax is the scale, and the codes sit
on the midpoint grid. 8-bit stores `q - 128` as int8; 4-bit stores two unsigned codes a byte, low nibble first. The
query is rotated the same way and the merged attention output is rotated back, so the stored keys and values stay
rotated. Indexer keys and pooled block keys stay bf16.

A token costs 30,784 bytes in bf16, 18,304 in int8 and 11,648 in int4, counting the scales and the MTP head's own
cache: 1.68x and 2.64x smaller (the keys and values alone shrink 1.88x and 3.56x). The startup admission counts
those bytes, so an omitted `--context` admits a longer window at int8 and int4, and an explicit `--context` is
checked against the quantized cache. The dtype holds on every path: prompt chunks and decode windows, the MTP head,
`"draft": false` requests, `--parallel N` streams and their kept prompt states, and both ranks of `--tp 2`, which
refuse to start with different `--kv-dtype` values.

A quantized cache changes the output, so its replies differ from bf16's. Drafted output still equals
`"draft": false` output at the same dtype, and a resumed prompt equals a fresh one. The MLX path and the other
families refuse `--kv-dtype` before any download.

`--mtp-confidence P`, from 0 to 1, sets the probability under which a chain stops before a later draft; the CUDA
default is 0.70, for one stream and for concurrent rounds. Only Flash Next's CUDA engine has this rule, so the MLX path and the other families refuse it.

## Draft vocabulary provenance

Both backends read the public list in `src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt`.
It contains 79,591 sorted IDs; MLX pads it with the lowest unused IDs to a multiple of 64 at load.
The target sampler still reads the full vocabulary, so the list affects proposals only.

The corpus is Homebrew CPython 3.14.5's standard-library `*.py` files, excluding `site-packages` and
`__pycache__`. It contains no repository or PyPI package text. Use `tokenizers==0.22.2` and
`tools/draft_vocab.py` with SHA-256
`1baf0dd08669355cf9cf6e32998e5436ce8712c3ab1bffe38d369f3fa3a86b56`.
The tokenizer JSON SHA-256 is:

```text
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```

Place that tokenizer at `tokenizer.json` and copy the clean stdlib into an empty `cpython` directory,
preserving relative paths. The [Nemotron corpus-copy example](nemotron-3.5.md#draft-vocabulary-provenance)
shows the file-selection rules; change its destination to `cpython`. Then run from the repository root:

```bash
TOKENIZERS_PARALLELISM=false python3 -B tools/draft_vocab.py tokenizer.json draft_vocab.txt --size 79591 --keep-below 65536 --min-count 1 --added-tokens 'cpython/**/*.py'
```

The generator retains all IDs below 65,536 and tokenizer-added IDs, then adds corpus IDs by frequency
and fills remaining places with the lowest unused IDs. It skips empty, unreadable and oversized files.
Expected output SHA-256:

```text
88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e
```

Check this hash before adopting a rebuild; a different stdlib distribution can change the corpus.
The current list replaces the older list whose PyPI corpus was not reproducible.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above; omit tensor-parallel
flags for one rank. The client supplies fixed public prompts, 64-token replies and seeds 1234 through
1238. Historical rates measured with the earlier draft list do not qualify the current list.

Use a separately pinned public long-context fixture when measuring prefill and reuse. Check fresh versus
resumed prompts across sparse-attention transitions and template changes, as well as drafted versus
serial output. Decode, cold/resumed latency, concurrent throughput and peak memory are
TBD [release-0.3.5].

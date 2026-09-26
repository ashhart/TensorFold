# GLM-5.3-Flash (`glm5_next`) on two DGX Sparks

GLM-5.3-Flash runs on TensorFold's CUDA engine only, tensor parallel over two DGX Sparks: the 4-bit checkpoint
is 182 GB and one Spark has 128 GB. Measured on two Sparks (GB10, 128 GB unified memory each) linked by their
200 Gb/s ports, in NVIDIA's `pytorch:26.07-py3` container. Package: `src/tensorfold/families/glm5_next/`
(`cuda/` holds the engine). Checkpoint: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` (MLX affine 4-bit, groups of 64,
with the MTP layer). Draft model: `incoai/GLM-5.3-Flash-DFlash2`.

## Run it

Set up both Sparks as in the [runbook](../../RUNBOOK.md#dgx-spark) (the container with the network devices, NCCL
pointed at the link), then on each:

```bash
tensorfold pull Vontra/GLM-5.3-Flash-MLX-4bit-MTP
tensorfold pull incoai/GLM-5.3-Flash-DFlash2        # optional, non-commercial (below)
```

The checkpoint is MIT-licensed. The DFlash2 draft model is optional and licensed CC BY-NC-ND 4.0 (non-commercial,
no derivatives), so pull it only if those terms fit your use. Without it GLM drafts with its MTP head only, and the
default policy uses `c3:0.35` for greedy requests and `a:0.6:0.85` for sampled ones (their rows in the tables
below).

Start rank 1 on the second Spark, then rank 0 on the first; both take rank 0's address on the link:

```bash
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 1 --master 192.168.100.1
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 0 --master 192.168.100.1 --host 0.0.0.0 --port 8080
```

Each rank reads its half of every layer straight from the pulled checkpoint (`cuda/split.py` has the rules) and
holds 90.8 GB. Loading took about 155 s a rank from a folder holding only its half (below). A first start also
compiles the kernels. Through this package, with an empty kernel cache, both ranks were ready 202 to 228 s after
starting. Read from the full checkpoint, a rank reads about a third more, because the halves of the down
projections interleave. The ranks compare their settings before loading and refuse to start when they differ, for
example when the draft model was pulled on one Spark only (pull it on both, or pass `--drafter none` to both).

A Spark short on disk can write its half once (about 91 GB) and serve that folder instead of the checkpoint:

```bash
python -m tensorfold.families.glm5_next.cuda.split ~/.cache/huggingface/hub/models--Vontra--GLM-5.3-Flash-MLX-4bit-MTP/snapshots/<revision> --rank 1 glm-rank1
tensorfold serve glm-rank1 --tp 2 --rank 1 --master 192.168.100.1
```

### Draft policies

Every reply is byte-identical to serial decoding on the same two ranks whatever the policy; the policy only
changes the speed. The default, `auto`, depends on the request's temperature:

- Greedy: each round drafts with the checkpoint's MTP head (up to 3 drafts while their probabilities' product
  stays at or above 0.35, `c3:0.35`) or with DFlash2 (up to 5, `fc5:0.3`). The engine starts with two rounds
  of each, then keeps the drafter that has committed more tokens per millisecond in this request, switches when
  the other is 3% ahead, and gives the other one round every 8 to keep its count current. The milliseconds are
  a verify window of each size, an MTP draft and a DFlash2 block, timed at load and the same on both Sparks, so
  both make the same choice. On the benchmark's greedy code prompt about half the rounds end up on DFlash2, on
  the chat prompt about 40%.
- Sampled: 1 to 3 MTP drafts a round, the number set by the running acceptance (`a:0.6:0.85`). DFlash2's sampled
  chains measured slower here.

Without the draft model, greedy requests use `c3:0.35`. A request can ask for another policy after an `@` in its
model name, for example `"model": "GLM-5.3-Flash-MLX-4bit-MTP@c3:0.35"`:

| Spec | Drafts a round |
| --- | --- |
| `auto` | the default above |
| `auto:E:EVERY:MARGIN` | the same choice with E rounds of each first, one round of the other every EVERY rounds and a MARGIN to switch, for sampled requests too |
| `0` | none (serial) |
| `N` | N MTP drafts |
| `a` or `a:LOW:HIGH` | 1 to 3 MTP drafts: 1 while the running acceptance is under LOW, 2 under HIGH, else 3 |
| `cN:P` | up to N MTP drafts while the product of the drafts' own probabilities stays at or above P |
| `fN`, `fcN:P`, `fa:...` | the same with DFlash2 drafts (needs the draft model on both Sparks) |

`--mtp-drafts N` sets a fixed default, `--no-drafts` serves serially, and `"draft": false` in a request decodes
that one serially (the reference the drafted replies equal). `"ignore_eos": true` decodes `max_tokens` tokens
past an end-of-sequence token, as the benchmarks below do.

The engine keeps the committed state after the last request's prompt and after its reply. A prompt that extends
either (the next turn of a conversation) resumes from it instead of prefilling from the start, with the same
bits as a fresh prefill.

## What decides the speed

- 45 decoder layers over a hidden size of 4,096: 34 Kimi delta attention (KDA) layers (64 heads of 128, a gated
  delta rule with a per-channel decay) and 11 DeepSeek sparse attention (DSA) layers (MLA with a 1,536 query
  rank and a 512 latent, plus an indexer that keeps the best 2,048 keys in pools of 4).
- The first 3 layers have a dense MLP (12,288 wide); the other 42 route each token to 8 of 288 experts (2,048
  wide) plus a shared expert.
- Four residual streams mixed by manifold-constrained hyper-connections (a 24-way mix, 20 Sinkhorn steps).
- Vocabulary 154,880, and one extra decoder layer as the MTP head.
- Each rank reads about 5.0 GB a token: experts 2.68 GB, KDA 1.46, DSA 0.39, dense MLP 0.13, router and
  hyper-connections 0.17, its half of the head 0.18.

## Against vLLM

Decode tokens per second after the first token, one stream, 64-token replies, through the same OpenAI client
(`tools/bench_openai.py`) for both engines. Each cell is the median of five seeds, 1234 to 1238, after a warm-up
request. Code prompt: "Write a short Python function that computes the Fibonacci sequence and explain it." as a
raw completion. Chat prompt: "Explain how matrix multiplication uses a GPU in plain English, then give a small
numerical example." through the chat template with thinking off. Sampled means temperature 1, top-k 20, top-p
0.95.

The baseline is vLLM serving `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (EXL3 4 bpw experts, fp8 KV cache) tensor
parallel over the same two Sparks, through that recipe's own launch script: with MTP at 3 drafts, and with its
default DFlash2 at 7 drafts. Its drafted output is not byte-identical to its serial output. Ours is: every
in-engine run compares the drafted token ids with serial decoding on the same engine by SHA-256.

TensorFold ran from this package, installed with pip in NVIDIA's container on both Sparks and started as above from
folders holding each rank's half, with its default policy (`auto`). The numbers come from one session on 26
September (11:59 to 12:03 UTC) with the decoding code that ships, before the release validation. The validation's
own session gave the same chat-sampled, code-greedy and chat-greedy medians within 1% (43.3, 66.8, 44.8), but page
migration slowed the sampled-code runs of every policy there, and a rerun's measured pass as well (Slow runs on
GB10, below).

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| vLLM, MTP=3 | 24.5 | 24.3 | 32.2 | 24.7 |
| vLLM, DFlash2 at 7 drafts | 22.2 | 21.5 | 28.8 | 22.6 |
| TensorFold, default policy (`auto`) | 49.4 | 43.3 | 66.3 | 45.2 |
| TensorFold / vLLM MTP=3 | 2.02x | 1.78x | 2.06x | 1.83x |

Serial decoding (`--no-drafts`) ran at 31.2 to 33.8 tok/s per cell in two earlier sessions with the same kernels.

Single runs vary a lot at temperature 1 because the reply is a function of the seed. The default policy ran at
33.5 to 55.9 tok/s over the five code seeds, which is why the tables use medians.

### Per-cell policies

A request can name another policy (above). These are the policies measured for this recipe, in the same session
as the table above. The best policy in each column was picked from these four on the same prompts and seeds it is
scored on, so the last row is a best case for this benchmark, not a prediction for new prompts.

| Policy | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| `auto`, the default | 49.4 | 43.3 | 66.3 | 45.2 |
| `a:0.6:0.85`, MTP only | 49.3 | 43.2 | 52.9 | 40.9 |
| `c3:0.35` | 46.1 | 45.2 | 57.8 | 37.2 |
| `fc5:0.3`, DFlash2 drafts | 45.6 | 43.9 | 67.3 | 42.2 |
| Best in the column / vLLM MTP=3 | 2.02x | 1.86x | 2.09x | 1.83x |

Slow runs pulled down two greedy-chat medians here: `c3:0.35` lost three of its five runs to them (its others
reached 44.4 to 44.8) and `fc5:0.3` two (42.5 to 42.7). The default is the best or within 1.5% of it in three
columns. On sampled chat `c3:0.35` beat it by 4% in this session and trailed it in two earlier ones (41.1 and 42.3
against 43.2), because its seed 1237 run swung between 39.6 and 45.2 from session to session.

## Slow runs on GB10

Some runs came in well under the speed the same request reached in other runs, with no other job on either Spark.
Counting a run as slow below 85% of the best run of the same request, 19 of the 120 runs in the first release
session were slow, at 41 to 81% of that speed, and 20 of 200 in two earlier sessions with the same kernels. That
session measured `a:0.6:0.85`, then the default, three times:

| `a:0.6:0.85` | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| First pass | 31.0 | 43.2 | 52.0 | 37.5 |
| Rerun, pass 1 | 49.0 | 43.2 | 52.3 | 40.9 |
| Rerun, pass 2 | 49.2 | 43.0 | 34.8 | 40.7 |

Slow runs moved the first pass's medians for sampled code and greedy chat, and pass 2's for greedy code. Runs that
were not slow matched the earlier sessions seed for seed, within a median of 0.1 to 0.5% for each policy. In the
final release validation, two or three of the five sampled-code runs were slow in every policy's pass.

During two reruns, a sampler on each Spark recorded once a second the GPU's SM clock, its active clock event
reasons and its power, and the kernel's `pgmigrate_success` and `compact_stall` counters from `/proc/vmstat`. All
four slow stretches fell inside bursts of page migration (4 KB pages):

| Slow stretch | Runs slowed | Migrated, first Spark | Migrated, second Spark |
| --- | --- | ---: | ---: |
| First session, 1 | 2 of 5 greedy code runs | 0.19 GB | 1.25 GB |
| First session, 2 | 4 of 5 greedy code runs and 1 greedy chat run | 1.51 GB | 0.77 GB |
| Final tree, 1 | 2 of 5 sampled chat runs, at 22 and 32 tok/s | 0.30 GB | 0 |
| Final tree, 2 | the last sampled code run, 44.6 against 49.4 | 0 | 1.57 GB |

Outside those stretches, neither Spark migrated more than 40 MB in a second. During them GPU power dropped on both
Sparks, so the GPUs sat waiting: a stall on one rank holds the other at the next all-gather. The SM clock stayed
at 2.4 to 2.6 GHz with no clock event reason active, and direct-compaction stalls rose by 7 at most. So the slow
runs are not clock throttling, and page migration is the likely cause. What starts the migrations is not known;
the runs changed no system setting. Dropping the page cache after loading made it worse (Tried and rejected,
below).

A median of five hides two slow runs in a greedy cell, where the five replies are identical. In a sampled cell one
slow run can move the median, because the seeds differ in speed. Run a cell again when its greedy runs disagree,
or when a seed runs slower than it did in another pass.

## What paid

| Step | Effect |
| --- | --- |
| A row-invariant 4-bit matmul in Triton (a tensor-core dot per 64-input group, then scale and bias, groups in order, the K split fixed by the weight's shape) on words regrouped once at load | exact windows; dense matmuls near the memory's read rate |
| Grouped expert kernels: every distinct expert of a window read once, each (row, expert) pair computed with the same bits whatever the other rows; GLM's limited SwiGLU in the gate/up epilogue | a verify row costs only the experts it adds, 6 to 10 ms |
| Expert kernels at 2 groups a step and 2 stages (same bits) | a MoE layer 12% faster at 1 row, 18% at 4 rows |
| KDA as one kernel per layer and window (one block a head runs the chain of rows), and one launch that replays every layer's kept prefix at commit | exact KDA windows, no per-row launches |
| The router's logits in 8 K slices added in order (72 programs for 288 experts instead of 5) | 23.4 to 12.8 us a call |
| K splits and launch settings per matmul shape, each timed on distinct weight copies so caches cannot help | dense matmuls 11.05 to 9.73 ms a one-row forward; with the router's slices, serial 31.8 to 33.8 tok/s |
| CUDA graphs per window size, the all-gathers captured with them | serial step 29.6 ms |
| The checkpoint's MTP head read with vLLM's convention (the final-normed hidden row, not the streams' mean before the norm) | next-token agreement 0.739 against 0.696 |
| DFlash2 at 4 bits on the engine's matmul, its attention in a Triton kernel (PyTorch's SDPA with a mask fell back to fp32 math), both ranks drafting with half of it | a block of drafts in 3.7 to 4.2 ms; best on greedy code |
| The drafter chosen per greedy request (`auto`), with both drafters' caches kept current: a drafter that sat out takes the committed rows it missed when it next drafts | against `a:0.6:0.85` alone in one session: greedy code 52.9 to 66.3, greedy chat 40.9 to 45.2 tok/s; sampled the same |

## Tried and rejected

- MTP drafts over the first 32,768 token ids (a smaller draft head): 8.7% of the reply tokens have larger ids.
- MTP drafts sampled at 0.7 times the temperature: no gain.
- DFlash2 in bf16: slightly more accepted on chat, but its block took 7.5 ms.
- Other launch settings for the 16-row matmuls, other hyper-connection K splits, more K slices everywhere: within
  noise.
- NCCL channel and rail settings: none beat the defaults (an all-gather captured in a CUDA graph takes about
  28 us against 17 us eager).
- Dropping the page cache after loading: it set off about 25 GB of page migration during the benchmark.
- The same per-request choice for sampled requests: 2 to 4% slower than MTP drafts alone on the sampled cells,
  because DFlash2 rounds lost 8% on two of the five code seeds. On these prompts even a perfect choice per
  seed would not move the sampled medians: the median seed is one where MTP wins.
- Timing each drafter piece once at load: a slow moment right after loading put a DFlash2 block at 9 ms
  (about 3 ms in rounds) and the choice kept greedy code on MTP. Each piece is now timed in turns over several
  passes and keeps its fastest run.

## Exactness

Checked on the full model across both Sparks:

- Serial steps against verify windows of 2 to 8 rows, with full and partial keeps and other prefill chunkings:
  87 of 87 rows bit-identical.
- CUDA-graph steps against eager steps: 24 of 24 windows and 4 of 4 MTP steps bit-identical.
- Every drafted decode (MTP and DFlash2 policies, sampled and greedy, both prompts) equal to serial decoding by
  token-id SHA-256. Greedy hashes: code `9c65654e926ec8fb`, chat `b6d1267266302457`.
- Against a reference forward on dequantized weights, 150 teacher-forced tokens: top-1 agreement 97.3%, NLL
  0.947 against 0.963 (its matmuls ran in TF32 in NVIDIA's container).
- Through this package's server, on the final code: 9 of 9 drafted replies equal to the same requests sent with
  `"draft": false` (code, chat and JSON prompts, 96 tokens, seeds 1234 and 1235 and greedy); 12 of 12 replies of
  the default policy equal to serial decoding by token-id SHA-256 (both benchmark prompts, greedy and seeds 1234
  to 1238), the greedy ones with the hashes above; prompts resumed from a kept reply or prompt equal to fresh
  prefills.

The kernel tests (`tests/cuda/test_glm_kernels.py`) check row invariance of every kernel on synthetic weights and
compare with torch definitions computed in float64. `tests/cuda/test_glm_engine.py` runs the whole engine on a
synthetic two-layer checkpoint with a one-layer synthetic draft model: drafted replies equal serial ones for every
policy, including the drafter choice made to switch every round, and resumed prompts equal fresh prefills. The
container sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which makes fp32 matmuls TF32 and would loosen fp32 references
past the tests' tolerances.

## Limits

- Two Sparks only, one request at a time.
- Contexts up to 2,051 tokens were measured: below that DSA's indexer keeps every key, so attention is dense.
  `--context N` on both ranks takes longer contexts (DSA's sparse top-k, eager past 2,051 tokens, about 0.4 MB
  of cache a token on each rank). That path was checked only on a truncated model (layers 0 to 3 and the MTP
  layer, one GPU, a 2,300-token sequence): 99.6% top-1 agreement with the reference past 2,050 tokens, verify
  windows bit-identical to serial steps (21 of 21 rows), MTP-drafted decoding equal to serial. The full model on
  two Sparks, DFlash2 or `auto` drafts, and prefix reuse past 2,051 tokens were not checked, and its speed was
  not measured.
- A request whose prompt plus `max_tokens` needs more context than the server was started for gets an HTTP 400
  naming the `--context` to restart both ranks with. Without `max_tokens`, a reply stops where the context ends.
- A client that disconnects does not stop its reply early; both ranks finish it.
- On GB10, bursts of page migration can slow a run to half speed or less (Slow runs on GB10, above).

## Where the time goes

Timed at load through the whole two-rank step (CUDA graphs and all-gathers included, the fastest of 7 runs), a
verify window takes 29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6 and 69.2 ms at 1 to 8 rows; an MTP draft with its
sampling 1.7 ms and each further chained draft 1.5 ms; a DFlash2 block with its host chain 3.2 ms. A one-row step
takes 29.6 ms on each rank: routed and shared experts 13.4 ms (near the read rate), dense 4-bit
matmuls about 9.7 ms, 90 all-gathers about 2.4 ms, hyper-connections and KDA chains about 2.2 ms. Every extra
verify row costs 6 to 10 ms, almost all of it the experts that row adds, so acceptance decides the rest: the MTP
head's first draft is right 70 to 96% of the time and its third 10 to 36%. About half of an MTP draft step is
the 154,880-row head. The next things to try: picking drafts on the GPU without a host round trip per
step, a smaller head for drafts only, and verify trees with a second candidate at the first position.

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
2,048 tokens. Cold and resumed prompts use the same rendered-token boundaries, and reuse starts only
at these cuts. Templates without detected markers use 2,048-token chunks. Snapshots include the prefill
path, resolved matmul route and GPU identity; changing arithmetic requires a fresh cache.

Load-time shared-forward checks compare each stream with its own call. A failed check limits forwards
to one stream, while successful checks allow the lane engine to combine requests.

## CUDA

Use the [container setup](../../RUNBOOK.md#nvidia-gpus). For two ranks, pull the checkpoint on both and
start rank 1 first:

```bash
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The default CUDA cap is six MTP drafts, with chains stopping below the configured confidence threshold.
`--mtp-drafts N` changes the cap; `--no-drafts` or `"draft": false` selects serial decoding.
Single-request serving uses CUDA graphs for verify windows and draft steps. Two-rank reductions add
gathered partials in rank order.

With one GPU, `--parallel N` enables eager shared forwards for up to N requests; CUDA
`--parallel auto` selects one request. Two ranks serve one request at a time and reject `--parallel N`
when N exceeds one. The single-request engine retains prompt and reply states for prefix reuse; the
concurrent decoder retains prompt snapshots per stream. Cache capacity is allocated at startup; inspect
the reported capacity rather than assuming an older fixed token limit.

N-gram tables are file-backed host data. On unified-memory GPUs they compete with weights and cache
allocations for RAM, so a checkpoint's GPU allocation alone does not describe its memory requirement.

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

# Nemotron 3.5 Lightning

The MLX family is `src/tensorfold/families/nemotron_h/`, with Metal kernels in
`src/tensorfold/kernels/nemotron/lightning/v1/`. It combines Mamba-2, attention and MoE blocks.

## Run

```bash
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold serve TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --name bench
```

The checkpoint includes `mtp-4bit.safetensors`; `pull` and `serve` check that it is available.
`--no-drafts` or request field `"draft": false` selects serial decoding. MLX supports affine 4-bit
projections and experts in groups of 32 or 64; unsupported formats are refused before weight downloads
and checked again at load. CUDA supports the named 4-bit/group-64 checkpoint.

## MLX execution and exactness

The lane engine verifies MTP drafts and context copies. TensorFold projections use tensor-unit kernels
where available and row-exact kernels elsewhere. Fused routing, Mamba updates and expert kernels keep
serial and window arithmetic consistent. Drafting depends on the load-time row check, not an MLX downgrade.

Rollback retains the accepted Mamba convolution and recurrent states and trims attention caches.
Alternating KV buffers let pipelined decode avoid overwriting state still read by the preceding step.
The lane engine can combine requests after load-time shared-forward checks. Prompt chunks start at
detected assistant-message boundaries and the second message at least 256 tokens after the previous
chunk start, or after 2,048 tokens when no earlier boundary qualifies. Prefix reuse starts only at these
cuts, so cold and resumed prompts use the same chunks; templates without markers use 2,048-token chunks.

## CUDA

CUDA reads this model's MLX 4-bit checkpoint only; NVFP4 and EXL3 exports of it are not read yet. Its prompt matmuls
have no FP8 kernel, so prompt precision does not change and `--prefill-fp8` is refused.

Use the [CUDA container setup](../../RUNBOOK.md#nvidia-gpus). One or two ranks are supported.
Pull the checkpoint on each rank and start rank 1 first:

```bash
tensorfold serve TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The default cap is three MTP drafts. Later drafts stop when their cumulative head confidence falls below
20%; the first draft is retained. `--mtp-drafts N` sets a cap from zero to eight. Context copies can also
supply proposals, and verification uses windows of up to 16 rows.

The shared grouped expert kernel handles routed experts and two half-width shared experts. Mamba
convolution and recurrent state commit by replaying the kept rows before the next window. Attention
uses fixed 512-key chunks and merges them in order. Prefill uses separate prompt-chunk kernels.

Requests take turns. The engine retains prompt and reply states for prefix reuse; `"draft": false`
uses a separate serial engine. The default requested context is 16,384 tokens, subject to startup
memory admission; `--context` sets an explicit window. Inspect the reported capacity before sending
long requests. Both backends use the same public draft list below.

## Draft vocabulary provenance

The shipped `draft_ids.txt` contains 32,768 sorted IDs. It can be rebuilt byte for byte from CPython
3.14.5's standard-library Python files, excluding `site-packages`, plus tracked Python and Markdown
files from TensorFold commit `cfea94372391f3761d42d0d8946462adf28a68b4`.
No PyPI package source or working-tree text is part of this corpus.

Tokenizer JSON SHA-256:

```text
623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7
```

Use `tokenizers==0.22.2` and the archived commit's `tools/draft_vocab.py`, whose SHA-256 is
`980f3d0c3520260d49841b85b5298b08109b7148e99fb5d7a03f4011df7b26a4`.
Place the matching tokenizer at `tokenizer.json`. In an empty output directory within a repository clone,
export the tracked corpus and copy a clean CPython 3.14.5 stdlib:

```bash
git archive --format=tar --prefix=corpus/tensorfold/ cfea94372391f3761d42d0d8946462adf28a68b4 | tar -xf -
python3.14 -B - <<'PYTHON'
from pathlib import Path
import shutil
import sys
import sysconfig
assert sys.version_info[:3] == (3, 14, 5)
source = Path(sysconfig.get_path("stdlib")).resolve()
for path in sorted(source.rglob("*.py")):
    rel = path.relative_to(source)
    if "site-packages" in rel.parts or "__pycache__" in rel.parts:
        continue
    path.resolve().relative_to(source)
    dest = Path("corpus/cpython") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
PYTHON
TOKENIZERS_PARALLELISM=false python3 -B corpus/tensorfold/tools/draft_vocab.py tokenizer.json draft_ids.txt --size 32768 --min-count 1 'corpus/cpython/**/*.py' 'corpus/tensorfold/**/*.py' 'corpus/tensorfold/**/*.md'
```

The verified stdlib came from Homebrew CPython 3.14.5. The selected corpus contains 1,849 stdlib Python
files and 173 package Python/Markdown files. The generator reads 1,995 nonempty UTF-8 files and
10,172,943 tokens. It skips unreadable files and text over its default 2,000,000-character limit.
It keeps IDs below 1024, then IDs by frequency, then the lowest unused IDs until full.

Expected output SHA-256:

```text
436840405e3507339efe85c410bb87a927ede7eb44eae620c0dda225c392d45f
```

A different stdlib distribution can change the selected files. Check the output hash before adopting a
rebuild. This subset affects draft proposals only; the target still verifies against its full vocabulary.

## Native prompt reuse

The Zig engine keeps a stream's prompt state at each planned chunk end: every Mamba conv and SSM state, the KV rows and
the draft head's KV rows (`zig/src/families/nemotron/snapshot.zig`). A later request whose prompt extends a kept state
resumes there. Both the keep and the resume are copies on the GPU, on the engine's own queue, so neither waits for the
host. `--prompt-cache-gib` sizes the memory as for Flash Next; `0` turns it off.

Measured on an M5 Ultra (256 GB), macOS 27.0.1, Zig 0.17.0, checkpoint revision `d9d758fb`, with
`--context 32768 --temperature 0 --no-thinking`, greedy, 96 reply tokens, a 9.2k-token chat over four turns:

| | Turns 2-4, cache off | Turns 2-4, cache on |
| --- | --- | --- |
| Prompt time | 1.29-1.36 s | 0.28-0.31 s, 9,213 tokens resumed (turn 4: 9,556) |

A fresh 9.2k-token prompt costs the same with the cache on or off: 1.07 s mean over four prompts each way, alternated,
while the cache keeps two states (106 MiB each) per prompt. Every reply equalled its cache-off reply: one conversation
alone, two conversations at once with `--parallel 4` (each equal to its solo run), and `--no-drafts`; drafted output
equalled `--no-drafts` throughout.

`--learn` keeps the states at shared cuts (a system prompt and its tools) on disk, under an identity made of the
checkpoint's config and tensor index, each shard file's size, inode and change times, the prefill step, the head,
every kernel source, a probe's prompt-pass bits, the OS build and the chip
(`zig/src/families/nemotron/learned_prompts.zig`). Weights that `--slide` writes into the shards start a new identity,
and a live weight change stops the server reading or writing learned states until it restarts. The probe runs a fixed
4,143-token prompt at startup, about a second. A later server with the same identity reads a learned state back and
resumes a fresh conversation from it: with the 9.2k-token system prompt above, a new server's first turn took 0.27 s
instead of 1.30 s, resuming 9,213 tokens from disk, and every reply equalled the cache-off reply. The two learned
states took 213 MB on disk.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above.
Compare drafted/serial and resumed/fresh output on each backend and rank count, plus concurrent/solo
requests on MLX. Decode rate, cold/resumed first-token latency and peak memory are TBD [release-0.3.5].

On a 64 GB M5 Pro with TensorFold 0.3.5.1 and `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@8bbcb5b6`
(#70; setup in the [Qwen3.8-27B recipe](qwen3.8-27b.md#a-64-gb-m5-pro-on-0351)), the fitted context was the full
262,144 tokens and the lifetime peak footprint 42.21 GiB, reached during load and warm-up. A 261,780-token prompt
prefilled in 311.5 s and resumed in 0.63 s with 261,774 tokens cached. Decode medians were 146.0, 123.3, 131.3 and
135.3 tok/s with MTP (code sampled, chat sampled, code greedy, chat greedy) and 107.8, 109.0, 109.5 and 109.0 with
`--no-drafts`. All 180 concurrent replies at 1, 2, 4 and 8 streams equaled their solo runs, and every solo run
equaled `"draft": false`. These are 0.3.5.1 results, not a later release's.

## Shared rounds on the Zig engine

The native server runs every live stream's window in one forward. That forward holds four rows a lane (`--parallel`),
64 at least and 128 at most, so up to `--parallel 16` nothing changes from the 64-row buffers the engine always
allocated, and at 32 or 64 lanes each drafting stream keeps rows for its drafts instead of the round splitting. Each
row past 64 costs one Mamba state slot, 2 MiB a Mamba layer, 47 MiB across Lightning's 23, so `--parallel 64` takes
about 2.9 GiB more than before and `--parallel 8` the same. The startup timing sweep times shared rounds up to that
width, so the planner prices them from measurements rather than a straight line past 32.

Measured on an M5 Ultra (256 GB, macOS 27.0.1) with `--parallel 64 --context 32768 --prompt-cache-gib 0`, 256
tokens a reply, greedy, two alternated rounds, `tools/bench_concurrent.py --alone --serial`:

| Sessions | Code, before | Code, after | Chat, before | Chat, after |
|---|---|---|---|---|
| 8 | 939-944 tok/s | 944-953 | 717-721 | 713-717 |
| 32 | 779-795 | 958-1,019 | 740-755 | 843 |
| 64 | 779-795 | 968-1,006 | 738-753 | 847-849 |

Every concurrent reply equaled its solo run and every solo run equaled `"draft": false` (832 replies checked a
binary). One stream decodes the same as before: 505 and 347 tok/s on the two decode prompts. A 64-row cap alone,
measured on the way, gave most of the 32-session gain and a quarter of the 64-session one.

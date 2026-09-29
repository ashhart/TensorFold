# Nemotron 3.5 Lightning

The MLX family is `src/tensorfold/families/nemotron_h/`, with Metal kernels in
`src/tensorfold/kernels/nemotron/lightning/v1/`. It combines Mamba-2, attention and MoE blocks.

## Run

```bash
tensorfold pull Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --name bench
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

Use the [CUDA container setup](../../RUNBOOK.md#nvidia-gpus). One or two ranks are supported.
Pull the checkpoint on each rank and start rank 1 first:

```bash
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
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

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above.
Compare drafted/serial and resumed/fresh output on each backend and rank count, plus concurrent/solo
requests on MLX. Decode rate, cold/resumed first-token latency and peak memory are TBD [release-0.3.5].

On a 64 GB M5 Pro with TensorFold 0.3.5.1 and `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit@8bbcb5b6`
(#70; setup in the [Qwen3.8-27B recipe](qwen3.8-27b.md#a-64-gb-m5-pro-on-0351)), the fitted context was the full
262,144 tokens and the lifetime peak footprint 42.21 GiB, reached during load and warm-up. A 261,780-token prompt
prefilled in 311.5 s and resumed in 0.63 s with 261,774 tokens cached. Decode medians were 146.0, 123.3, 131.3 and
135.3 tok/s with MTP (code sampled, chat sampled, code greedy, chat greedy) and 107.8, 109.0, 109.5 and 109.0 with
`--no-drafts`. All 180 concurrent replies at 1, 2, 4 and 8 streams equaled their solo runs, and every solo run
equaled `"draft": false`. These are 0.3.5.1 results, not a later release's.

# GLM-5.3-Flash

The `glm5_next` family serves `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` on two-rank CUDA and, on a Mac with
256 GB, on the MLX lane engine ([Apple Silicon](#apple-silicon-mlx)).
The checkpoint uses affine 4-bit weights in groups of 64 and includes its MTP layer.
Kimi delta attention, sparse MLA and MoE blocks mix four residual streams.

## CUDA

Use the [two-rank container setup](../../RUNBOOK.md#nvidia-gpus) and pull the same checkpoint on both ranks:

```bash
tensorfold pull Vontra/GLM-5.3-Flash-MLX-4bit-MTP
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

Rank 1 starts first and rank 0 serves HTTP. Use the same context and drafting settings on both ranks.
The optional `incoai/GLM-5.3-Flash-DFlash2` model has CC BY-NC-ND 4.0 terms; pull it on both ranks only
when those terms fit the intended use. The CLI uses it automatically once it has been pulled.
Without it, the engine uses MTP drafts; `--drafter none` explicitly selects MTP-only drafting.
Give both ranks the same drafter setting. `--no-drafts` disables all drafting for the serial reference.
A checkpoint with neither an MTP head nor a supplied DFlash2 model is refused unless drafts are disabled.

### Draft policies

For the affine checkpoint, the default `auto` policy uses MTP for sampled requests. For greedy requests with DFlash2 available,
it compares committed tokens per estimated round time and chooses a drafter. It periodically probes
the other drafter and discards its old rate after switching away, so later probes can change the choice.
Every policy verifies against the same target, and `"draft": false` selects the serial reference.
A request can select a policy after `@` in its model ID, such as `bench@c3:0.35`, or with `tf_policy`.
`--mtp-drafts N` selects a fixed depth at startup. With DFlash2 available, `--mtp-drafts 0` selects
`fc5:0.3`; without it, zero selects the serial reference.

| Policy | Meaning |
| --- | --- |
| `auto` | Default per-request selection |
| `0` | Serial |
| `N` | Fixed number of MTP drafts |
| `a:LOW:HIGH` | MTP depth from running acceptance |
| `cN:P` | MTP chain capped at N and a probability-product threshold |
| `fN`, `fcN:P`, `fa:...` | Corresponding DFlash2 policies, requiring its checkpoint |

The default context is 2,051 tokens, where attention stays dense. A larger positive `--context` enables
sparse attention beyond that boundary if the startup memory estimate admits it on both ranks.
`--context 0` instead targets the affordable native window. An explicit reply reservation beyond the allocated window receives HTTP 400 before streaming; an omitted reply limit
is capped to the remaining space. Larger-context restart advice appears only when the estimate allows it.

Prompt prefill uses the shared CUDA prefill kernels. Decode uses CUDA graphs in the dense attention
range and eager execution beyond it. The engine keeps prompt and reply states for prefix reuse and
serves one request at a time. Both ranks finish a started reply after a client disconnects.
Full-model long-context qualification, including sparse attention and prefix reuse, is TBD [release-0.3.5];
allocation capacity is not qualification.

### EXL3

`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` is an experimental CUDA checkpoint. The reader supports 4-bit
mcg-codebook routed experts with BF16 weights elsewhere, not arbitrary EXL3 layouts. Start it with the
same two-rank command, substituting its checkpoint ID on both ranks. With DFlash2 available, the EXL3
`auto` policy uses DFlash2; without it, MTP remains available.

The expert decoder and BF16 target matmul keep row arithmetic fixed. A quantized copy of the head may
propose drafts, but target verification retains the BF16 head. EXL3 speed, capacity and long-context
qualification are TBD [release-0.3.5].

## Apple Silicon (MLX)

On a Mac with 256 GB and MLX 0.32.2 or later (`serve` refuses an older MLX):

```bash
tensorfold pull Vontra/GLM-5.3-Flash-MLX-4bit-MTP
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP
```

The model decodes through the lane engine's family rounds with the checkpoint's MTP head. A round's drafted rows
are verified in one forward, and every row of a window gets its one-row call's bits, so drafted replies equal
`"draft": false`. A load-time check sets the widest exact window (up to 16 rows) and a second one whether several
streams' rows can share a forward; concurrent requests share rounds when it passes. The family sets
`MLX_ENABLE_TF32=0`, so M5-generation GPUs keep fp32 matmuls in fp32. Prompts prefill in the engine's chunks and a
resumed prompt gets a fresh prompt's bits.

The weights take about 170 GiB, so the family states an 85% memory allowance on Macs of 256 GB or less, for a
machine with nothing else loaded; concurrent streams are still admitted within it less what other programs hold.
Prompt admission counts the latent cache's growth and a prompt chunk's indexer workspace.

The load-time checks narrow the window or stop sharing forwards where bits would differ (on the CPU, where
nothing serves, a mixed-bit checkpoint's window narrows to 7 rows). Real-weight qualification on the 0.3.5 line,
against mlx-vlm 0.7.3's server (mlx_lm has no GLM-5.3-Flash), is TBD [release-0.3.5.1].

### Experts on SSD

`--ssd-experts GIB` leaves the decoder layers' routed experts (159.5 GiB of the 169.2) in the checkpoint and
streams them into a GPU pool of that many GiB. Everything else stays resident, the MTP layer included. With it,
a 128 GB Mac can serve GLM-5.3-Flash:

```bash
python -m pip install "tensorfold[ssd]"       # cmake and nanobind, to build a small MLX extension on first use
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP --ssd-experts 64
```

- After each layer's router, the GPU signals the host through a shared Metal event and waits.
- The host reads the picks and copies any expert missing from the pool from SSD into a free slot. It then
  updates the slot table and lets the GPU go on.
- The expert kernels are the resident ones with only the weight address changed, so replies are the
  resident model's tokens.
- A prompt chunk loads each layer's picked experts in turn.
- The pool keeps the most recently used experts, and reads bypass the page cache.
- The extension builds against the installed MLX with the Xcode command line tools.

On an M3 Ultra held to a 128 GB Mac's budget (TENSORFOLD_MEMORY_LIMIT_GB=89.6) with `--ssd-experts 64`, the
streamed run was compared with the resident one on the same machine:
- Replies were the same tokens: 36 of 36 serial and cell requests, plus the 2k prompt.
- Two concurrent streams each matched their solo reply, and resumed prompts matched fresh ones.
- The footprint peaked at 87.6 GiB against 184.0 resident.
- Decode ran at 8.9-10.5 tok/s against 62.8-73.6 resident.
- A 2k prompt prefilled at 144 tok/s against 451.

### Mixed-bit checkpoints

The loader reads two layouts of the same weights: the original one (`Vontra/GLM-5.3-Flash-MLX-4bit-MTP`) and the
one mlx-lm's converter writes (`language_model.model.*`, one fused `conv1d`, `forget_gate.*`, the absorbed
`embed_q` / `unembed_out` pair in place of `kv_b_proj`, the MTP layer as `mtp.0.*` with a bf16 `eh_proj`). Such
conversions usually store per-tensor overrides: routed experts at 4 bits, attention, shared experts and the head at
8, some layers at 5 or 6. `layouts.py` maps the names, a stack of parts at different widths runs part by part, and
8-, 6- and 5-bit tensors take row kernels transcribed from MLX's one-row `qmv_fast` / `qmv_quad` loops, so verify
windows keep one-row bits. On grant-ai's abliterated conversion, on a 256 GB M3 Ultra, the contributor measured
46.3 tok/s drafted, equal to `"draft": false`.

### Prefill

Prompt chunks attend as decode does: each query reads its own selected keys from the latent cache, so prefill cost
and memory stay flat with context. The contributor measured 336 / 334 / 309 tok/s at 10k / 35k / 103k tokens on a
256 GB M3 Ultra, against 331 / 260 / 143 for the previous prefill.

## Responses and exactness

When thinking is disabled, the server closes the template's open think block so the response reaches
`content`. Both servers turn GLM's native `arg_key`/`arg_value` tool calls into OpenAI `tool_calls`,
decoding each value by the tool's schema (a string keeps its exact text); the CUDA server keeps Qwen's XML
parameter values as text.

CUDA tests cover row arithmetic, recurrent rollback and synthetic model execution. Validate real
weights separately for drafted/serial and resumed/fresh output, with both thinking modes.
CUDA graphs and eager execution must agree under the same rank configuration.
Use a separate fp32 reference for quality checks, with TF32 disabled on that reference.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above. Retain model and
runtime revisions with every run. Decode rate, cold/resumed first-token latency and peak memory are
TBD [release-0.3.5]. Record the selected drafter policy with the result.

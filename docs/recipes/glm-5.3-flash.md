# GLM-5.3-Flash

The `glm5_next` family serves `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` on two-rank CUDA and, on a Mac with
256 GB, on the MLX lane engine ([Apple Silicon](#apple-silicon-mlx)).
The checkpoint uses affine 4-bit weights in groups of 64 and includes its MTP layer.
Kimi delta attention, sparse MLA and MoE blocks mix four residual streams.

## CUDA

On CUDA GLM-5.3-Flash runs on two ranks from Mia-AiLab's EXL3 checkpoint (`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`,
experimental; [EXL3](#exl3)) or from the MLX 4-bit checkpoint, the portable option that a 256 GB Mac serves too.
No NVFP4 checkpoint of it is read. `tensorfold serve` loads the checkpoint you name; it picks none by itself. Prompt
precision does not change here: neither checkpoint has an FP8 prompt kernel, so `--prefill-fp8` is refused.

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
`TF_GLM_MTP` decides whether the CUDA engine loads the MTP head: `1` (the default) keeps it beside DFlash2, so MTP
policies and `auto`'s per-round choice stay available; `auto` leaves it out when DFlash2 is loaded or drafts are
disabled; `0` leaves it out. Left out, it saves each rank the head's weights (about 2 GiB for this checkpoint), its
cache rows and decode buffers, and prompts skip its absorb; MTP policies (and `--mtp-drafts N`) then draft with
DFlash2. Replies are the same either way. On two GB10s with DFlash2, `auto` read prompts about 3% faster and decoded
sampled code faster, but greedy chat about 4% slower, so the head stays by default. Give both ranks the same setting.

### EXL3

`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` is an experimental CUDA checkpoint. The reader supports 4-bit
mcg-codebook routed experts with BF16 weights elsewhere, not arbitrary EXL3 layouts. Start it with the
two-rank command above, substituting its checkpoint ID on both ranks. With DFlash2 available, the EXL3
`auto` policy uses DFlash2; without it, MTP remains available.

The expert decoder and BF16 target matmul keep row arithmetic fixed. A quantized copy of the head may
propose drafts, but target verification retains the BF16 head. EXL3 speed, capacity and long-context
qualification are TBD [release-0.3.5].

#### Dense projections from an EXL3 pack

The checkpoint quantizes the routed experts only: attention, the shared experts, the three dense MLPs and the head
stay BF16, about 9.7 GiB a rank that every decode round reads. `TF_GLM_DENSE_EXL3` names one or more safetensors
files (`:`-separated) holding EXL3 groups for some of those matrices under the checkpoint's own module names
(`<module>.trellis`, `suh`, `svh`, and `mul1` or `mcg`), for instance the non-expert groups of
`turboderp/GLM-5.3-Flash-exl3`, whose KDA `qkv_proj` is cut back into `q_proj` / `k_proj` / `v_proj`. Each matrix a
pack holds runs on `tensorfold.cuda.exl3`'s row-invariant `Exl3Linear` (the decode kernel up to 128 rows, the prompt
GEMM above); the others keep the BF16 path. A rank takes the tiles its BF16 twin's split rule gives it (output
columns for q/k/v, q_b, gate and up; input rows for o and down; whole for q_a and kv_a; its vocabulary columns for the
head), so the ranks' all-reduce returns the full layer's output. A pack is lossy like any EXL3 weight, and a pack that
holds `lm_head` replaces the BF16 head for verification too; leave the head out to keep it.

Measured on two DGX Sparks with a pack of attention and shared experts at 4 bpw, dense MLPs at 5 and the head at 6
(MiaAI-Lab's GLM-5.3-Flash TensorFold recipe, whose FP8 dense weights serve the matrices the pack lacks; sparkDash,
1 stream, prose / code / structured): FP8 51.4 / 116.1 / 99.7 tok/s, the pack 66.8 / 122.9 / 107.1. HumanEval
(164 problems x 5 samples) 96.3% against 96.4% with FP8; a 68k-token prompt filled in ~40 s against 42.5 s with
`TF_GLM_PREFILL_ROWS=4096`, which amortizes the per-chunk weight unpack over twice the rows.

### Draft policies

For the affine checkpoint with its MTP head loaded, the default `auto` policy uses MTP for sampled requests.
For greedy requests with DFlash2 loaded beside the head (`TF_GLM_MTP=1`, the default),
it compares committed tokens per estimated round time and chooses a drafter. It periodically probes
the other drafter and discards its old rate after switching away, so later probes can change the choice.
Without the head (`TF_GLM_MTP=auto` beside DFlash2, or `0`), `auto` drafts with DFlash2.
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

Prompt prefill uses the shared CUDA prefill kernels. Decode uses CUDA graphs, past the dense limit one per
pool bucket. The engine keeps up to 8 conversations' prompts (`TF_GLM_CACHE_ENTRIES`): when another
conversation takes the attention caches, a kept prompt's rows are saved. Kept states and saved rows together get
`TF_GLM_CACHE_GIB` (default 3), or less when the window leaves less memory on either Spark; the startup log says
when it is less, and the memory estimate includes it. It serves one request at a time. Both ranks finish a started
reply after a client disconnects.

DFlash2 attends only its 2,048-row sliding window: a block pass reads only the window's tiles, and the drafter keeps
its context in a ring of that window, its block and a tile (2,176 rows, 21 MiB a rank whatever the window, instead
of 10 KiB a rank for every token of the window). A kept prompt DFlash2 can resume from holds a copy of the window
(20 MiB a rank, within `TF_GLM_CACHE_GIB`). The drafts are the same bits. `TF_GLM_DRAFT_RING=0` keeps the
whole-window buffer instead (give both ranks the same value). The memory estimate counts the draft model as it is
held (4-bit copies and its selector's codebooks, 0.63 GiB a rank), not at 4 bytes a checkpoint value.

### Long contexts: the latent cache

The DSA layers are NoPE MLA: head h's key is `Wk_h c` and its value `Wv_h c`, with `c` the token's 512-wide
normalized latent and `Wk_h`, `Wv_h` blocks of `kv_b_proj`. So `score_h = (Wk_h^T q_h) . c` and
`out_h = Wv_h (sum_j p_j c_j)`. The CUDA engine caches `c` only (bf16, 1 KB a token and layer, shared by the
heads and both ranks; `TF_GLM_LATENT=0` returns to per-head keys and values), absorbs the query once per row,
attends over latents and expands the attended latent once. Past 2,051 tokens a row attends to its top 512 index
pools (2,048 tokens) plus its incomplete pool; the pools are ranked by one radix-select kernel over the pools the
chunk can see, rounded up to a power of two. Every kernel computes a row alone in an order fixed by its own
position, so a decode window's rows keep the serial steps' bits; prompt chunks give the same bits for any
chunking. The memory estimate sizes the latent cache (`mla_geometry(latent=True)`).

Measured on two DGX Sparks (GB10, 128 GB each) with the MLX 4-bit checkpoint, MTP drafts only,
`--context 262144`, a synthetic codebase with one hidden fact, cold prompts:

| Prompt | Prompt reading | First token | Decode at that depth | Hidden fact |
| --- | ---: | ---: | ---: | :---: |
| 32,770 tokens | 1,138 tok/s | 29 s | 50.9 tok/s | found |
| 131,074 tokens | 898 tok/s | 146 s | 47.2 tok/s | found |
| 261,906 tokens | 849 tok/s | 309 s | 31.9 tok/s | found |

An earlier run of the same kernels read 131k and 256k prompts at 1,002 and 1,006 tok/s and decoded at 40.2 tok/s
at 256k; runs vary with page migration on the Sparks. Through the 256k prompt the torch peak stayed at 93.7 GiB
against a 96.1 GiB estimate, the kept conversations' 1.6 GiB included. MemAvailable never fell below 9.5 GiB;
the admission reserve (a tenth of RAM) also carries 3.5-4.8 GiB of CUDA context, NCCL and graph memory outside
the estimate. Drafted replies equaled serial ones (9/9, 2k to 16k), resumed prompts equaled fresh ones (6/6),
and four interleaved conversations each equaled their solo replies. `--context 0` allocated a 487,495-token
window, not measured that far.

Short prompts, with the recipe's default drafting (DFlash2, `auto`), decode at 53.8 / 43.4 / 85.0 / 50.0 tok/s
(code and chat, sampled and greedy, 64 tokens, median of 5 seeds) against 0.3.6's 53.4 / 44.1 / 63.7 / 45.5. The
latent path rounds attention differently, so some replies differ from 0.3.6 (the greedy cells' texts, hence their
speeds); `TF_GLM_LATENT=0` gives 0.3.6's replies exactly.

## Apple Silicon (MLX)

On a Mac with 256 GB and MLX 0.32.2 or later (`serve` refuses an older MLX):

```bash
tensorfold pull Vontra/GLM-5.3-Flash-MLX-4bit-MTP
tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP
```

The chat template names `low` and `high`. GLM's default is Max, and `high` suits agent and coding work.
`medium` is heard as `high`. `xhigh` stays `xhigh`, which this template renders as Max.
`--reasoning-effort high` selects High, and `--reasoning-effort low` selects Low.

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

### Float32 activations

Set `"tensorfold_activation_dtype": "float32"` in a checkpoint's `config.json` to run the residual stream, the MLA
latent and indexer caches and the KDA states in float32 instead of bf16. The hyper-connection split, the gated-delta
recurrence and the templated one-row GEMVs (`matmul_rows`) run their Metal kernels in float32. Quantized `qmv_rows`
stays on its bf16 kernel, so a float32 row uses MLX's one-row call. The fused KDA step, the sparse-decode attention
and the fused bf16 matrix kernels step aside for their MLX-op paths. Rows keep their bits, so drafted replies still
equal `"draft": false`, and without the key nothing changes (bf16 logits are byte-identical to before).

On an M3 Ultra with the resident 4-bit checkpoint, float32 activations decode at about 6-9 tok/s against
48-57 for bf16, and prefill is about 1.2-1.4x slower, so the mode is for quality work, not speed.

Measured on a 512 GB M3 Ultra with GLM-5.3-Flash converted losslessly from a Q8_0 GGUF, against a float64
reference of the same weights on 579 held-out items (MMLU-Pro 200, BBH 200, HumanEval 50, MBPP 129; per-item
token-weighted RMS difference of mean NLL):

| activations | BBH | HumanEval | MBPP | MMLU-Pro answers equal to the reference | prefill 4k / 32k tok/s | decode @32k tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 0.219 | 0.016 | 0.038 | 187 / 200 | 365 / 338 | 20.4 |
| float32 | 0.011 | 0.002 | 0.003 | 199 / 200 | 371 / 312 | 19.6 |

llama.cpp's Metal path on the same Q8_0 weights (F32 KV cache) measured 0.073 / 0.016 / 0.035 and 190 / 200.
Memory: the caches double in size and admission counts them at 4 bytes a value; the weights are unchanged.
The CUDA engine refuses the key. `--ssd-experts` stays on the bf16 expert kernels and refuses this mode.

### 8-bit from a Q8_0 GGUF, without loss

A Q8_0 block (32 int8 values, one fp16 scale d) is exactly MLX's affine 8-bit format in groups of 32 with
q + 128, scale d and bias −128·d. `tools/glm5_q8_0_gguf_to_mlx.py` converts llama.cpp's `glm5next` Q8_0 GGUF this
way; the tensors llama.cpp keeps unquantised (the indexer and KDA low-rank projections) stay unquantised, and the KDA
decay rate is stored as `A`. The Mac engine reads the result: fp16 scales are widened to float32 once at load, and
every projection returns its input's dtype.

```bash
python tools/glm5_q8_0_gguf_to_mlx.py --gguf 'GLM-5.3-Flash-Q8_0-*.gguf' \
    --config zai-org-GLM-5.3-Flash/config.json --tokenizer-dir zai-org-GLM-5.3-Flash --out GLM-5.3-Flash-q8_0 --verify
tensorfold serve GLM-5.3-Flash-q8_0
```

On a 512 GB M3 Ultra, a 310 GiB Q8_0 GGUF became 328.5 GiB in 85 shards in about six minutes; `--verify` found every
one of its 1,383 tensors equal to the GGUF's values. The weights take 365 GiB resident; the GGUF has no MTP layer, so
the model decodes without drafts. The group-32 8-bit layers run MLX's one-row calls, not the fused 4-bit kernels.

### Prefill

Prompt chunks attend as decode does: each query reads its own selected keys from the latent cache, so prefill cost
and memory stay flat with context. The contributor measured 336 / 334 / 309 tok/s at 10k / 35k / 103k tokens on a
256 GB M3 Ultra, against 331 / 260 / 143 for the previous prefill.

### Scripts that load the backbone directly

`tensorfold serve` wires the weights (`mx.set_wired_limit`, `server/residency.py`) and caps MLX's cache of freed
buffers. A script that calls `families.glm5_next.weights.load_backbone` directly, such as a scorer reading logits
or a profiler, should do the same: without wiring, macOS pages parts of a 170-365 GiB checkpoint out between steps, and
on a 512 GB M3 Ultra each MoE layer then took about 0.1 s per call (4.3 s per decode step instead of 0.1 s); with the
wired limit set to the whole working set and no cache cap, the cache of per-request buffers was wired too and grew
past 440 GiB. Wire what is resident after loading, cap the cache, and clear it between requests:

```python
import mlx.core as mx
from tensorfold.families.glm5_next.weights import load_backbone
from tensorfold.server.residency import wire_resident

info = mx.device_info()
mx.set_wired_limit(int(info["max_recommended_working_set_size"]))   # as the family's load() does
model = load_backbone(model_dir)
wire_resident(mx, int(info["max_recommended_working_set_size"]))   # then only the weights stay wired
mx.set_cache_limit(8 * 2**30)                                       # the server's --mlx-cache-gib default
# ... per request: run, then mx.clear_cache()
```

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

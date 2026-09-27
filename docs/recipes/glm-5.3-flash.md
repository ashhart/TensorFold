# GLM-5.3-Flash

The `glm5_next` family serves `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` on two-rank CUDA.
There is no MLX backend for this family in this release.
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

## Responses and exactness

When thinking is disabled, the server closes the template's open think block so the response reaches
`content`. The CUDA tool parser recognizes JSON and Qwen function/parameter envelopes; it does not
convert GLM's native `arg_key`/`arg_value` syntax or apply schema-based typing to XML values.

CUDA tests cover row arithmetic, recurrent rollback and synthetic model execution. Validate real
weights separately for drafted/serial and resumed/fresh output, with both thinking modes.
CUDA graphs and eager execution must agree under the same rank configuration.
Use a separate fp32 reference for quality checks, with TF32 disabled on that reference.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above. Retain model and
runtime revisions with every run. Decode rate, cold/resumed first-token latency and peak memory are
TBD [release-0.3.5]. Record the selected drafter policy with the result.

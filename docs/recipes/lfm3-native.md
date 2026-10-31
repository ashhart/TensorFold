# LFM3 native Metal bring-up

Tracking issue: [#1491](https://github.com/ashhart/TensorFold/issues/1491).

The native Metal engine implements LFM3-1.7B text generation and LFM3-VL-2B image-to-text generation.
Both read the official MLX affine 4-, 6-, and 8-bit checkpoints without converting stored weights.
This is an experimental family port. Device and model checks cover an M4 Pro and an M5 Ultra; they do not qualify other Apple GPUs.

## Weight licence

The official text and VL MLX exports retain the **LFM Open License v1.0**, rather than TensorFold’s Apache-2.0 code licence.
See the [text weights licence](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/blob/main/LICENSE) and
[VL weights licence](https://huggingface.co/LiquidAI/LFM3-VL-2B-MLX-4bit/blob/main/LICENSE).
The checked Macs are an M4 Pro with 48 GB RAM and an M5 Ultra with 256 GiB RAM.

## Run the models

Build the native engine with the Zig version in `.zig-version` and Xcode's Metal toolchain:

```sh
zig build native -Dcpu=apple_m1
./zig-out/native/bin/tensorfold-native serve LiquidAI/LFM3-1.7B-MLX-4bit --context 8192
./zig-out/native/bin/tensorfold-native serve LiquidAI/LFM3-VL-2B-MLX-4bit --context 8192
```

Replace `4bit` with `6bit` or `8bit` for those exports. A complete local checkpoint folder also works.
The VL checkpoint loads its own image tower. Send images through OpenAI `image_url` content parts.
The default image budget is 256 tokens per request, with at most four images.
Video input is not implemented. JPEG and WebP decoding require the matching shared decoder libraries.

The shared lane engine handles target verification, sampling and concurrent streams.
Convolution history and KV state are kept per stream; partial acceptance restores the accepted row's state.
Raw prompts of at least 256 tokens can build a repeat-cache entry: the first request records interest,
the second keeps a prefix state, and a later repeat resumes it. Image cache identity includes image content and grid.

```mermaid
flowchart LR
  I[Image pixels] --> V[Vision encoder and projector]
  V --> E[Placeholder embedding rows]
  T[Text tokens] --> E
  E --> D[Shared LFM3 decoder]
  D --> L[Existing lane engine]
```

## Inspect a checkpoint

Use Python 3.11 or newer. The inspector uses the standard library. It resolves a full Hugging Face commit,
reads the config and index, and requests only the safetensors headers. It refuses a server that returns a
complete weight file instead of the requested byte range.

```sh
python3 tools/lfm3/inspect_checkpoint.py LiquidAI/LFM3-1.7B-MLX-4bit --output text4.json
python3 tools/lfm3/inspect_checkpoint.py LiquidAI/LFM3-VL-2B-MLX-4bit --output vl4.json
```

Replace `4bit` with `6bit` or `8bit` to inspect those exports. Pass `--revision COMMIT` to repeat a pinned inspection.
Use `--local-dir MODEL_DIR` instead of a repository name to inspect files already on disk.

## Check the native text layout

Use the Zig version in `.zig-version`.

```sh
zig build test-lfm3-host tf-lfm3-check
./zig-out/bin/tf-lfm3-check text4.json
./zig-out/bin/tf-lfm3-check vl4.json
python3 -m unittest discover -s tools/lfm3 -p 'test_*.py' -v
```

The native checker reads the saved metadata, validates shard header geometry, and checks the LFM3 text decoder's
required names, dtypes, shapes, and byte counts. It performs no projection, GPU allocation, or model inference.
For VL, it checks the text decoder only. A passing result does not validate vision weights or image processing.

The config reader preserves full-attention versus sliding-attention layers, distinct rotary settings, sandwich norms,
and each module's affine width and group size. Unsupported geometry and conflicting quantization declarations fail.

## Checkpoint details

All six inspected MLX exports store the decoder under `language_model.model.*`, including the text-only model.
Each decoder has 594 tensors. Convolution kernels are BF16 with shape `[2048, 7, 1]`.

Text 4/6/8-bit and VL 6/8-bit use affine groups of 64. VL 4-bit uses groups of 32 and an 8-bit embedding override.
The 138 additional VL vision/projector tensors are BF16 in all three exports. Global quantization metadata must not
cause these dense tensors to be treated as packed affine weights.

The header inspections used these revisions:

- Text 4-bit: `d0ca73dcfa70ce18733754a492cdc1fb0123262d`.
- Text 6-bit: `26be898c36b545d625d84ac0761c6c3eb664aef3`.
- Text 8-bit: `ee7667f3cfccb2a41725f91820531823fcabc6b7`.
- VL 4-bit: `c823fade98e35a3d6942ad4d38b1ea15f675cd12`.
- VL 6-bit: `6d71bfae121abedc75f6d531fc7f6ceb4f5e3501`.
- VL 8-bit: `83336d5be3874a2bdd3297bde77b2114dbd3a816`.

## Qualification and remaining gate

Detailed counts, hashes, numerical evidence and reproducible requests are in the [receipts](../receipts/lfm3/README.md).

All six checkpoints passed 303 greedy/seeded-sampling, cache, concurrency and cancellation checks on one optimized M4 binary.
The final source build also passed 303/303 on an M5. Text4 and VL4 additionally passed real grammar, thinking, tool,
streaming, history and image-conditioned tool checks with the matching optional grammar library.
The 4-bit text long exactness suite passed all 73 cases through 8,193-token seams.
The final family contract passed 50 kernels and caught all 13 real controls, with zero limit problems.

Independent projection audits covered 66 class/layout cells and 60,954,624 outputs with no violations of existing FP32/BF16 bounds.
The row-pair optimization preserves the shared arithmetic: 132 checkpoint calls and a full 8,193-token logit comparison were byte-identical.
On the M5, controlled 4-bit cold requests improved text engine prefill from 1.240 to 0.813 seconds at 2k and
73.402 to 61.633 seconds at 64k; VL decoder requests improved from 1.262 to 0.815 and 73.920 to 59.647 seconds.
These are engine prefill measurements; the receipt also retains client TTFT. All compared full output-token hashes matched.

Native inference retains the checkpoint's BF16 activation boundaries and FP32 reductions.
Text trigonometric values round to BF16 before rotation; vision axial trigonometric values remain FP32.
The CPU FP64 oracle records numerical differences rather than inventing a tie policy.
MLX's vector/matrix quantized paths use different effective reconstruction, and bounded FP32 differences can propagate through history.
See the exact primitive and whole-model evidence before interpreting a next-token mismatch.

```sh
zig build tf-lfm3-forward tf-lfm3-vision tf-lfm3-project tf-lfm3-profile tf-lfm3-exact -Dcpu=apple_m1
./zig-out/bin/tf-lfm3-exact MODEL_DIR --long
python3 tools/lfm3/serve_check.py --binary ./zig-out/native/bin/tensorfold-native --model-dir MODEL_DIR --sampled --cancel --output served.json
```

Long-context model fidelity, maximum image budgets, 6/8-bit performance and other Apple GPU generations remain separate qualification work.
Structured output requires the native grammar library; logprobs and video are not advertised.
A complete earlier global M5 contract run found one Qwen4 failure that also reproduces on the unchanged pinned base.
It is not a final-source global pass. Agree model-fidelity and performance gates on #1491 before claiming release readiness.

## Default-off tiled prompt experiments

Two development build options expose the current M5 experiments independently. Both default to `false`;
release-version builds reject either option when enabled.

```sh
zig build native -Dcpu=apple_m1 -Dlfm3-residual-ffn=true -Dlfm3-tiled-attention=true
```

`lfm3-residual-ffn` changes only the FFN gate, up and down projections. It reconstructs each affine weight
as a BF16 high part and residual part, accumulates both products in FP32, and rounds the result to BF16.
The candidate requires 4-bit weights, groups of 64, 33–128 rows, dimensions divisible by 64, and K at most
12,288. Flagged nonfinite tiles use the existing row projection. Other shapes and widths retain that path.
`lfm3-tiled-attention` uses 64-query/32-key tiles, strict QK products, and BF16 high-plus-residual
probabilities for PV with FP32 accumulation. It handles nonshared slices of 9–64 rows, causal full attention
or the 256-token sliding window. Shared rows and slices of at most eight rows retain the existing attention.
Both modules check tensor-unit capability before compiling their optional pipelines; M4 and older devices
retain the existing paths. Enabling the build options selects the candidates when the serving backend initializes.
Direct `Model.load` developer harnesses must explicitly enable the corresponding candidate to exercise it.

These are work-in-progress arithmetic changes, not release-qualified replacements. The
[combined provisional serving receipt](../receipts/lfm3/m5-composed-prompt-provisional-serving.json)
records the frozen private prototype's cold 2k/8k measurements. Its bounded Text4 comparison used 128 prompt
tokens and 16 fixed teacher-forced continuations against an independent FP32 graph and actual official MLX BF16.
The combined candidate passed separate prompt/continuation logit RMS and maximum-error comparisons, while
the proposed top-20 probability diagnostics retained breaches. Those diagnostic metrics are an explicit proposal
under the [FP32 criterion](https://github.com/ashhart/TensorFold/issues/1491#issuecomment-6516252750),
not a verbatim maintainer specification or a reason to hide the breaches.
The [held-input attention receipt](../receipts/lfm3/m5-composed-attention-held-input.json) covers 22 actual
128-row cells from baseline and candidate histories; it does not substitute for propagated full-model checks.

Integrated-source kernel contracts, nonfinite attention behavior, arbitrary tails, cache/resume/partial-acceptance
seams, cancellation/concurrency, wider quantizations, images and longer contexts remain qualification work.
Earlier family receipts above do not automatically qualify these opt-in arithmetic paths. No decode-throughput
or M4 candidate speed improvement is claimed.

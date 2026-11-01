# Reproduce the bounded FP32/BF16 comparison

These tools publish the comparison used for the default-off tiled prompt experiment.
They compare independent NumPy FP32 logits (F), actual pinned official MLX-VLM BF16
logits promoted to FP32 (B), and captured native BF16 logits (N). They do not use
`BF16(F)` as B. The checkpoint reader shares the tensor loader with the older FP64
reference; the model forward in this tool uses FP32 arithmetic.

The maintainer's [ratified criterion](https://github.com/ashhart/TensorFold/issues/1491#issuecomment-6537143418)
is no further from FP32 than the reference's own BF16, retaining top-token differences
and margins. Whole-phase logit RMS/max and fixed-F-top-20 probability mean/max are an
explicit executable proposal for that criterion. The exact metric set is not a verbatim
maintainer specification. Row/column exceedances remain visible; no tie waiver applies.

## Generate the three inputs

Run from the checkout root. `MODEL_DIR` must be the same admitted TEXT4 checkpoint
for all three runs. The frozen token fixture has 128 prompt tokens and 16 fixed
teacher-forced continuation tokens. The measured rows are absolute positions 127–143.
The MLX reference requires MLX 0.32.4 and the official source files matching the
provided pins. A different installed revision is refused rather than silently reused.

```sh
python3 tools/lfm3/full_model_reference.py --backend fp32 \
  --model-dir "$MODEL_DIR" --tokens tools/lfm3/fixtures/fp32-envelope-text4-144.npy \
  --prompt-len 128 --chunk 2048 --output fp32.npy

python3 tools/lfm3/full_model_reference.py --backend mlx \
  --model-dir "$MODEL_DIR" --tokens tools/lfm3/fixtures/fp32-envelope-text4-144.npy \
  --prompt-len 128 --chunk 2048 --expect-mlx 0.32.4 \
  --official-pins tools/lfm3/fixtures/fp32-envelope-mlx-0.32.4-pins.json \
  --output reference-bf16.npy

zig build tf-lfm3-forward -Dcpu=apple_m1
./zig-out/bin/tf-lfm3-forward "$MODEL_DIR" \
  tools/lfm3/fixtures/fp32-envelope-text4-144.npy native.bf16 prompt:128 --nax-prompt
```

The native flag explicitly enables both development candidates through the public
model APIs. It refuses a device without the optional pipelines, rather than letting
an unsupported device produce a misleading candidate receipt. Omit the flag for the
baseline. Native inference remains native; Python generates development references.
Use the project's GPU lock/admission guard and one model process at a time. The FP32
reference runs on the CPU. References record model/input/output/source hashes and
NumPy/MLX versions; BLAS implementations can affect the FP32 reduction order.

## Run the failing check

```sh
python3 tools/lfm3/score_model_phases.py \
  --fp32 fp32.npy --reference-bf16 reference-bf16.npy \
  --native native.bf16 --output phases.json
```

The sidecar reference JSON files must accompany the NPY files. The scorer validates
input, geometry, selected-row and output identities before scoring, and emits prompt
and continuation phases separately. Exit 1 means at least one proposed metric fails;
inspect the JSON to distinguish a probability diagnostic from logit RMS/max failure.
An identity/geometry exception is not a numerical comparison result. Stale output is
removed before validation; a failed identity check cannot reuse an earlier receipt.

For the retained public `8df18abc4` candidate, phase-level logit RMS/max pass while
prompt probability mean/max and continuation probability max fail. All 17 top choices
match F. This is bounded teacher-forced evidence, not arbitrary-prompt, image, width,
long-context, or production fidelity qualification. The retained raw hashes and actual
metrics are in [the public runtime receipt](../../docs/receipts/lfm3/m5-public-tiled-prompt-runtime.json).

## Isolate a route and sample more boundary rows

`tf-lfm3-forward` also accepts `--tiled-attention` or `--nax-prompt` (last argument) to enable one
route at a time (`--nax-prompt` adds the tensor-unit projections and the tiled attention); with neither flag it
runs the baseline route.

The phase scorer's prompt phase is a single row. `prompt_row_sweep.py` scores many boundary rows of one
causal sequence instead: produce FP32 and official-BF16 rows once with `full_model_reference.py --rows`, capture
native `prompt:N` on `tokens[:N]` for each N, and run
`prompt_row_sweep.py --fp32 ... --reference-bf16 ... --positions ... --route NAME=DIR/PREFIX --lengths ...`.
It reports per-row RMS/max/top-20 probability, top-1 agreement and margins with no pass/fail rule. The retained TEXT6
result (25 rows, baseline vs tiled attention vs official BF16) is in
[the route-isolation receipt](../../docs/receipts/lfm3/m5-text6-route-isolation-row-sweep.json).

For vision checkpoints, `tf-lfm3-forward MODEL TOKENS.npy out.bf16 prompt:N IMAGE_ROWS.bf16 START` injects
the projected BF16 image rows at absolute position START; pass the same rows as float32 NPY plus `--image-start`
to `full_model_reference.py`. TEXT8 and VL4/6/8 comparisons (including text after real image rows) are in
[the width/image route receipt](../../docs/receipts/lfm3/m5-width-image-route-comparison.json).

```sh
python3 -m unittest discover -s tools/lfm3 -p 'test_*envelope.py'
python3 -m unittest discover -s tools/lfm3 -p test_full_model_reference.py
```

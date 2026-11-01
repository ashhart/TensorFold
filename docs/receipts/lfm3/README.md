# LFM3 Metal qualification receipts

## Current M5 long-context checks

[`current-m5-long-context.json`](current-m5-long-context.json) records all 26 completed requests:
VL6/8 text prompts at 32k/64k, text4/6/8 at 131072/262128 prompt tokens, and VL4/6/8 at 127984 prompt tokens.
Each cell has two fresh-process legs. All requests return 16 tokens with zero cached input;
all 13 repeated output-token hashes match. All 16 servers exit zero, and all three packets release their leases.
The receipt includes artifact/checkpoint pins, exact input and output hashes, timings and process memory measurements.
These are native text-context checks, including VL checkpoints; they do not establish image-context, FP64 or comparative speed acceptance.
The earlier 20-request snapshot remains in `current-m5-long-context-partial.json`.

## Tensor-unit route gates (M5 Ultra)

- [`m5-nax-exactness.json`](m5-nax-exactness.json): prompt cuts, resume and verification windows against their one-pass references, byte for byte, at all six widths.
- [`m5-nax-serving-lifecycle.json`](m5-nax-serving-lifecycle.json): served drafted/plain, concurrent/solo, cache, image and cancellation checks at all six widths, plus cancellation and concurrency with 31k-token prompts.
- [`m5-nax-real-image-rows.json`](m5-nax-real-image-rows.json): native projected rows of five real images against an independent FP64 graph and official MLX BF16.
- [`m5-nax-32k-fidelity.json`](m5-nax-32k-fidelity.json): 32768-token prompt boundary and 16 teacher-forced rows against independent FP32 and official MLX BF16, for the tensor-unit and the earlier prompt route.
- [`m5-nax-launch-fusion-rejected.json`](m5-nax-launch-fusion-rejected.json): fused decode dispatches measured and not adopted.
- [`m5-nax-decode-share.json`](m5-nax-decode-share.json): four concurrent requests with and without `--decode-share` against mlx-vlm: TTFT, aggregate prefill and decode, mean and p95 ITL.

These receipts cover native LFM3-1.7B and LFM3-VL-2B from the official MLX exports.
They distinguish implementation checks from the still-required maintainer model/performance gate.
Checkpoint revisions and run instructions are in the [family recipe](../../recipes/lfm3-native.md).

## Builds and hardware

- Initial corrected native: `a8df9730ee637237545311a6753dd4bdcb13e860700d65a01a033ae818ee0d18`.
- Optimized benchmark native: `5c8a04d4af599879797f4f5bed8e00ab7d9d70732e698bd2a1ec2a01ec762e09`.
- Guarded row-pair build: `251079309919aa9b21ca8cc6a89c58fc3fc9a9baf800fa5ce6a6af313a5b5f65`.

The guarded row-pair build additionally closes a diagnostic encoder when counter setup fails.
That change does not alter the production projection or attention arithmetic. Its serving checks were rerun separately.

Correctness runs used an M4 Pro with 48 GB RAM, macOS 26.5.1 and Zig 0.17.0.
Controlled performance runs used an M5 Ultra with 256 GiB RAM and macOS 27.0 (26A428).
A measured footprint on the M5 is not a smaller-RAM qualification.

## Served and operator checks

The optimized M4 binary passed all six greedy/seeded-sampling suites: 303 passed, zero failed or skipped.
These cover drafted/plain, concurrent/solo, real cache reuse, edited prefixes, image order/content,
corrupt input and cancellation/recovery. The guarded row-pair binary also passed 303/303 on the M5.

Text4 and VL4 grammar runs on both hosts each passed 32 configurations, concurrent comparisons and five refusals.
On each host, full-token capture verified 64/64 configuration groups, including hidden reasoning, across drafted/plain/concurrent requests.
An additional 80 checks covered real required/named tools, typed arguments, streaming, tool-result history and image-conditioned calls.
Grammar requires the matching optional library; these tests used an isolated bundle with library SHA-256
`04133accdcecd01cd7dd4e8730855905850967ed8c6bd6d8a637258264ae7687`.

The 4-bit text long suite passed 73/73, including chunk and 8,193-token seams.
The row-pair LFM3-only contract run on the M4 reported:

```text
SUMMARY 50 kernels: 50 pass gates 1-4, 0 fail, 0 known findings; 13 controls: 13 red as planted, 0 missed; 0 limit problems; gates 19521 ms, limits 133 ms
```

The full earlier M5 runner reported 272/273 kernels passing, two known findings, all 298 controls red and zero limit problems.
Its Qwen4 HC activation failure reproduces exactly in a separate build of pinned base `b30d5b9d3`.
That is not a global pass or a final-source global-suite receipt.

## Shared-stream attention follow-up

`shared-attention.json` records the subsequent shared-dispatch source and binary identity.
Full and sliding attention reuse the serial partial and merge arithmetic, with stream-local cache addresses and row maps.
All 52 LFM3 kernels passed gates 1–4, all 14 controls were red as planted, and launch limits had zero problems.
The real text4 default exactness suite passed 68/68, including unequal two/four-stream state and logits.
All six sampled/cancellation serving suites on the M5 passed 303/303 with zero failures or skips.
This follow-up is byte-preserving shared scheduling; it does not qualify tensor-unit prompt attention or a speed gate.

## Numerical evidence

`projection-fp64-bounds.json` covers 66 projection/layout cells and 60,954,624 outputs.
The actual shared FP32 helper was captured before BF16 storage and compared with independently unpacked affine FP64 products.
No raw FP32 or BF16-store error exceeded the existing derived bounds. Worst-column errors were retained.

The row-pair optimization invokes the same shared helper and preserves its accumulation order.
All 132 checkpoint comparisons at widths 1 and 64 produced 123,814,080 bytes identical to the old selector.
A separate 8,193-token before/after check compared all 2,097,408,000 BF16 logit bytes exactly.
Thus this optimization does not worsen any output-based precision metric on those comparisons.

`fidelity-investigation.json` records the first actual primitive difference and its causal propagation.
Official MLX 0.32.4 uses different effective affine reconstruction in its vector and matrix paths.
The native engine consistently follows the FP32 affine equation rather than changing arithmetic with batch width.
The first captured FP32 error was about 6.81e-9, inside the existing derived 5.52e-5 pre-store bound;
it crossed a BF16 midpoint and propagated through recurrent history.

`model-fp64-comparison.json` retains all 18 originally disjoint native/MLX positions, including repeated histories.
With direct binary64-to-BF16 rounding, native belongs to the ideal model maximum set in nine cases, MLX in 16,
both in eight and neither in one. This ideal FP64 model is a diagnostic; it does not execute the native FP32 reductions.
No new tie tolerance or end-to-end acceptance threshold was applied. The primitive evidence does not replace that model gate.

The oracle is reproducible with `tools/lfm3/fp64_reference.py`; its tests include all finite BF16 encodings,
adjacent midpoints, double rounding, subnormals, overflow and six-bit cross-word unpacking.

## Binned projection candidate

`binned-projection.json` records the explicit candidate at `75478ed17`; ordinary model loading leaves it disabled.
It preserves each input value while partitioning exponent ranges and exact code nibbles. M5 uses the shared
`nax.h` tensor-unit fragments; M4 uses its fallback. Group affine sums use compensated FP32 pairs.
The integer argument is limited to finite normal aligned products without overflow or underflow.
It does not establish universal FP64 equivalence for extreme BF16 subnormals or arbitrary inputs.

The actual native M5 tool produced **61,907,040/61,907,040** direct FP64-to-BF16 reference values across 132 captures:
11 projection classes, all six checkpoints, and prompt/decode widths 64 and 1. All 66 first-row comparisons matched.
The six small attention-gate cells use the unchanged row fallback and are included explicitly.
No column had more FP64 error than either the retained original-native or MLX result.
Those two comparator captures are from the M4, not a same-device M5 performance comparison.
The file retains worst-column metrics, input/output hashes, artifact identities and the complete contract summaries.
Both M4 and M5 passed all six operator variants, both planted controls and launch limits.
These primitive results do not settle whole-model fidelity or served performance.

To reconstruct the fixtures, save `fixture_tokens` as a little-endian uint32 NPY array and run the default
`tf-lfm3-forward MODEL TOKENS.npy baseline.bf16 trace TRACE_DIR` for each named checkpoint.
Concatenate the trace files ending in `_<input_stage>.bf16` in filename order to recreate each `input_file`;
the first `2*k` bytes recreate its `one_row_file`. Verify the recorded input hash before comparing outputs.
Run `tf-lfm3-project MODEL MODULE INPUT.bf16 ROWS OUTPUT.bf16 --binned` at widths 64 and 1.
Save the receipt’s `cases` array as `manifest.json` beside the inputs for `tools/lfm3/projection_compare.py`.
Use the frozen candidate commit for these artifact identities.

`binned-row-twin.json` records the later opt-in row twin at `5a2bdea76`: all 132 captures retain
123,814,080 bytes exactly against the earlier FP64-exact tiled captures. Both M4 and M5 pass six
variants and two planted controls. The twin remains disabled for ordinary model loading because
its serving performance has not passed; these operator results do not qualify a whole model.

## Controlled performance

`cold-prefill-ab.json` contains all 16 matched pairs / 32 fresh-server cold requests.
It includes full output-token hashes, body hashes, client TTFT, engine prefill time and peak physical process footprint.
Both sides used the same body, exact prompt counts, 16 output tokens, context 70,000, four configured lanes and zero cached tokens.
The 2k and 8k cells have three alternating pairs each; 32k and 64k have one pair each.
Only 4-bit performance was measured.

Engine prefill seconds, before to after:

- Text: 2k 1.239530 to 0.812959; 8k 5.307524 to 3.608895; 32k 27.314691 to 20.488389; 64k 73.402432 to 61.632933.
- VL decoder, text requests: 2k 1.261845 to 0.815432; 8k 5.402433 to 3.617449; 32k 27.686643 to 20.536702; 64k 73.920498 to 59.646930.

These are engine prefill times, not client TTFT. All 16 paired full output-token hashes matched.
At 64k, physical footprint was 2,926,167,768 to 2,925,315,776 bytes for text and
3,586,115,720 to 3,584,559,264 bytes for VL. Device/pinned allocator peaks were not collected.

`decode-concurrency.json` records 256-token, three-repetition code/prose runs.
All 60 concurrent-to-solo comparisons and all eight serial checks matched.
Small decode-rate differences are descriptive; they are not a separately established speedup.

Earlier standard-server comparisons still showed MLX-VLM faster, especially for long cold prefill.
[`m5-text-slice-vs-mlx.json`](m5-text-slice-vs-mlx.json) measures this slice alone against mlx-vlm at all three widths:
cold prefill is about a tenth of its rate. The stacked LFM3-VL PR carries the prompt and attention speed work.
The faster arithmetic-changing matrix candidate failed the strict no-worse precision comparison and was excluded.
The slower uniform-exponential candidate was also excluded. No unvalidated GQA experiment is enabled.

## Paired FFN and CPU greedy follow-up

`greedy-dual.json` records the subsequent immutable `b804c6ed` native build.
The FFN gate/up projections share one dispatch while retaining each projection's shared helper and BF16 stores.
A SIMD CPU greedy reduction preserves the first maximum, signed-zero ties and nonfinite rejection;
masked and positive-temperature sampling keep the original path.
All 58 LFM3 kernel checks passed, all 16 controls were red, and launch limits had zero problems.
The real text4 exactness suite passed 68/68; six M5 sampled/cancellation serving suites passed 303/303.
Actual projection pairs retained 247,628,160 bytes, and six complete teacher-forced logit arrays retained 394,752,000 bytes.

Matched M5 old/new/new/old runs used three repetitions and 256 output tokens per request.
All 240 concurrent full-token digests matched across the corresponding prompts and legs;
240/240 concurrent-to-solo and 32/32 serial checks passed.
Text code/prose single-stream rates changed from 309.27/322.59 to 317.39/332.83 tok/s;
four-stream aggregate rates changed from 682.50/717.25 to 836.65/856.70 tok/s.
VL decoder code/prose rates changed from 292.69/290.90 to 301.68/299.90 tok/s;
four-stream rates changed from 656.00/658.15 to 813.40/810.10 tok/s.
These are medians of the two run medians, with all raw three-repetition cells retained.
This ABBA experiment measures only 4-bit performance and does not assert standard-server parity.

## Reproduce

Use the pinned Zig version and a complete local checkpoint. Developer references need NumPy; the MLX reference also needs MLX.

```sh
zig build native tf-lfm3-forward tf-lfm3-project tf-lfm3-profile tf-lfm3-exact tf-kernel-contract -Dcpu=apple_m1
zig build test test-lfm3-host test-kernel-contract test-golden fuzz-cortheon -Dcpu=apple_m1
python3 -m unittest discover -s tools/lfm3 -p 'test_*.py'
./zig-out/bin/tf-kernel-contract --only LFM3
./zig-out/bin/tf-lfm3-exact MODEL_DIR --long
python3 tools/lfm3/serve_check.py --binary ./zig-out/native/bin/tensorfold-native --model-dir MODEL_DIR --sampled --cancel --output served.json
python3 tools/lfm3/reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output mlx.bf16
python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output fp64.bf16 --weight-mode exact
./zig-out/bin/tf-lfm3-forward MODEL_DIR TOKENS.npy native.bf16
./zig-out/bin/tf-lfm3-profile MODEL_DIR TOKENS.npy profile.json
```

Repeat the served command for all six checkpoints. Run one model process per GPU, using the native lock and admission guards.
The synthetic cold-prompt request bodies and rendered token arrays are regenerated by `tools/lfm3/cold_prompts.py` (`--write DIR`); it checks all 16 files against the SHA-256 hashes in `cold-prompt-sha256.json`.
The native server did not return its rendered arrays, so matching counts are not described as a captured native-token identity proof.
To repeat cold HTTP measurements, start a fresh owned server with `--context 70000 --parallel 4 --prompt-cache-gib 0`,
POST a saved body to `/v1/chat/completions`, and retain usage, full token hash and timings from the stream.
The supplied bodies omit the optional model alias. Keep the exact bodies unchanged between compared builds.

## All-width matched baseline

`all-width-performance-baseline.json` records the shared-attention binary before the SIMD greedy and paired-FFN follow-ups.
Both native and the standard server completed 72 retained single-stream requests, 360 retained concurrency requests and 48 cold requests in total.
Every measured decode/concurrency reply has 256 tokens; each cold request has 16 output tokens and zero cached prompt tokens.
All 180 native concurrent-to-solo and 24 serial comparisons matched. Exact public prompts, client hashes, token hashes and timings are retained.
There is one cold sample per cell and three decode/concurrency repetitions. These baseline rates remain below the standard server; they are not a performance-gate pass.
The native binary SHA is `d1656249bdb581b147faacbd3b3c557c00b6c45b710eeddce517a67178b6ba2b`.

The public long-prompt wrapper is `tools/lfm3/bench_decode.py`; its prompts match the recorded JSON exactly.
Use `python3 tools/lfm3/bench_decode.py decode URL MODEL --tokens 256 --reps 3 --temperatures 0 --output decode.json`.
For concurrency use the `concurrent` mode with `--levels 1,4 --alone --serial` on native, or `--levels 1,4 --no-seed` on standard.
Verify every reply length in the evidence; a shorter standard reply does not establish an equal-length rate.

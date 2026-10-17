# Qwen CUDA first-token handoff on DGX Spark

This change delivers the committed first token before preparing the initial
speculative draft. On the tested Qwen3.8-27B/DFlash2 configuration it reduces
loaded-model time to first text. Draft computation still runs, and complete
response times and startup are not uniformly faster.

## Source and environment

- Baseline: PR836, `5e1c3feaa79e40f7659537d96b939129abe39df9`.
- Validation source: `fd306e82e020c1a58f160e247efd013441cfa826`.
  The publication uses the identical 13 changed source files; their SHA256s
  are in [evidence.json](evidence.json). CUDA kernels and build inputs are
  unchanged, and all 37 compiled CUDA fatbins match the baseline byte for byte.
- One NVIDIA DGX Spark, GB10/SM121, 128 GB unified memory; Ubuntu 24.04.5 LTS,
  aarch64. Host driver `580.178.04` (`nvidia-smi` CUDA line `13.0`, the
  driver ceiling). That driver string was recorded on this Spark on 16 October
  2026, and `/usr/bin/nvidia-smi` is dated 10 August 2026, before the 15 October
  runs. Kernel `7.0.0-1019-nvidia` was read on this Spark on 17 October 2026;
  it was not logged during the timing runs themselves.
- Both builds used `nvcr.io/nvidia/pytorch:26.07-py3`, image
  `sha256:15b8de8baabe68e46afc51e88f3ccad5042013bafe8f244ec082ce0bb06d47e3`
  (created 2026-07-25), target `aarch64-linux-gnu.2.28`, Zig 0.17.0,
  ReleaseFast. The toolkit in that image is CUDA `13.3.1.008`. Its `nvcc` is
  release 13.3, V13.3.73 (`cuda_13.3.r13.3/compiler.38244171_0`), confirmed by
  running `nvcc --version` in the image on 17 October 2026. The host `nvcc`
  (13.0, V13.0.88) was not the compiler.
- Native server SHA256, shared by the loaded panel, both startup orders and
  the smoke runs: baseline
  `5700939fd17c0ad52a84c7d4bdbeeba7975a6c96952375579fda0a5d70c306c6`,
  candidate
  `e42ab4fc5079bfdeafe61234de4d044fec62e18877568e87fb19bb7e9523d581`.
  Each binary's source archive matches the baseline or candidate archive in
  [evidence.json](evidence.json).
- Target: `TensorFold/Qwen3.8-27B-MLX-4bit`, revision
  `70ae7fac63274ff2eac54152031433374cb80f2f`.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- BF16 KV, 32,768-token context, 16 server slots, zero restored prompt-cache
  tokens. Short timing requests are greedy with 256 generated tokens.
- The serving engine is Zig/CUDA. The checkpoint's MLX name identifies its
  storage format; the MLX runtime is not used for these measurements.

## Loaded-model results

The process order is baseline/candidate/candidate/baseline. Each number below
is the geometric mean of the two per-process medians for that arm. Percentage
changes use the geometric candidate/baseline ratio. These are descriptive
measurements from one ABBA sequence, not a statistical significance claim.

Fresh short requests:

| Case | Streams | TTFT, baseline → candidate (ms) | Decode, baseline → candidate (tok/s per stream) | Full response, baseline → candidate (s) |
| --- | ---: | ---: | ---: | ---: |
| Coding | 1 | 131.9 → 119.1 | 53.94 → 53.96 | 4.860 → 4.845 |
| Coding | 4 | 326.2 → 316.4 | 25.08 → 25.77 | 10.400 → 10.181 |
| Coding | 16 | 1111.2 → 1102.6 | 6.10 → 6.40 | 42.682 → 40.583 |
| Chat | 1 | 144.5 → 136.5 | 33.70 → 32.92 | 7.712 → 7.884 |
| Chat | 4 | 338.9 → 324.0 | 18.89 → 19.24 | 13.862 → 13.624 |
| Chat | 16 | 1152.3 → 1138.6 | 5.45 → 5.41 | 47.556 → 48.209 |

Loaded short TTFT is 0.77–9.77% lower across all 12 fresh/repeat cells.
Beside that gain, fresh single-stream chat completion is 2.23% longer
(decode 2.32% lower) and fresh sixteen-stream chat completion is 1.37% longer.
This is delivery ordering after prefill. It is not faster prefill arithmetic
and it is not a startup speedup. All repeat-cell results and both original
observations are in [metrics.csv](metrics.csv), including the slower cases.

Fresh four/sixteen-stream batches contain distinct prompts. Repeated batches
send the same `s00` prompt to every stream after a primer, rather than repeating
each distinct fresh prompt. At one stream, fresh and repeated cell prompts are
identical. Neither arm restores cached tokens. Request IDs, counts and body
hashes are retained in the evidence file.

The evolving-history control uses a 4,419-token shared prefix and a 4,445-token
follow-up, with 32 output tokens. Initial/follow-up TTFT changes are -0.49% and
-0.65%; completion changes are +0.18% and +0.10%. This does not exercise a
prefix-cache acceleration path.

## Startup and remaining trade-offs

Each cell was measured in both baseline-first and candidate-first order after
restarting the server and loading the model. Readiness was roughly 27–30.5 s.
Filesystem cache state was warm or uncontrolled: these are process-cold tests,
not controlled storage-cold starts. No startup speedup is established.

| Case | Streams | Launch-to-ready change | Launch-to-first-text change | Launch-to-last-DONE change |
| --- | ---: | ---: | ---: | ---: |
| Coding | 1 | +0.63% | +0.63% | +0.06% |
| Coding | 4 | +3.36% | +3.32% | +2.72% |
| Coding | 16 | +0.15% | -0.13% | +0.97% |
| Chat | 1 | -0.81% | -0.80% | -1.27% |
| Chat | 4 | +0.94% | +0.87% | +0.73% |
| Chat | 16 | -3.24% | -3.02% | -0.99% |

Lower latency changes are better. The first sixteen-stream chat request batch
after startup has 4.05% higher median request TTFT across the two orders.
Warm repeats within the startup experiment also vary: single-stream coding
has 5.19% lower measured decode and 5.21% longer completion; single-stream chat
has 4.10% lower decode and 3.96% longer completion. These cells use the startup
protocol and are separate from the loaded-model panel above. The CSV records
all first-request, warm-repeat and launch-inclusive measurements.

TTFT ends at the first nonempty text SSE; full response ends at DONE. Decode
is `(tokens - 1) / (last text arrival - first text arrival)`. Delivering the
first token earlier can lengthen that decode interval without making completion
later. Actual completion regressions also occur and are retained here.

Speculative work differs in 108/196 panel pairs, 48/90 forward startup pairs
and 49/90 reverse startup pairs. Unchanged-arm repeats also differ. These
receipts do not isolate a kernel or scheduler cause for the timing changes.
The reported prefill timing boundary moves with the handoff; it does not show
faster prefill arithmetic.

## Correctness and checks

- 376 paired timing requests match baseline/candidate request bodies, tokens
  and output hashes: 196 panel, 90 forward startup, 90 reverse startup.
  This includes served 1/4/16-stream comparisons and 40 anchor, primer,
  changed-prompt and longer-history control pairs beyond the 336 short pairs.
- Six additional smoke pairs pass. Sampled code/chat controls use temperature
  0.7, top-k 20, top-p 0.95 and seed 1234; draft-on/off output matches within
  and across arms. Sampled performance was not measured.
- Disconnect after the first text is followed by drained connections/waiting
  requests and a successful recovery request in both arms.
- Host fixtures cover commit/logprob uniqueness, cancellation at handoff,
  max-one/EOS cleanup, terminal trace/allocation failure, no requeue after
  an emitted token, and another job's draft failure while a terminal job lives.
- The qualified Spark candidate passed 483/483 host tests and built successfully.
  Local publication formatting and diff checks pass. The lean checker reports
  527 inherited findings on both baseline and candidate, with zero new findings;
  its nonzero exit is not described as a clean baseline.
- All four accepted model runs completed and restored the pre-existing service
  configuration. Minimum available memory was 61.31 GiB; the 8 GiB reserve held.
- No model/device qualification was run for Metal, Windows, other CUDA devices
  or other Qwen checkpoints. The long 2k/8k/32k/64k prefill ladder was not run;
  the served change concerns first-token delivery after prefill. No resumed
  cache path or image-input performance claim is made.

## Related work

This is a focused follow-up to BobClawblaw's PR836 Qwen CUDA engine and the
first-token work discussed in PR747, following the maintainer's invitation
to port focused changes. PR608 and PR726 provided the earlier engine and
integration work; their older performance figures are not used as this baseline.

PR853/PR888 change admission and prompt reuse, and PR861 changes planner paths;
integration must preserve their semantics and explicitly carry the callback
through any new admission path. PR928's adoption-mode fix and PR929's TCP_NODELAY
change are complementary. None of these changes is claimed by this receipt.

Authored by Jonathan Regan-Beasley in a private capacity. This contribution
does not represent the views or positions of any organisation.

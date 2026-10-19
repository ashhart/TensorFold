# DeepSeek-V4.1-1017: shared family, sparse attention and TCP MoE slots

Runtime `cdfa5203d19805b1e544b7b39647b5bc9170e913` improves the measured TCP
pair from 3.0992 to 3.6003 decode forwards/s by exchanging complete routed
slots once per layer. One Mac remains faster at 4.8619 forwards/s. The two-Mac
path still duplicates the complete backbone and attention; it provides no RAM
reduction. These are development receipts, not whole-model release qualification.

## Pins and method

The upstream base is `d1018b318fb51981c4a18f12e40269a4d3ee9713`.
[build.json](build.json) records Zig 0.17.0, `-Dcpu=apple_m3`, the source tree
and binary SHA256s. Both Macs have an M3 Ultra, 80 GPU cores and 256 GB of
unified memory. They use the [same checkpoint and original 1017 Engram tables](../ds1017-tp2-2026-10-19/model.json)
as the earlier receipt. All 69 files, 413,892,636,159 bytes per Mac, were hashed
before stopping production. [checkpoint-verification.json](checkpoint-verification.json)
records the receipts and unchanged file identities before and after execution.

All three configurations enable `TF_DSV41_PARALLEL_ATTENTION=1`, split weight
residency, periodic residency refresh and SSD Engram. `TF_DSV41_SERIAL` is
unset. The predecessor attention comparison is a separate
[three-run experiment at 61ab890f](../ds1017-attention-2026-10-19/README.md).
The current single-Mac result repeats its approximately 4.86 forwards/s with
the explicit 256-thread reservation.

Each generation uses 32 frozen prompt IDs and produces 128 completion IDs.
Prefill produces the first completion token, leaving 127 decode forwards.
Decode timing excludes checkpoint loading and startup pairing. The second Mac
supplies the three-run solo reference; the first Mac also has one solo sample.
All raw records remain in [execution.json](execution.json) and the calculated
summaries in [measurements.json](measurements.json).

| Configuration | Decode forwards/s | Samples |
| --- | ---: | ---: |
| Single Mac, second host | 4.8619 | Median of 3 |
| TCP output columns, rank 0 / rank 1 | 3.0992 / 3.0992 | 1 per rank |
| TCP routed slots, rank 0 / rank 1 | 3.6003 / 3.6003 | Median of 3 per rank |

The slot partition assigns slots 0/2/4 to rank 0 and 1/3/5 to rank 1. Each
rank calculates complete projections for its three slots and duplicates the
shared expert. The ranks gather completed routed rows and combine them in the
original six-slot order. Generation exchange counts are 15,489 for columns and
5,249 for slots: 128 forwards at 121 or 41 exchanges, plus startup identity.
This preserves each full input reduction and avoids a floating-point all-reduce.

## GPU time and remaining overhead

On rank 0, the column sample takes 40.9782 seconds, including 28.8097 seconds
of GPU timestamps. The slot sample at the median wall time takes 35.2744
seconds, including 29.6257 seconds of GPU timestamps. The difference between
wall and GPU time falls from 12.1684 to 5.6487 seconds. This remainder includes
CPU encoding, waiting and transfer; it is not a measurement of network time
alone. GPU time does not improve in this comparison.

The 268-token teacher prefill also remains slower on two Macs: 9.7314 seconds
for columns and 9.5075 for slots on the second host, against 7.5705 solo.
Loading is separate and varies substantially. The loader creates fresh Metal
buffers with uncached reads at each invocation. Five bound shards on the first
Mac, totalling 26.6 GB, live on an external volume. TP `load_seconds` also
includes peer startup and must not be interpreted as a disk benchmark.

## Numerical, state and execution checks

All six slot generation token files match the
[frozen original IDs](../ds1017-tp2-2026-10-19/prompts/gen-short-tokens.json).
Both ranks' teacher logits, with columns, slots and one-token slot chunks,
match SHA256 `910189ca833f31f7f905667f2df2f5b767391deac975dd34e7782ab694ad8eb2`.
Two interleaved lane slots at capacity 8192 and chunk 16 finish at positions
134/134 with zero bit disagreements against solo execution. Every recorded
model phase passes its Engram-aware execution I/O gate, with zero page-ins.
This permits counted Engram reads and does not mean zero disk I/O.

Each Mac passes 100 DeepSeek GPU tests, 29 EXL3 GPU tests and 28 reference
tests. The attention contract passes nonfinite, repeat/guard, shape and exact
sum gates, plus both actual M3 launch limits. Five planted numerical controls
are caught, with zero missed controls. The global limits check remains red:
twelve M1/M2 Flash-Next declarations lack compiled limits. The unchanged
upstream binary reproduces exactly those findings on both Macs. They remain
open candidate findings. [checks.json](checks.json) retains all summaries,
local Linux workflow results, template checks and the reproduced desktop
failures. Candidate and unchanged base both report 819/821 desktop tests;
typecheck, mobile and whitespace pass. No Windows execution is claimed locally.

[transport.json](transport.json) records established data and acknowledgement
sockets on `bridge0` and `en3`, between the two TB5 link addresses. An earlier
`SYN_SENT` snapshot showed a different source address; it did not establish a
permanent connection failure. No network configuration changed. This experiment
uses ordinary TCP and does not qualify MCDMA or RDMA.

## Reproduce and preserve the service

Build the runner and checks at the pinned runtime with Zig 0.17.0 and
`-Dcpu=apple_m3`. Reuse the earlier receipt's prompts, checkpoint manifest and
original Engram SHA256s. Set `DSV41_ENGRAM_STREAM=1`,
`TF_DSV41_SPLIT_RESIDENCY=1`, `TF_DSV41_RESIDENCY_REFRESH=1` and
`TF_DSV41_PARALLEL_ATTENTION=1`. For the pair, set `TF_DSV41_TP_SETTINGS` to
each rank's TCP settings. Use `TF_DSV41_TP_SLOTS=0` for columns and `1` for slots.
Start both model commands together after both file checks complete. Retain
`tokens.json`, teacher `logits.f32` and every `residency.json`, including failures.

The protected window restores both original Qwen services and the original
reranker, then verifies real replies. [restoration.json](restoration.json)
records both Qwen replies as `4`, the reranker's correct synthetic ordering,
healthy services, cleared test ports, no remaining test processes and removed
window locks. No production binary was promoted.

The new native binary builds, but its HTTP serving path was not requalified
at this runtime. The [native HTTP, cancellation and peer-loss receipt](../ds1017-tp2-2026-10-19/README.md)
belongs to `b27e39af`. Whole-model FP64/noise-floor, original-checkpoint anchors,
MTP, long-context and Rust-normalizer qualification remain open under
[#847](https://github.com/ashhart/TensorFold/issues/847). Native serving keeps
the explicit `TF_EXPERIMENTAL=deepseek` gate.

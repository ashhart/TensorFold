# Ling tiny bf16 forward diagnostic

This diagnostic runs the 24-layer Ling-3.0-tiny backbone on Metal.
It reuses Kimi's row projections, RMS, residual add, KDA, MLA and expert combine;
Ling's grouped route and three MLA glue kernels; and LFM3's bf16-point SwiGLU.
No GPU kernel source or registration changes are introduced. `kernel_contract.zig`
only registers the separate diagnostic build module.

The supported checkpoint is `inclusionAI/Ling-3.0-tiny`, revision
`9a98e35fe1c9ee255f78dd64771c7ae15a799481` (32 bf16/F32 shards).
The forward refuses incompatible config geometry, quantization and expert clamps.
Checkpoint tensors stay mapped; interleaved projection weights are cached separately.
This is a correctness diagnostic, not serving or a qualified performance path.

## Run

Use the repository's pinned Zig and build for the compatible Apple CPU target:

```sh
zig build tf-ling-forward tf-ling-checkpoint tf-ling-projection -Dcpu=apple_m1
export TF_LING_CHECKPOINT=/path/to/Ling-3.0-tiny
export TF_LING_PYTHON=/path/to/python-with-torch-and-safetensors
zig build test-ling-forward -Dcpu=apple_m1
```

`TF_LING_PYTHON` takes precedence; otherwise the oracle uses `python3` on PATH.
Without the checkpoint or Python dependencies, the optional check prints an explicit
`SKIP`. It never downloads weights or changes another model service.
For persistent artifacts, run the wrapper directly:

```sh
python3 tools/zig/ling_forward_check.py --binary zig-out/bin/ling-forward --out /tmp/ling-receipts
```

The fixed IDs are `(37*t + 11) % 157182`, at 2, 8 and 32 rows.
The native diagnostic compares whole execution with single-row chunks and a fresh
instance restored after token 1. It compares logits, KDA recurrent/conv state and
MLA cache bytes. A truncated snapshot must be rejected.
Both the chunked and resumed instances' final recurrent states and caches are compared
with the whole run. A temporary one-byte corruption of the resumed final MLA slot
was planted after the logits check: the new state comparison failed, and the fault
was removed before the clean 2/8/32-row rerun.
Forced late-Forward and early-MLA buffer errors also test owned-buffer rollback.
Pipeline/library error cleanup is guarded by `errdefer`; driver object counts are
not independently instrumented by these allocator-failure checks.
The snapshot is an internal same-checkpoint diagnostic format, not a serving or
portable checkpoint handover API. These receipts exercise in-process restoration.
The oracle remains `families/ling/oracle/ling_forward.py`, using native expert picks
for both its float64 and bf16-point runs. Its optional `--layer-trace` writes residual
hidden rows without changing logits. No second reference formula is introduced.

## Measured numerical distances

On an Apple M5 Metal device, all native logits were finite, all next-token argmaxes
agreed with both reference precisions, and all three whole/chunk/resume scenarios
completed with exact bytes. Numerical acceptance is **unresolved**, not passed.
The reference's rounding nodes include linears, SiLU, products, norms and residuals.

| Rows | Metal RMS | bf16-point RMS | Metal maximum | bf16-point maximum | Rows with larger Metal RMS |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.0317046309744 | 0.0298559560322 | 0.213487908284 | 0.185932075501 | 2/2 |
| 8 | 0.0292602858773 | 0.0310663639610 | 0.213487908284 | 0.217792950829 | 5/8 |
| 32 | 0.0200687156935 | 0.0205596869419 | 0.213487908284 | 0.217792950829 | 19/32 |

These are distances from float64 on the same native route picks. The forced-route
selection gap is 0.00135229020 at 2/8 rows and 0.01254995249 at 32 rows.
The native route need not choose the reference's preferred set; forcing records that
difference instead of mixing reference routing with native routing.
Aggregate improvement at 8/32 rows does not establish a per-row or per-column bound.
The JSON receipts retain every row's RMS, maximum and argmax for both precisions.
Whether the maintainer's acceptance comparison is aggregate or row-wise remains a
review question; the implementation does not increase a tolerance to accept itself.

Runs used one low-priority Ling GPU job at a time beside an existing model service;
no timing or isolated performance result is claimed. The wrapper refuses a macOS
run with system-wide free memory at or below 40%.

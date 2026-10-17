# Dense 27B Living Weights on the owned CUDA engine

The learning work is by @ZackSample in #911.
The serving engine is by @BobClawblaw in #836, with shared image work from #885.
The integration uses the same Living Weights commit gate as the maintained Flash Next port.

The dense model learns through gate, up and down rows beside each MLP.
Its trainer calls the owned serving forward with optional intermediate keeps and MLP nudges.
Training and conversation taps currently run one row at a time.
Prefix reuse is disabled for training.
There is no external captured-kernel execution path in this adapter.
CUDA graph capture of the repository's own serving launches remains available when no learning hooks or pockets act.

Pocket replacement stages all three vectors before publication.
Read errors propagate to the common gate.
The gate judges the stable quantized form, retains earlier facts, and restores the complete pocket after rejection.
The loader separates saved pocket rows and down columns from the untouched base affine codes.
Images use the shared Qwen processor, decoder, tower and canonical identity.
Text uses #836's existing shader ABI; image rotary positions use separate additions.

For a fitted scalar z, the sharp-16 construction is z² sigmoid(16z).
It is not exactly relu(z)².
The common positive-row tests at z = 1, 1.5 and 3 use relative tolerance 1e-6.
A fitted negative margin of -0.5 bounds sigmoid(16z) below about 0.000336.
Quantization and the actual forward still need separate qualification.
Power-of-two scaling does not preserve arbitrary subnormal or overflowing f32 values.

Host tests and compile-only checks are evidence about source integration, not CUDA numerical qualification.
Build on a CUDA machine with the repository's nvcc workflow, then run:

```
tf-cuda-test q27-back
tf-cuda-test q27-gemm
tf-cuda-test gdn-back-27b
tf-cuda-test kernel-contract
tf-cuda-test kernel-contract-limits FATBIN_DIRECTORY
tf-qwen35-learn FRESH_MODEL IDS_JSON_ARRAY
```

The model check requires a fresh untaught folder and at least 24 token IDs.
It checks serving/trainer loss, packed/solo loss, and directional differences at early and answer rows.
All configured finite-difference step sizes must meet the median and worst error bounds.
The resource extension deliberately fails until real per-SM compiler receipts replace the pending entries.

CUDA numerical runs, learned and fresh full-token parity, sequential held-out retention, unrelated drift,
rejected-round conservation, save/reopen/delete, image/cache identities, drafted/plain, seeded,
resumed/fresh and concurrent/solo qualification are unchecked on this Mac.
The review records that the contributor's older 27B recipe produced garbage on wide unrelated prompts after sixteen facts.
That historical failing control is not replaced by host tests of this port.

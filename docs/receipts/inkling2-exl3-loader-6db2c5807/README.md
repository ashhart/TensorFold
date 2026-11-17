# Inkling EXL3 loader checks

Measured source: `6db2c580780f850452c5e7adf6658ef7ff03aaf7`, on main
`6d0a128b0cde35aa091bda933577cfced3519b8a` plus the original shared host package
`7b00db2aa1d3a223edd82ea9aa60698bcaa6aab8`. Following changes move the Unreleased entry and add these receipts only.

Checkpoint: `kortexa-ai/Inkling-2-Small-EXL3-2.59bpw@05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`.
Its weights, scales and tokenizer files were unchanged. The loader uses the original auxiliary config and the
shared family parser, plan and dense tensor names. No operator or serving path is added by this contribution.

## Commands

Zig 0.17.0; cross-compiled from Linux x86_64 to Linux ARM64, baseline CPU.
The two GB10 runs used Linux `7.0.0-1019-nvidia` and NVIDIA driver `580.178.04`.

```sh
zig build -Dtarget=aarch64-linux-gnu -Dcpu=baseline tf-inkling2-exl3-load --prefix out-arm64
```

On two idle 128 GB DGX Sparks, each with one GB10, run its own rank against its corresponding checkpoint directory:

```sh
out-arm64/bin/tf-inkling2-exl3-load /checkpoint/rank0 0 2 1
out-arm64/bin/tf-inkling2-exl3-load /checkpoint/rank0 0 2 48
out-arm64/bin/tf-inkling2-exl3-load /checkpoint/rank1 1 2 1
out-arm64/bin/tf-inkling2-exl3-load /checkpoint/rank1 1 2 48
```

Each command ran in its own process with a 420-second bound and a 4 GiB host-memory floor. Both one-layer
canaries and both complete 48-layer loads exited zero. The checkpoint files were read without conversion.

## Results and limits

- Rank 0: 102934272000 owned device bytes (95.865 GiB), 48 layers. [Raw output](gb10-rank0.txt).
- Rank 1: 103340365824 owned device bytes (96.243 GiB), 48 layers. [Raw output](gb10-rank1.txt).
- These counts cover the loader's owned weight allocations, not total GPU/process footprint or model KV/workspace.
- All probe processes exited; the preceding production pair was restored and passed a real completion check.
- The full CPU host aggregate passed 2138/2138 steps: 20394 tests passed, 141 skipped, plus cached checks.
- Allocation-failure cleanup has not been qualified. Successful process exit is not that proof.
- No forward, draft, token/state equality, model fidelity, speed or full-window quality result is claimed here.
- Only GB10/SM121 was exercised. Other CUDA targets, Metal, HIP and Windows were not run locally.

The ARM64 loader executable's SHA256 is recorded below.
e8d8e5d64fc7be8b50b9b09338437cc3e5aec29c22c5135b27d9d55a4a01fcc1  tf-inkling2-exl3-load

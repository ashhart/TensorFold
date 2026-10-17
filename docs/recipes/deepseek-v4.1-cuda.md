# DeepSeek-V4.1-Flash on CUDA (Zig): tools and checks

The `deepseek_v41` family runs on the native Zig engine with its own CUDA kernels (`zig/kernels/cuda/deepseek_v41/`).
Its development tools are Python and live outside this repository, in jayleaton/deepseek-v41-tensorfold-spark,
pinned to commit c1cb91840fb8:
[`tools/tensorfold-dsv41/`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/tree/c1cb91840fb8ff5db0bd27418fc7cc766a222cbc/tools/tensorfold-dsv41).
That folder's README lists each tool's dependencies (the TensorFold Python engine, PyTorch, Triton, a GPU where
noted) and `MAP.tsv` names the TensorFold path each file came from; its `check.py` runs the tools' own checks without
a GPU.

Nothing in `zig build` or `zig build test` runs Python. The checks below read fixtures committed here; references,
captures, AOT compiles, SASS identity and GPU measurements are separate receipts made with the tools on real hardware.

## SASS identity of the kernel copies

The family's kernel copies (line 1 of each names its source) must compile to the reference extensions' SASS. With
Docker and `nvcr.io/nvidia/pytorch:26.07-py3`, from a TensorFold checkout:

```
OUT=/scratch/sass SM=121 TF_DSV41_KERNELS=$PWD/zig/kernels/cuda/deepseek_v41 TF_DSV41_REF_SRC=/path/to/reference \
  /path/to/tools/tensorfold-dsv41/kernels/sass_check.sh [name ...]
```

`TF_DSV41_REF_SRC` (and optionally `TF_DSV41_REF_SRC_R1` / `_X3GM2` with `TF_DSV41_REF_REV*`) names the reference
repository and revision; the check stops before building without them.
`-Dnvcc=zig/kernels/cuda/deepseek_v41/nvcc-docker.sh` builds the images themselves through the same container.

## Stored fixtures checked natively

Paths under `zig/src/families/deepseek_v41/`. Host suites (`zig build test-dsv41`, part of `zig build test`; no GPU):
- `fixtures/widths-exl3-2p8.txt` and `fixtures/config.json`: a 2.8 bpw pack's expert widths and release config
  (`host_test.zig`).
- `fixtures/lanes-ref.bin`: the run-time lane layout (`prepare.zig`).
- `fixtures/engram-hash-ref.bin` and `fixtures/engram-rows-ref.txt`: Engram's hasher and rows (`engram_host.zig`,
  `engram_rows_test.zig`).
- Not stored: the device kernel fixtures (`cuda_tests/oracle.py`, `cuda_tests/dsv41_capture.py` in the tools)
  that `tf-dsv41-test all DIR` and `tf-dsv41-test replay DIR` compare on a GPU.
- `kv/sched_golden.json`, `kv/session_slots_golden.json`, `kv/session_slots_split_golden.json`: the scheduler and
  session slots against the reference (`kv/sched_test.zig`, `kv/session_slots_test.zig`; `zig build test-kv`).
- `fixtures/dense-k2.txt`, `fixtures/expert-k2.txt`: the K2 width listings the emitters and plans are tested on;
  `fixtures/chain-op-cap16.json`: the drafting chain's op cap (`slot_drafts_chain_test.zig`).

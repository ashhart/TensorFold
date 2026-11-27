# A9 kernel controls — the committed controls launch the real CUDA kernels

Owner ask A9 (the 04:59Z round): the committed tests recompute Zig references without
launching the CUDA kernels. This directory is the kernel-launching layer over those
same three controls, with committed fail-before injectors.

## What each control is

| control | kernels launched (real, `qwen35_draft_lilicorr.cuh` verbatim) | face |
|---|---|---|
| FACTOR | `tf_lilicorr_widen` + `tf_lilicorr_factor` | rows 56, hidden 1024: three dot terms + bias + silu, bf16 out, vs a f64 host mirror of the exact expression |
| TAIL | `tf_lilicorr_factor_rows` | hd=6 rows=9 (tail warps t=6,7 alive as load workers) — closed form `v = t + 6r - 8.5` for EVERY row, plus the committed C2 goldens (rows 0-2) |
| CHAIN | `tf_lilicorr_rms` + `tf_gemv_bf16_rows` (in-proj) + `tf_lilicorr_lat_attn` + `tf_lilicorr_l2norm` + `tf_lilicorr_dots` | the drafter-head chain at real head shapes (qkv 56x3072, lattice 56, head_dim 128); feeds are device-resident — no host round-trip between stages; start/pair scores vs a f64 mirror |

Face bars: 2 bf16-ULP of the host reference (f64) + a 1e-4 floor (FACTOR), 5e-3 abs on
the closed form + the goldens (TAIL), 5e-3 rel + 1e-3 abs floor (CHAIN). The chain also
asserts determinism (second run into fresh buffers must be bit-identical).

Disclosed scope limits (see the .cu header): the CHAIN edges use the flat
`stride=1024` dots face (OUT_VEC and IN_VEC alias one row-per-1024 buffer) rather than
the interleaved [56][2048] live face; the chain is one full pass of the score chain, not
the whole head + walk (the walk is a host-graph property and stays covered by the
committed Zig cell).

## Fail-before injectors (committed, deterministic, non-UB)

| injector | defect re-introduced (recorded class) | expected RED face |
|---|---|---|
| `a9_inject_factor.sh` | the cross term's W-slab folds onto the anchor-term slab (`w_cross = w_row + hd`) — the slab mis-index class | every FACTOR cell |
| `a9_inject_tail.sh` | inactive tail warps abandon the shared-row load duty ("exiting them left whole Ain rows uninitialized" — the 11/25 A1 fuzz class) without its barrier violation: warps 6,7 skip the loads but stay for every `__syncthreads` | only TAIL rows r in {6,7} (the row band only threads 192..255 fill) |
| `a9_inject_chain.sh` | the S4-era l2norm tail width verbatim (`width = ... : stride`), the kernel's own comment names it "a real normalization defect" | the anchor tail's inv carries the over-run's stale ssq -> every CHAIN start score wrong |

The injectors sed a SCRATCH COPY of the kernel header; the repo tree is never modified
and the RED is a wrong-answer RED (no crash, no hang, no allocation accident).

## How to run (the two-runner law)

From any checkout of this commit, on the GPU box (nvcc + device needed):

    bash zig/tests/cuda/lilicorr_kernels/a9_run.sh runner1

The runner compiles the harness against a scratch copy of the header, runs the clean
GREEN matrix (all controls MATCH, rc 0), then for each control applies the committed
injector, recompiles, and grades the RED matrix (that control FAILs, the other two stay
MATCH). Exit 0 = every face graded true.

The two independent runners bar is met by running the SAME script from two separate
checkouts of this commit, e.g.:

    bash zig/tests/cuda/lilicorr_kernels/a9_run.sh runner1            # workspace checkout
    git worktree add /tmp/a9_runner2 HEAD
    bash /tmp/a9_runner2/zig/tests/cuda/lilicorr_kernels/a9_run.sh runner2 /tmp/a9_runner2

With the HEAD commit checked out, the clean runs ARE "GREEN at HEAD"; the injector runs
ARE the committed fail-before RED. The graded faces land as logs next to the script
(`a9_<label>_<control>_<clean|inject>.log` + the injected header copy).

## Relationship to the committed Zig cells

The Zig fixtures in `cuda_lilicorr_test.zig` stay: they pin the reference math and the
source contracts. This directory adds the layer the ask names: real kernel launches on
real GPU inputs, injectable to a deterministic RED on the same compiled control.

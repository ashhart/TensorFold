# Liquid CUDA kernels: SASS identity against main

Regenerated after #1508 merged main `af9777b49` (merge `2fc30ddf1`), as asked in review. Main carries the shared CUDA
headers (`zig/kernels/cuda/common`), #1492 and #1503, so this compares that main with the merge head `2fc30ddf1` (the
commit before this receipt; nothing under `zig/kernels` changes after it). The receipt before it, at the rebase onto
`b4f1bf263` with head `cde825799`, gave the same table.

`python3 cuda_sass_identity.py --repo . -j 1 --receipt sass.tsv af9777b49 2fc30ddf1`, nvcc 13.0 (V13.0.88), sm_80,
86, 89, 120 and 121, with the tool as main had it at 1.1.9 (`git show 465e48b62:tools/zig/cuda_sass_identity.py`), as
for the first receipt: today's `tools/zig/cuda_sass_compare.py` refuses a catalog addition (the `lfm3` image) and stops
at the first added function, so it can't write this receipt. `sass.tsv` has every function's SASS sha256 on both
sides; `summary.txt` is the tool's output.

| image | functions on main | identical | changed | added |
|---|---|---|---|---|
| prefill_attention | 70 | 70 | 0 | 5 (`pattn_kernel<128, 8, 4, 8>`) |
| contract_controls | 450 | 450 | 0 | 15 (`tf_ctl_liquid_conv_7`, `tf_ctl_liquid_commit_7`, `tf_ctl_pattn_128`) |

Each added function is on 5 SMs. `nemotron_attention` is no longer reached: its head-64 segments came in with #1503.
`lfm3` is a new image, on one side only. nvcc 13.2 gives the same verdict: the same 450 + 70 identical and the
same 20 added (`summary-nvcc13.2.txt`).

At the rebase, two `-j 2` runs of the trees then, made while other nvcc builds ran, each saw one more function differ:
main's `tf_ctl_moe_router_tiles_row` control on the head side, on sm_89 once and sm_86 once, with a different digest
each time. The `-j 1` runs then and now, and the nvcc 13.2 runs, show it identical. That source is main's, byte for
byte, and this branch doesn't touch it. The 1.1.9 tool has no second pass to flag unstable functions, as today's has.

The tool exits 1 because it counts added functions as differing. Its `warp_sum` control changes 35 functions, so a
changed source does show. Its `fp4pair` control can't be planted on today's `common/fp_pairs.cuh` (pattern found 0).

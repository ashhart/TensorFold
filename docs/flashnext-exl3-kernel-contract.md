# Flash Next EXL3 kernel contract

Flash Next EXL3's canonical expert reduction starts in the one-rank slice: `fn_exl3_leaf_tree`,
`fn_exl3_leaf_partial` and `fn_exl3_leaf_finish` register in `contract_flashnext_exl3_canon.zig`. Leaves,
tree additions and transported partials are float64; only the root converts once to float32, round-to-nearest-even.
Outputs match the host tree bit for bit (NaN payloads aside), including float64's exceptional classes.
Controls flatten the tree, drop NaNs, write an input, use bf16 adds or narrow partials to float32; each must fail
its intended gate. Independent integer-product references check error and TP independence without a sampled-error
tolerance. Checkpoint BF16 state boundaries are unchanged.

The three Liquid-family planted faults live in `contract_liquid_controls.cuh`, included by `contract_controls.cu`;
the NVCC depfile tracks that header. Their entry points, launches and fault arithmetic are unchanged.
The row-major prompt control supplies the production caller's `P256=false`, `IDS=false`, `PN=0` policy to the shared
row body; its neighbouring-dimension fault is unchanged.

CUDA also checks Flash Next EXL3's canonical expert reduction (`flashnext_batch.cu`): the float32 root,
float64 `fn_exl3_leaf_partial` over leaves 0-31, 32-63, all 64 and an interior subtree, and
`fn_exl3_leaf_finish` over the pieces of 2, 3, 5 and 8 ranks (rows 1, 2, 5 and 16 at width 2560).
Each output is byte-equal to `exl3_canon.zig` (NaN payloads aside), has float64's exceptional class and
satisfies its numerical bound. Eight controls flatten a tree, drop NaNs, write an input, use bf16 adds,
or round a partial through float32.
Leaves and fixed-tree merges stay float64, including across the typed NCCL all-gather; the root alone
rounds to float32. Every TP degree 1 to 8 gives the same bits. Exact integer-product host references
bound root error by `2^-24 |sum| + 18 * 2^-53 sum(|y w|) + 2^-150` and require every sampled row to lose
no accuracy to the slot chain or rank-split sums, without an error-comparison tolerance.

CUDA also checks Flash Next's EXL3 deep prefill: its rotate and gate-up pass (`tf_exl3pf_rotate`, `tf_exl3pf_gu`) and
its down pass (`tf_exl3pf_down`), on rows of ten routed picks and the shared expert at 1 to 1,408 rows, each bit-equal
to the grouped decode form of `exl3_experts` it replaces (the planted control runs the gate-up pass against a decode
form with the other SwiGLU roundings); gate 4 runs zero trellis words, whose one dyadic weight makes every dot an exact
FP32 integer. Its dense prompt tile (`exl3_candidate.cu`'s `tf_exl3_promptd_0`, `_1` and `_2`, one a codebook) is
bit-equal to `linear_body` at 1 to 2,048 rows (the 64-row block's and the 256-row chunk's edges included) at every
width of 1 to 8 bits, on each form `plan()` gives (a K split or none, 4 or 8 warps) and on a family's fixed plans (one
split of 4 warps, two of 8); gate 4 runs all-ones states, one weight at every width, where an fp16 sum overflows. Four
controls: the tile at another K split, one a codebook, and one that zeroes x's NaN and infinities (red at gate 1).

CUDA also checks the deep prefill's rotate and expert-parallel reduction (`flashnext_exl3_prefill.cu`,
`tf-cuda-test exl3-tppf-contract`): `tf_exl3pf_rotate` against a host fp32 Hadamard, bit for bit;
`tf_exl3pf_tp_combine` over degrees 1 to 8 into eight-byte float64 partials; `tf_exl3pf_tp_finish`
over degrees 2, 3, 5 and 8 with float64 merges and one float32 rounding; and `tf_exl3pf_tp_place`
writing only slot zero and its unit weight. Probes retain sub-blocks, table boundaries, masked and
foreign picks, exceptional classes and the canonical numerical bound, through 4,096 rows.
Gate 4 stresses rotate's cancelling transform and finish's shuffled integer pairs; combine's explicit
float64 check makes gate 4 inapplicable, and place has no sums. Sixteen controls plant rotate's NaN,
input, table-sign and bf16 faults; combine's flat order, dropped NaN, input write, bf16 adds and
float32 partials; finish's flat order, bf16 merges, input write and float32 partials; and place's
NaN, input and weight faults. `test-flashnext-tppf-contract-host` uses the same oracles and checks
cross-degree equality, including cancellation whose exact sum is 2.

Flash Next's 13 EXL3 launches (`flashnext_exl3_pending.zig`) carry only an nvcc 13.4 receipt: they stay pending under
13.0 to 13.3 and take their 13.4 usage from `cuda_134`, which `pending_132.lookup` serves for that release.
The first, `fn_exl3_split_f16`, also runs gates 1 to 4 on the CUDA runner (`contract_flashnext_exl3.zig`): an fp16
row splits into bf16 hi = bf16(x) and lo = bf16(x - hi), every output's bits must equal float64's (NaN stays NaN; an
infinity gives hi = Inf and lo = NaN), and the device's hi + lo must equal x for every finite pattern. Its shapes
hold all 65,536 fp16 patterns and partial last blocks (n = 1,000 and 65,537). Three controls go red: one flushing NaN
and Inf to zero (gate 1), one launching whole blocks only (gate 3, at every tail) and one writing hi over its input
(gate 2). Gate 4 is N/A: it has no sums.
The other 12 launches of the set also run gates 1 to 4 on the CUDA runner (see the coverage above): the canonical tree
and finish, the pick mask, the prefill's rotate, gate-up, down, combine, finish and place, and the prompt tiles.

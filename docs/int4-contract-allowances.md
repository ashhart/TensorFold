# int4 contract allowances

`fn_int4.cu`'s gate/up form rounds its gate sum, its up sum and silu to bf16 before the store, and the kernel contract
compares each output with float64, so those roundings need an allowance. This page derives that allowance, and gate
4's bf16 margin for the same form, from the kernel's arithmetic alone. Every constant below is a format constant, a
property of the reduction, or the accuracy the CUDA library states for `expf`. No number here comes from a device
run, and none changes with one: the device lines in the pull request are receipts of this arithmetic, not inputs to it.

The contract's rule for every allowance is the same: it is the worst case of the arithmetic the kernel performs,
written before any output is compared with it. Where the worst case of a stated form differs from the generic formula
in `contract.zig`, the registration derives its own (`VTable.bounds`), and this page is that derivation.

## What one gate/up output computes

Source: `zig/kernels/cuda/fn_int4.cu`, `pair`, `compute_stage` and `store` with `EPI` 2.

- A stored code `q` (0 to 15) enters a bf16 MMA as `128 + q`, and `136` is subtracted in bf16. Both are exact and
  leave `q - 8`. A product `x (q - 8)` has at most 12 significant bits, so fp32 holds it exactly.
- A group of 128 inputs is four k32 blocks in order, each two k16 MMA steps in order, each step `D = A B + D`; then
  `acc = fmaf(group sum, scale, acc)` with the fp16 scale widened exactly. The longest chain of roundings a term
  passes is `k / 16 + k / 128 + 36`: one MMA step an input's 16 products, one fma a group, and one step's error over
  its own products (`attention_range.mma_step`, the count every CUDA expert registers).
- The epilogue, with `u_b = 2^-8` for bf16's unit roundoff: `gv = bf(g)`, `uv = bf(u)`,
  `act = gv / (1 + expf(-gv))` in fp32, `s = bf(act)`, `p = s uv` in fp32, `out = bf(p)`.
- The reference is `want = silu(g*) u*` with `silu(x) = x / (1 + exp(-x))` in float64 and `g*`, `u*` its sums.

With `v = 2^-24` and `gamma(n) = n v / (1 - n v)`, the fp32 sums satisfy `|g - g*| <= gamma(depth) A_g` and
`|u - u*| <= gamma(depth) A_u`, where `A_g`, `A_u` are the sums of the terms' magnitudes.

## The allowance, term by term

`zig/tests/cuda/int4_ref.zig` `gateUp`. Each line says which rounding it bounds; `s = |silu(g*)|`.

| term | bounds |
|---|---|
| `dg = u_b \|g*\| + gamma(depth) A_g (1 + u_b)` | `\|bf(g) - g*\|`: the sum's error, then bf16's rounding of the value it landed on |
| `du` | the same for the up sum |
| `d = max \|silu'\|` on `[g* - dg, g* + dg]` | the slope: at the interval's ends and at its turning points, `+-2.3993572805` |
| `span = \|u*\| + du` | the largest up value the rounded silu meets |
| `d dg span` | the gate's fp32 error and its bf16 rounding, through the slope and times the up value |
| `u_b s span` | silu's bf16 rounding, at silu's own size |
| `(s + d dg) du` | the up sum's fp32 error and its bf16 rounding, times silu |
| `(1 + 2 u_b)` on the three lines above | those roundings evaluated at `s + d dg` rather than `s`, and the fp32 operations |
| `+ (s + d dg) span` where `g - dg < -88` | `expf` overflows fp32 above 88.72: the kernel's silu is 0 where float64 keeps a value |
| `5 v \|want\|` | `expf` (2 ulps, the CUDA library's bound), `1 + e`, the quotient, the product |
| `+ u_b (\|want\| + work)` | the store's bf16 rounding of a value already `work` from `want` |
| `+ 2^-133` | bf16's smallest step near zero (`outFloor`) |

Why the epilogue lines cover their roundings: `act` is within `5 v` of `silu(gv)` in relative terms, so `s` is within
`(u_b + 5 v) (s + d dg)` of `silu(g*)` once the gate's own `d dg` is counted, and `(d dg + u_b s) (1 + 2 u_b)` is
larger than `d dg + (u_b + 5 v) (s + d dg)` because `u_b d dg + 2 u_b^2 s` exceeds the `5 v` terms. The product of
two bf16 values is exact in fp32, so the product's rounding is counted and never spent. Each sum's error is counted
once: inside `dg` and `du`, where it enters.

## Why the generic bound is no bound for this epilogue

`contract.zig` `bound` counts the roundings between the sums and the store at the output's size: `mids u_b |want|`.
That is right when the rounded value has the output's size. The gate has not: the kernel rounds `g`, and that
rounding reaches the output as `u_b |g| |silu'(g)| |u|`, which is `u_b |want| |1 + g (1 - sigmoid(g))|`. The factor
is at most 1.1 for a positive gate and about `|g| - 1` for a negative one: 1.86 at `g = -3`, 9.5 at `g = -10.5`. So
for every gate below about `-2.2` the generic bound is short of the epilogue's own worst case by
`(factor - 1) u_b |want|`, before any sum error is considered. It also has no term for the up sum's fp32 error
carried by `silu(g)`, `gamma(depth) A_u |silu(g*)|`.

One stated gate shows the first gap with exact sums and no device (a host test in `int4_ref.zig`): at `g = -10.53`,
`u = -28.75`, `bf(g)` is `-10.5`, and that one rounding moves the output to 1.76 times `boundExact`; the whole
epilogue lands at 2.15 times it, and at 0.66 of the derived allowance.

The generic bound's absolute term, `gamma(depth) A_g (|u| + 1)`, hides the gap at the served width, where it is larger
than everything else. At the least `k` it is about 100 times smaller and the gap is the whole story. The gap needs the
roundings to agree in direction, so the form passes the generic bound on most shapes and fails it on one that holds
enough outputs: a bound that sometimes holds is not a bound.

The derived allowance holds on every output by construction, and it is the generic bound wherever the generic bound
was right. For a gate above 1 it is 1.05 to 1.09 times the generic bound at the served width: the up sum's error
through `silu(g)`, and nothing else. For a gate below 0 it is smaller, down to orders of magnitude smaller for a
strongly negative gate, whose output is tiny while the generic term still grows with `A_g |u|`.

## Gate 4's bf16 margin

Gate 4 replays each stress row's sums in fp16 and bf16 and requires each to miss by at least the registered factor
times the exact-sum allowance, so that a kernel summing in that precision cannot pass. The default factor is 100.

On the stress rows a gate sum and an up sum are 4 times eight codes of 1 to 3: 32 to 96, with `d = 1` and exact sums.
The allowance at `exact` is `((1 + u_b) (3 + 4 u_b) u_b + 5 v) (1 + u_b) + u_b` times `|want|`, which is
`4.039307 u_b |want|`. A low-precision sum that returns 0 misses by `|want|` itself, which is
`1 / (4.039307 u_b) = 63.38` times the allowance. That is the most any input can prove of a bf16 SwiGLU output whose
low-precision sum is 0, and a mirrored row makes it 0: in the halves order the two halves of the row are rounded
alike with opposite signs, so they cancel and the bf16 sum is 0 unless a rounding boundary falls between them, where
it is one bf16 spacing. An output misses by more than `|want|` only when both of its sums keep such a residual.
100 asks for a miss of `1.57 |want|`, which no sum returning 0 makes. fp16 overflows on these rows, where the partial
sums reach `5.6e6`, so its 100 holds.

The registration states 60, the largest multiple of 10 under the 63.38 ceiling, leaving 5 percent for the constants
above. The same registration shape is the relu^2 up of `experts_bf16.cu`, whose three bf16 roundings allow `3 u_b`
and whose stated 30 sits under its own ceiling of 85. The generic `boundExact` gives 63.81 on the same rows, so the
margin did not move with the allowance: it was 100 against a ceiling near 64 before the allowance was derived.

The stress rows themselves are integers from 8 to 2,040 with the eight nudged inputs at 4 to 16. Every partial sum
stays below `2^24`, so fp32 is exact in any order, and the larger the integers the further the fp16 and bf16 chains
land from the small result.

## The controls

The gates must still go red under this allowance, and they do, by arithmetic first:

- `tf_ctl_int4_swiglu` at fault 0 is the epilogue's scalar model: the same group sums in input order and the same
  epilogue, registered as a kernel. It passes gates 1 to 4, so the faults below are the only difference between it
  and the controls.
- Fault 1 takes silu of the up sum times the gate sum (gate and up swapped). Silu is not symmetric: where the gate
  is negative its silu is tiny, and the swapped output is the up sum's silu times a gate of full size.
- Fault 2 cuts gate, up and silu toward zero where the kernel rounds to nearest: one rounding mode away from the
  kernel. Its four errors all shrink the output's magnitude, each by up to twice the unit the allowance grants that
  rounding, so on a shape of any size some outputs are outside the allowance. The first served shape holds 3,840.
- Fault 3 rounds each group's sums to bf16 before their scales (gate 4): the group sums of a stress row reach
  `2.5e5`, where one bf16 rounding is hundreds, against an exact-sum allowance near 65.

Both gate 3 controls are outside the generic bound of the kernel's first commit too, so they were red before the
allowance was derived and stay red after it.

## What pins it on the host

`zig build test-int4-host` runs `int4_ref.zig`'s tests with no device:

- the derived allowance covers 4,000 epilogue draws, and the generic `boundExact` is short on some of them;
- the stated gate above: outside the generic bound twice over, inside the derived allowance;
- gate 4's ceiling on every stress sum of the form: between 60 and 64, and under 64 with the generic `boundExact` too;
- the planted faults outside the allowance on 4,000 seeded draws: the cut epilogue on some, the swap on most.

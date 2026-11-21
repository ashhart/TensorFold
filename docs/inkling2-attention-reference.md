# Inkling 2 decode attention: arithmetic reference

This page fixes the arithmetic of Inkling 2 decode attention: the values, where each one rounds to bf16, the order of
every fp32 sum, what NaN and infinity do, and a bound for an fp32 result against float64. Beside this page, the
reference is a float64 host model (`zig/src/families/inkling2/attn.zig`) with an fp32 model in that order, an oracle
(`zig/src/families/inkling2/oracle/attn.py`) and host tests (`zig/src/families/inkling2/attn_test.zig`). It is a
TensorFold reference, written from `transformers` 5.23.0 `models/inkling2/modeling_inkling2.py` (cited as TF523:line).
It is a reference for review, not a claim that fp32 scores or P give the same tokens as higher precision; token-level
gates stay with the engine.

## Scope

One call serves `rows` consecutive decode rows at positions `pos0 .. pos0 + rows - 1` over a dense cache of positions
`0 .. pos0 + rows - 1`. Per row and query head it gives `hd` outputs. A sliding layer keeps the 1024 newest keys
(`window = 1024`), a global layer keeps all keys (`window = 0`). Not in scope: the log-tau scaling of `q` and `rel` in
global layers (TF523:589-602), which comes before this step, the q/k/rel clips, the output projection, and keys beyond
8192, which the host model refuses.

## Values and where they round

Notation: `bf(x)` is round-to-nearest-even to bf16; every input (`q`, `k`, `v`, `rel`, `proj`, `sink`, `gate`) is a bf16
value held exactly in fp32. Query head `h` reads key/value head `h / (heads / kv_heads)` (TF523 `repeat_kv`, :380-381).
For row `t` and key `s` with `dist = t - s`:

| Step | Formula | Reference | Rounded to bf16 |
|---|---|---|---|
| dot | `a = sum_i q[i] * k[i]` over `hd` | TF523:383 `torch.matmul` | yes, `bf(a)` |
| scale | `b = bf(a) * hd^-0.5`, the scale as an fp32 constant | TF523:383 `* scaling`, :414 | yes, `bf(b)` |
| bias table | `r = sum_j rel[j] * proj[j][dist]` over `d_rel` | TF523:184 | yes, `bf(r)` |
| bias | `r` only where `0 <= dist < rel_extent`, else 0 (no clamped column) | TF523:186-188 | n/a |
| logit | `l = bf(b + r)` | TF523:385 | yes |
| sink | `sink` is a bf16 logit per row and head, with a zero value | TF523:389-392, :453/:577 | input |
| softmax | `p_s = exp(l_s - m) / (sum_s exp(l_s - m) + exp(sink - m))`, `m = max(l_s, sink)`, in fp32 | TF523:391-392 | `p_s = bf(p_s)` |
| values | `o = sum_s p_s * v_s`, the sink contributes no term | TF523:396 `torch.matmul` | yes, `bf(o)` |
| gate | `y = bf(o * sigmoid(gate))`, `sigmoid` in fp32 | TF523:496 (sliding), :628 (global) | yes |

On `rel_extent`: sliding layers use `rel_extent = sliding_window_size` (TF523:413, :512), so every key in the window has
a bias; global layers use `config.rel_extent`, and a key farther away than `rel_extent - 1` gets no bias. The window
rule is `t - s < 1024` (mask from `create_sliding_window_causal_mask`, TF523:40, :1291; `masking_utils.py`:95).

The host model's flag `rounded` selects between this table (all bf16 roundings, bf16 output) and the same arithmetic
without any bf16 rounding (fp32 output), which the bound below covers.

## Order of every sum (fixed, no atomics)

The fp32 model adds in the order of a block of 128 threads, four warps of 32 lanes, with every operation rounded to
nearest (an `fma` once):
- Dot over `hd`: lane `l` accumulates `fma(q[d], k[d], acc)` for `d = l, l + 32, l + 64, l + 96` in that order; the
  warp sum is an xor tree with offsets 16, 8, 4, 2, 1.
- Bias over `d_rel`: lane `j < d_rel` holds the product `rel[j] * proj[j][dist]`, the same xor tree sums the lanes
  (lanes `>= d_rel` hold 0).
- Maximum `m`: the largest of the sink and the logits under `fmaxf`'s rule, so the order of the comparisons does not
  change it.
- Denominator `Z`: each thread adds `exp(l_j - m)` for `j = tid, tid + 128, ...` in that order; xor tree per warp; then
  `(w0 + w1) + (w2 + w3)`; then `exp(sink - m)` is added last. `exp` is CUDA's `expf` (its error: see the bound below),
  and `p = exp / Z` is one quotient.
- Values: for each output index `d`, one sequential `fma(p_j, v_j[d], acc)` over the keys in position order.
- Gate: `sigmoid(g) = 1 / (1 + expf(-g))`, then one product with `o`.

## NaN and infinity

IEEE propagation, with two places that need a stated rule:
- `fmaxf` passes a NaN by, so a NaN logit or a NaN sink does not stop `m`; the NaN's `exp` then makes `Z` NaN, so every
  output of that row and head is NaN (host reference: `attend`, tested for a NaN sink and a NaN key).
- A `+inf` sink makes the maximum infinite and `exp(sink - m)` is `exp(inf - inf)`, so the row's outputs are NaN; a
  `-inf` sink contributes 0 (host reference: `attend`, tested for both).
- A non-finite value `v` stays non-finite where its weight is positive (`inf * p`); a zero weight times an infinite
  value is NaN, as in `torch.matmul`.
- An infinite gate saturates `sigmoid` to 1 or 0 (finite); a NaN gate makes the output NaN.

## FP64 check and its bound

The float64 model (`zig/src/families/inkling2/attn.zig`) evaluates the same formula in float64 with exact bf16 inputs
and no intermediate rounding. The oracle evaluates the formula in float64 with Python's standard library, and runs
`transformers`' own `Inkling2RelativeLogits` (TF523:165-188) and `eager_attention_forward` (TF523:368-399) on the same
bf16 inputs; the causal and sliding mask (`masking_utils.py` of the same release, :76, :95 and :630, as TF523:1290-1291
builds it) and the output gate (TF523:496, :628) it writes out from the source. Its fixture keeps both results. The host
tests hold the float64 model to the oracle's float64 values within `(|y| + top) 2^-40` per element, `top` the head's
largest `|y|`, the fp32 model without roundings to them within `(|y| + top) 2^-16` at least twice over, and the fp32
model with the table's roundings to the bf16 bits of the `transformers` path (at most one value in a thousand one step
apart). They do not evaluate the bound below. For an fp32 result `y` in the order above, with finite inputs and no fp32
value that overflows (this rules out a gate of -89 or below, where `expf(-g)` overflows and `sigmoid` gives 0), the
bound against the float64 output `want` is

`|y - want| <= (1 + 2^-20) * (mag * (exp(L) - 1) + (nk + 1) * (vmax + 1) * 2^-145)`

`L = 2 eta + 3 lambda(gamma(5)) + 4 lambda(u) + S`

`S = lambda(gamma(nk / 128 + 8)) + lambda(gamma(nk)) + lambda((nk + 1) 2^-145)`

with `u = 2^-24`, `gamma(n) = n u / (1 - n u)`, `lambda(a) = -ln(1 - a)`, `nk` the number of keys of that row,
`mag = sum_s (p_s |v_s|) * sigmoid(gate)` the magnitude of the output's terms, `vmax` the largest `|v|` over the row's
keys and the head's values, and `eta = dl + u (m - min(sink, l_s) + 2 dl)`, with float64's maximum `m` and logits `l_s`,
the largest error of an fp32 exponent `l_s - m` or `sink - m`. `dl`, the largest error of a logit in the row, is the
largest `gamma(13) * scale * sum|q k| + gamma(7) * sum|rel proj| + 2^-141` over its keys, the `gamma(7)` term only for a
key that gets a bias. Each rounding of the order above is counted:
- `gamma(13)`: a product `q k` passes at most four `fma`s in its lane's chain (every `hd` up to 128), five tree adds,
  the scale constant (an fp32 rounding of float64's `hd^-0.5`, within two roundings of the exact value), the scaling's
  product and the bias add.
- `gamma(7)`: a bias product passes its own rounding, five tree adds and the bias add (every `d_rel` up to 32).
- `eta`: an fp32 exponent `l_s - m` carries its logit's error (`dl`) and one rounding of the subtraction; `sink - m`
  carries only the rounding. The fp32 maximum (`fmaxf`, no rounding) lies within `dl` of float64's, so that rounding is
  at most `u (m - min(sink, l_s) + 2 dl)`. Both errors count against float64's values less the fp32 maximum, a shift
  that leaves `p` unchanged.
- `2 eta`: an exponent's error moves its own term and the terms of `Z` (numerator and denominator).
- `3 lambda(gamma(5))`: NVIDIA's CUDA Programming Guide (v13.5, section 5.6.7, Table 54) lists `expf` with a maximum
  error of 2 ULP and states: "The maximum ULP error is stated as the maximum observed absolute value of the difference
  in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision
  obtained according to the round-to-nearest ties-to-even rounding mode." and "The error bounds are derived from
  extensive, though not exhaustive, testing. Therefore, they are not guaranteed." Within 2 ulp of the correctly rounded
  `e^x`, `expf` is within `gamma(5)` of `e^x`; it counts in the numerator, in the terms of `Z` and in `sigmoid`.
- `4 lambda(u)`: the quotient `p`, the add and the quotient in `sigmoid`, and the final product.
- `lambda(gamma(nk / 128 + 8))`: a term of `Z` passes at most `ceil(nk / 128) - 1` rounded adds in its thread's chain
  (the first, onto 0, is exact), five in the xor tree, two combining the warps and the sink's add.
- `lambda(gamma(nk))`: the value chain of `nk` `fma`s.
- Subnormal results: below `2^-126` an `fma`, product or quotient is within `2^-150` of its exact value instead of `u`
  relative, `expf` within `2^-147`, and an add there is exact. This gives the `2^-141` in `dl` (at most 161 products and
  `fma`s feed a logit), `lambda((nk + 1) 2^-145)` for `Z` (its largest term, `expf(0)`, keeps it above 1/2) and the term
  with `vmax`. In `sigmoid`, `2^-147` is below `gamma(5)` of `1 + exp(-g)`.
- The factors multiply: `lambda(a)` bounds `|ln(1 + d)|` for `|d| <= a`, and `lambda(gamma(n))` does so for `n`
  roundings, so `L` bounds the logarithm of each term's relative error and `exp(L) - 1` the error itself.
- `1 + 2^-20` covers float64, in `want` and in evaluating the bound: each float64 step rounds at `2^-53`, its `exp`
  within 1 ulp and the scale as `1 / sqrt(hd)`, and its sums are at most 128 times as long as the fp32 chains, so its
  error stays below `2^-22` of the bound.

The bound is derived from these counts and that `expf` bound, not fitted to runs.

With the bf16 roundings the bound is an interval: the order above run in fp32 with each `expf` over its whole range
under that bound (within `gamma(5)` of `e^x`, or `2^-147` below `2^-126`), with every stage rounded to bf16 as in the
table, gives the interval a correct output lies in. No claim of bit equality with PyTorch is made.

## Falsification

Each of these single changes to the host model turns the host tests red: a rounding left out (the dot, the scaled dot,
the bias table, the biased logit, `p`, the value sum, the gated output), a window one key too wide, the sink left out of
the sum, the sink taken at half its weight, the bias applied one distance too far, the scale or the gate missing, the
wrong key-head group, a value sum in reverse key order, each shape or length guard on its own, the maximum taken
without the sink, and the sink's term clamped below at -1.

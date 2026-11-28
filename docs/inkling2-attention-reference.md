# Inkling 2 attention: arithmetic reference

This page writes down the arithmetic of Inkling 2's attention as #4125's kernels run it: the values, where each one
rounds to bf16, the order of every fp32 sum, what NaN and infinity do, and a bound for an fp32 result against float64.
Beside it are a float64 host model (`zig/src/families/inkling2/attn.zig`), the fp32 model in the kernels' order
(`zig/src/families/inkling2/attn_order.zig`), the bound as code (`attn_bound.zig`), an oracle
(`zig/src/families/inkling2/oracle/attn.py`) and host tests (`attn_test.zig`, `attn_bound_test.zig`,
`attn_order_hand_test.zig`). The kernels are the reference: the host model follows their source, it does not run them.
Every statement about a kernel's source carries `file:line` at commit `6dbb32f048a6` (#4125 as merged), in the files
`zig/kernels/cuda/inkling2/attention.cu` (`attention.cu`), `prompt_attention.cu`,
`zig/src/families/inkling2/kernels.zig` (`kernels.zig`), `zig/build/cuda_kernels.zig`, `zig/build/cuda_fatbin.zig` and
`zig/tests/cuda/contract_inkling2_attn_ref.zig` (`contract_inkling2_attn_ref.zig`). Statements about `transformers`
5.23.0 cite `models/inkling2/modeling_inkling2.py` as TF523:line. Nothing here claims that fp32 scores or P give the
same tokens as higher precision; token-level gates stay with the engine.

## Scope

The page covers `tf_ink2_rel_table`, `tf_ink2_attn`, `tf_ink2_attn_fold`, `tf_ink2_attn_prompt` and `tf_ink2_merge`:
from the q, the key and value caches, the rel taps and `rel_logits_proj`, the sink and the gate, as bf16 values, to the
bf16 output per row, query head and output index. A sliding layer keeps the 1024 newest keys, a global layer all keys.
Query head `h` reads key head `h / (heads / kv_heads)` (`attention.cu:159`, `:182`). The prompt kernel runs the same
arithmetic as the others (`prompt_attention.cu:1`; its fold at `:37-47` and its tile body at `:154-197` repeat
`attention.cu:130-140` and `:232-267`), with the differences that matter for a non-finite value listed under NaN and
infinity. Not in scope: the preparation before this step (q and k clips, the log-tau scaling of `q` and `rel` in global
layers, `attention.cu:46-89`), the output projection, and any change to a kernel: this page changes none.

The kernels keep a ring cache (a key at position `p` sits in slot `p % ring`, `attention.cu:87`, `:207`;
`kernels.zig:41`, `:54-56` round the ring to a multiple of 64); the host model takes a dense cache, one place per
position, which gives the same values as a ring holding every position. Its shapes are the kernels' at `hd = 128`,
`d_rel = 16` and 1024 bias columns (`kernels.zig:5-7`) and, for tests, any smaller `hd` (a head is zero-padded to a
multiple of 16: zero products add nothing), `d_rel <= 32` and any number of columns. The fp32 model in the kernels'
order has no limit on the number of keys; the float64 model in `attn.zig` takes at most `most_keys = 8192`
(`attn.zig:8`).

## Values and where they round

Notation: `bf(x)` is round-to-nearest-even to bf16; every input (`q`, `k`, `v`, `rel`, `proj`, `sink`, `gate`) is a
bf16 value held exactly in fp32 (`attention.cu:25`). The product of two bf16 values is exact in fp32 (a 16-bit
significand) unless it overflows or lies below `2^-134` (a sufficient condition: a 16-bit product from `2^-134` up
ends on the fp32 grid `2^-149`; below it the product may be inexact), so only additions and the operations listed
round. For row at position `t`, key `s` and `dist = t - s`:

| Step | Value | Where | Rounded to bf16 |
|---|---|---|---|
| bias table | `tab[d] = bf(sum_j rel[j] * proj[j][d])`, `j` in order | `attention.cu:116-120` | yes, once |
| dot | `D = sum_i q[i] * k[i]`, fp32 accumulator | `attention.cu:222-231` | no |
| score | `X = (D * scale + bf(tab[dist])) * kLog2e`, `scale = 1 / sqrtf(128)` in fp32; bias 0 where `dist >= 1024` | `attention.cu:14`, `:239-240`; `kernels.zig:180` | no |
| mask | `X = -inf` for a key that is not live: `t - s >= window`, `s > t`, or outside the step | `attention.cu:237`, `:240` | n/a |
| P | `P = bf(exp2f(X - mu))`, `mu` the running maximum of the step | `attention.cu:258` | yes, once |
| denominator | `l += P` over the keys of a lane, in fp32 | `attention.cu:259` | no |
| values | `acc += P * v`, fp32 accumulator | `attention.cu:268-279` | no |
| sink | `sk = bf(sink) * kLog2e`, its term `exp2f(sk - mf)` | `attention.cu:377-380` | no |
| output | `y = o * a / (ll * a + exp2f(sk - mf))`, `out = bf(y * (1 / (1 + expf(-gate))))` | `attention.cu:380-382` | yes, once |

The kernels round to bf16 at exactly three points: the bias table, P and the output. The score (the dot, the scale, the
bias add and the log2e product), every denominator and every sum of values stay fp32. P is rounded once and that one
value serves the products and the denominator (`attention.cu:259` adds `bf(p)`). The sink is never rounded after its
input and enters the denominator only at the merge. The scores are in log2 units; `exp2f` of a difference of two
log2-unit values is `e^x` of the natural-log difference.

## Order of every sum

1. Bias table: the taps `j = 0 .. 15` add into one fp32 accumulator in order, then one rounding
   (`attention.cu:118-120`).
2. Score: 8 chained `k16` mma, `k = 0 .. 7`, from a zero accumulator, fp32 accumulation
   (`attention.cu:224-230`; the prompt kernel adds the same 8 in the same order, `prompt_attention.cu:147-152`). Then
   `(sc * scale + bias) * kLog2e` (`attention.cu:240`).
3. A step is a run of 64-key tiles (aligned to multiples of 64, `attention.cu:205`) from a fresh start: `m = -inf`,
   `l = 0`, `acc = 0` (`attention.cu:201-203`). A step is a 256-key chunk on a sliding layer and a 1024-key sub-chunk on
   a global layer (`kernels.zig:12-15`, `attention.cu:127`, `:198`). Per tile:
   - the maximum `mx` over the tile's live keys (order does not matter), `mn = fmaxf(m, mx)`, `mu = mn == -inf ? 0 :
     mn`, `alpha = exp2f(m - mu)`, `m = mn`, `l *= alpha` (`attention.cu:232-252`);
   - `P = bf(exp2f(sc - mu))` and `l += P`, a lane `t = lane & 3` holding the keys `8j + 2t + {0, 1}` of the tile for
     `j = 0 .. 7` in that order (`attention.cu:255-261`); then `acc *= alpha` (`attention.cu:262-267`);
   - the values: for `u = 0 .. 3` one mma over 16 keys onto `acc`, so four chained mma a tile in key order
     (`attention.cu:268-279`).
4. End of a step: the four lanes' `l` add as `(l0 + l1) + (l2 + l3)` (`attention.cu:281-284`, two xor shuffles; every
   lane ends with the same bits since addition commutes).
5. Global layers: the sub-chunks of a chunk of 16384 keys fold in order into one `(M, L, O)`, a row folding the
   sub-chunks up to its own position (`attention.cu:285-295`, `:354-365`; `prompt_attention.cu:214-228`). A fold:
   `mn = fmaxf(M, m)`, `a = exp2f(M - mn)`, `b = exp2f(m - mn)`, `L = __fadd_rn(__fmul_rn(L, a), __fmul_rn(l, b))` and
   the same for each output index (`attention.cu:130-140`; `prompt_attention.cu:37-47`).
6. Merge: the chunks of the row, `c = lo / chunk .. pos / chunk`, fold in order into `(mm, ll, o)` with
   `mn = fmaxf(mm, m)`, `a = exp2f(mm - mn)`, `b = exp2f(m - mn)`, `ll = ll * a + l * b`, `o = o * a + po * b`
   (`attention.cu:351-376`). A sliding layer's chunk is one step; a global chunk is its folded sub-chunks.
7. Sink and output: `sk = bf(sink) * kLog2e`, `mf = fmaxf(mm, sk)`, `a = exp2f(mm - mf)`,
   `y = o * a / (ll * a + exp2f(sk - mf))`, then `out = bf(y * (1 / (1 + expf(-gate))))` (`attention.cu:377-382`).

A step starts from its own maximum, so a weight rounds to bf16 against its step's running maximum, not against the
maximum of earlier steps; the maximum of earlier steps meets it at the fold and merge, by `exp2f` and a product, which
do not round to bf16.

What the order does not fix:
- The order of the 16 products and the accumulator inside one mma, and its rounding, are the hardware's and are not
  written down in the repository. The products are exact (bf16 times bf16), so only the additions round. The host model
  adds them in ascending order, each rounded to nearest and unfused: that is the model's choice.
- Where the source writes a plain `*` and `+`, nvcc may contract them into one fused operation: `attention.cu:119`
  (`acc += r[j] * bf(...)`), `:240` (`sc * scale + bias`), `:252` with `:259` (`l *= alpha`, then `l += P`),
  `:373-374` (`ll * a + l * b`, `o * a + po * b`), `:380` (`ll * a + exp2f(...)`), `:258` (the `* kLog2e` of `:240`
  with the `- mu` in `exp2f(sc - mu)`), and the same lines of `prompt_attention.cu` (`:160`, `:172`, `:185` with
  `:193`, `:192`); the list names the sites found and excludes no other. The kernels build with `-O3` and no other
  floating-point flag (`zig/build/cuda_kernels.zig:161-162` at that commit, passed to nvcc at
  `zig/build/cuda_fatbin.zig:24`), and NVIDIA's nvcc documentation (v13.5, options `--fmad`, `--prec-div`, `--ftz`)
  gives the default of `--fmad` as "This option is set to true and nvcc enables the contraction of floating-point
  multiplies and adds/subtracts into floating-point multiply-add operations". We state what the source and the build
  line say and nothing about the code nvcc generated. The folds at `attention.cu:134`, `:139` use `__fmul_rn` and
  `__fadd_rn` and do not contract. The host model rounds the product and the sum separately at every open site; the
  bound below counts roundings and holds for either choice, a fused operation having fewer.
- Two assumptions the bound rests on and no host test can check: that the tensor-core adders do not flush subnormal
  numbers to zero (a flush would be an absolute error of up to `2^-126` an operand, which the bound does not count),
  and that `exp2f` and `expf` on the device keep to the 2 ULP of the hypotheses below; the host model uses Zig's `@exp2`
  and `@exp`, so no host test can push that 2 ULP.
- Several streams: the host model takes a call as one stream, its blocks starting at the call's first row; the kernel
  forms blocks per segment (`kernels.zig:59-70`, `attention.cu:150-151`). For one stream they are the same.
- Division and subnormal numbers: the nvcc documentation says of `--prec-div` "This option is set to true and nvcc
  enables the IEEE round-to-nearest mode." and of `--ftz` "This option is set to false and nvcc preserves denormal
  values."; the build line sets neither.

## NaN and infinity

What follows is what the order above gives, taking IEEE propagation in the mma and the shuffles as given; it is a
consequence of the source and the host model, not a measurement on a device.
- A key that is not live scores `-inf` after the mma (`attention.cu:240`), so a NaN or infinite `k` at a masked key
  changes nothing; its weight is `exp2f(-inf - mu) = 0`.
- `fmaxf` passes a NaN by, so a NaN score does not stop the maximum, but `exp2f(NaN)` is NaN, `P` is NaN, and so are
  `l` and every output index of that row and head.
- A value `v` that is not finite at a live key keeps its output index non-finite where its weight is positive
  (`inf * P`); where the weight is 0, the product is `0 * inf = NaN`.
- Masked keys that a block loads: a block loads the keys `max(step start, tile_lo) .. min(step end, tile_hi)`, where
  `tile_lo` and `tile_hi` come from the first and last of its 64 (row, head) pairs, and zero-fills the rest of a
  64-key tile (`attention.cu:155-157`, `:204`, `:211-216`). A key in that range that is masked for one row (`kp > pos`
  or `kp < lo`, `attention.cu:237`) meets that row's `P = 0` in the mma, so a non-finite `v` there is `0 * inf = NaN`
  in the row's output index of that column (the columns of `v` where it is not finite), whatever the row's own keys
  hold, and the NaN stays through the rescales, folds and the merge. The other columns of the row stay finite. Which
  rows are affected depends on the call: rows of one block at different positions see each other's later keys, and in
  a sliding layer a row sees the keys between the block's first window start and its own. A call of a single row has
  no such key (its range is its live keys). The host model reproduces this rule for the decode kernel (`block_pairs =
  64` in the model, `tile_pairs` at `kernels.zig:10`). The prompt kernel loads `b_lo .. min(b_hi, tile_hi)` and skips
  a tile past a warp's last row (`prompt_attention.cu:84`, `:60-61`, `:143`), so its set differs; the host model does
  not reproduce it.
- A NaN sink gives a NaN denominator. A `+inf` sink makes `mf` infinite, `a = exp2f(mm - inf) = 0` and
  `exp2f(inf - inf)` NaN: every output of that row and head is NaN. A `-inf` sink contributes 0.
- An infinite gate saturates the sigmoid: `+inf` gives 1, `-inf` gives 0 (so `y * 0` is 0, or NaN if `y` is
  infinite); a NaN gate gives NaN.

## FP64 check and its bound

The float64 model (`attend` with `f64` in `attn.zig`, at most `most_keys = 8192` keys) evaluates the formula of the
table above without any rounding: with `x = (q . k) hd^-1/2 + bias` in natural-log units, `y = sum_s e^(x_s - M) v_s /
(sum_s e^(x_s - M) + e^(sink - M))` and `out = y * sigmoid(gate)`; the bias is the exact `sum_j rel[j] *
proj[j][dist]` where `dist < rext`. The bound's own float64 value is `head` in `attn_bound.zig`: the same formula in
log2 units, run a second time beside `attend`, without `attend`'s limit of 8192 keys. The oracle evaluates the same
formula with Python's standard library (and `transformers`, see the fidelity check). The fp32 model in the kernels'
order (`attendOrder` in `attn_order.zig`) follows the order above, rounds the bias table, P and the output to bf16 and
nothing else, and is held to float64 by the bound below.

Domain. The bound is claimed for a row and head whose inputs satisfy these conditions, checked in float64 from the
inputs (`head` in `attn_bound.zig` returns `error.AttnDomain` outside them): every `q`, `rel`, `sink` and `gate` of
the pair, every `proj` entry it reads and every `k` and `v` of the keys its block loads is finite; `gate >= -87` for
every output index; `A_s < 2^120` and `B_s < 2^120` (below) for every live key and `|sink| < 2^120`; `nk <= 2^20` and
`nk * vmax < 2^120`. They are sufficient and not sharp, and outside them the page claims nothing. Inside them `exp(L)`
can still overflow in float64: it does once `L` is above about 709.78, and `L` grows with the size of the scores, so
well inside the domain; there the bound is infinite and claims nothing, and `limit` returns `+inf` rather than `0 *
inf`. The test `bound: where exp(L) overflows the bound is infinite, never NaN` has `q = k = 2^50` and `v = 0`: it
prints `L 6.20e24`, the model gives 0 and the bound is infinite. The need is real: at `gate = -89` the kernel's
`expf(89)` is `inf` (`e^89` is about `4.5e38`, above the largest fp32 number), the sigmoid `1 / (1 + inf)` is 0 and
the output 0, where float64 has `1 * 2.2e-39`; and at `-88` the sigmoid `2^-126.96` is already subnormal and its
quotient has more than `u` of error (`e^87` is `2^125.5`, `e^88` is `2^126.96`, `e^88.5` is `2^127.68`).

For one row and head with `nk` live keys, let `u = 2^-24`, `u_mma = 2^-23`, `u_bf = 2^-8`, `gamma(n) = n u / (1 - n u)`
and `lambda(a) = -ln(1 - a)`. The fp32 result `y` before the output's rounding satisfies, against the float64 result
`want`,

`|y - want| <= (1 + 2^-20) * (mag * (exp(L) - 1) + sg * (nk + 1) * (2 vmax + 1) * 2^-130 * exp(L)) + 2^-149`

`mag = sg * sum_s (P_s |v_s|)` with `P_s` the float64 softmax weights (the sink in the denominator), `sg` the sigmoid of
the gate and `vmax` the largest `|v|` of the row's keys. The terms of `L`, with `T` the tiles of a step (16 global, 4
sliding), `S` the sub-chunks of a chunk (16 global, 0 sliding), `C` the chunks of the row (global: `floor(t / 16384) +
1`; sliding: `floor((window - 1) / 256) + 2`, the most a window spans), `M = ceil(hd / 16)` and `E = (T - 1) + S + C +
1`:

`L = L_num + L_den + lambda(gamma(5)) + 4 lambda(u)`

`L_num = eta + lambda(gamma(5)) + lambda(u_bf) + E lambda(gamma(6)) + 4 T lambda(17 u_mma) + (S + C) lambda(u)`

`L_den = eta + lambda(gamma(5)) + lambda(u_bf) + E lambda(gamma(6)) + (16 T + 2 + S + C + 1) lambda(u)`

`eta = ln 2 * (d + u ((Xmax - Xmin) + 2 d))`, `d` the largest absolute error of a score in log2 units, over the keys and
the sink, and `Xmax`, `Xmin` the largest and smallest log2-unit score (`log2(e)` times `x`) over the keys and the sink;

`dX_s = log2(e) * (hd^-1/2 * A_s * G_dot + B_s * G_bias) + 2^-133`, `G_dot = (1 + 17 u_mma)^M (1 + gamma(6)) - 1`,
`G_bias = (1 + gamma(d_rel)) (1 + u_bf) (1 + gamma(3)) - 1`, `A_s = sum_i |q_i k_i|`, `B_s = sum_j |rel_j proj_j|` (0
where the key gets no bias); for the sink `gamma(2) |log2(e) sink| + 2^-133`. The same `L` with the same counts is
what `attn_bound.zig` evaluates. Each term is a count of roundings in the order above:

- The hypotheses. `exp2f` and `expf` are within 2 ULP of the correctly rounded result: NVIDIA's CUDA Programming Guide
  (v13.5, section 5.6.7.2, Exponential Functions, Table 54) lists `expf(x)` and `exp2f(x)` with "2 ULP" for `float`.
  Section 5.6.7 says "The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs
  between the value returned by the function and a correctly rounded result of the corresponding precision obtained
  according to the round-to-nearest ties-to-even rounding mode. The error bounds are derived from extensive, though
  not exhaustive, testing. Therefore, they are not guaranteed." Within 2 ULP of the correctly rounded value, which
  itself is within `u`, the result is within `gamma(5)` of the exact one. The second hypothesis: an mma's result
  differs from the exact sum of its 17 operands (16 products and the accumulator) by at most `17 u_mma` times the sum
  of their absolute values. It is the assumption of `zig/tests/cuda/contract_inkling2_attn_ref.zig`: `u_mma = 2^-23`
  "a tensor-core add: its adders may truncate" (`:24`), the dot's error `abs * scale * 136 * u_mma` (`:158`, 136 is 8
  chained mma times 17) and `tiles * (4 * 17 + 1)` in `depth` (`:85`). It is not a guarantee of the hardware, and it
  covers every order inside an mma. The other fp32 operations round to nearest (above), and a fused operation has
  fewer roundings than the unfused pair, so every count holds for the open sites.
- `G_dot`: `M` chained mma (the dot's error against `A_s` is `(1 + 17 u_mma)^M - 1` by induction over the chain), and
  six roundings of the score: the scale constant (`1 / sqrtf(128)` is two roundings of `hd^-1/2`), the product with it,
  the add of the bias, the constant `kLog2e` (one rounding of `log2(e)`) and its product.
- `G_bias`: the fp32 tap sum (`d_rel` adds), the bf16 rounding of the table (`u_bf`), and three roundings after it
  (the add, `kLog2e`, the product). The table's bf16 rounding is a term of its own and the largest of the score.
- `eta`: the score's error `d` and the roundings of the differences `X - mu`, of every `exp2f` argument of a rescale
  and of `sk - mf`. Each is at most `u` of its difference; all differences of a weight are `<= 0` and add up to
  `Xmax' - X_s`, so a weight's exponent error is at most `u (Xmax - Xmin + 2 d)`. In natural units, a factor of `ln 2`.
- `lambda(gamma(5))`, `lambda(u_bf)`: the `exp2f` of P, and P's bf16 rounding (`attention.cu:258`): the main term of
  `L`. It acts on the numerator and, because `l` sums the same rounded P (`attention.cu:259`), on the denominator.
- `E lambda(gamma(6))`: a weight meets at most `T - 1` rescales of later tiles, `S` folds of a sub-chunk, `C` merges of
  a chunk and the sink's `a`, each an `exp2f` (`gamma(5)`) and a product (`u`).
- `4 T lambda(17 u_mma)`: the values' mma chain, four mma a tile over `T` tiles. `(S + C) lambda(u)`: the add of each
  fold and merge.
- `16 T + 2 + S + C + 1`: the denominator's adds: 16 a tile in a lane, two combining the lanes, one for each fold or
  merge (`S + C`) and the add of the sink's term.
- `lambda(gamma(5)) + 4 lambda(u)`: the sigmoid's `expf`, its add and its quotient, the quotient `o a / den` and the
  final product.
- Domain: under its conditions no fp32 operation of the order overflows. A dot, its scaled and biased value and the
  product with `log2(e)` stay below `2^123`; every `exp2f` has an argument `<= 0` and a result `<= 1 + gamma(5)`; a
  sum of `P v` stays below `2^121`; the denominator is at least `exp(-L)` (the largest weight is 1), so `y` stays
  below `2^122`; and `expf(-gate) <= e^87 < 2^126`, so `1 + expf(-gate)` stays below `2^126` and the sigmoid is a
  normal number.
- The last bf16 rounding of the output is not in `L`: the bound on the bf16 result is the fp32 bound `b` plus
  `u_bf (|want| + b) + 2^-134`.
- Subnormals: below `2^-126` an operation is within `2^-150` of its exact result, `exp2f` within `2^-148`, and an add
  is exact; bf16's half step there is `2^-134`. A weight is off by at most `2^-133` in absolute terms, the largest
  weight is `exp2f(0) = 1` so the denominator is at least `exp(-L)`, and `2^-130 (nk + 1) (2 vmax + 1)` covers the
  numerator, the denominator and the absolute errors of the last operations. In `dX_s` the bias table rounds to bf16,
  and where its sum is below `2^-126` that error is absolute, up to `2^-134` (half a bf16 step there), not `u_bf B_s`;
  times `log2(e)` it is below `2^-133.4`, as is the sink's product with `kLog2e`, and the other subnormal contributions
  (products below the fp32 grid, each at most `2^-150`) lie below `2^-140`; so `2^-133` covers the score's. The last
  product `y * sigmoid` rounds absolutely, by up to `2^-150`, where its result is below `2^-126`; the term `2^-149`
  after the factor `1 + 2^-20` covers that, which `sg` scales away in the other floor term.
- The factors multiply: `lambda(a)` bounds `|ln(1 + d)|` for `|d| <= a`, so `L` bounds the logarithm of each term's
  relative error and `exp(L) - 1` the error of the ratio `N / D`.
- `1 + 2^-20` covers float64, in `want` and in evaluating the bound: its sums of at most `2^20` terms round at `2^-53`
  each, below `2^-28` of `mag`, while `exp(L) - 1 >= 2^-8`.

The bound is derived from these counts and the two hypotheses, not fitted to runs. It makes no claim about bit
equality with another implementation.

The host tests evaluate the bound (`attn_bound_test.zig`, `attn_bound.zig`): the fp32 model in the kernels' order
against the float64 formula, before and after the output's rounding, on the oracle's fixture; on rows across the seams
(64, 1024 and 16384 keys, global, with the maximum rising across each; a sliding window with chunks at 256; three rows
of a block; the checkpoint's `hd = 128`, 16 taps, 1024 columns); and on a case for each term: cancellation in the
values, a large gap between scores, P nearest a bf16 half step, a dominant sink, P below the normal range, an output
below the normal range (where the last product's absolute error exceeds the sigmoid-scaled floor term), and the ends
of the domain. Every element is inside its bound, and over these cases the worst error over bound is 0.16 before and
0.26 after the output's rounding; the tests print each case. A further case aims at the table's rounding: one key's
table sum rounds up (the weak key) and the dominant key's rounds down (`bound: a table rounded up at the weak key and
down at the dominant one`). It prints 9.640e-1 before and 9.547e-1 after the output's rounding, so the bound is close
to tight for that rounding and the 0.16 and 0.26 are no margin.

Each rounding point and each seam also has a test whose expected value is written by hand in float64 and does not come
from the model (`attn_order_hand_test.zig`): the table's tie to even, P rounded once in the products and in the
denominator, the output after the gate, the dot and scale in fp32, the sink at the merge unrounded, the window, the
bias distance, the head group, and, for the seams, sums that only the stated grouping gets right: `2^24`, ones and
`-2^24` put in different tiles, sub-chunks, sliding chunks and global chunks give 0 or 2 depending on whether the ones
meet before or after the big values.

## Fidelity against Transformers eager (reported)

`transformers` 5.23.0 runs this attention at eager points that are not the kernels'. In the eager path (TF523:368-399
with the relative logits of TF523:165-188) the bf16 tensors round after the dot (TF523:383), the scale (TF523:383,
:414), the bias table (TF523:184), the biased logit (TF523:385), the normalised weights (TF523:391-392), the value sum
(TF523:396) and the gated output (TF523:496, :628): seven roundings, against the kernels' three, and the softmax
divides before P is rounded. The kernels do not follow that arithmetic and this page does not ask them to; the
comparison is a reported number, not a gate. The oracle (`zig/src/families/inkling2/oracle/attn.py`) runs
`Inkling2RelativeLogits` and `eager_attention_forward` on the same bf16 inputs as the fixture, builds the causal and
sliding mask (`masking_utils.py` of the same release, :76, :95 and :630, as TF523:1290-1291 builds it) and the gate
from the source, and the fixture keeps both its float64 result and its bf16 bits.

- The eager-point fp32 model (`attend` with `f32` and `rounded`, in `attn.zig`, the seven roundings above with a fixed
  fp32 order of its own) gives the bf16 bits of the `transformers` path on the fixture: 4800 of 4800 values equal,
  none a step apart. A libm with another last bit in `exp` may flip a rounding; the test allows one value in a
  thousand one step apart.
- The fp32 model in the kernels' order (`attendOrder`) against the same bits: 2118 of 4800 equal, 1689 one step apart
  and 993 further; the largest difference, as `|got - transformers| / (|transformers| + top)` with `top` the head's
  largest `|transformers|`, is 6.1e-2. This is the arithmetic difference between the kernels' three roundings and the
  eager path's seven, on the six fixture cases (tiny-random `inkling-2`, the checkpoint's shapes among them).
- The float64 model equals the oracle's float64 values within `(|y| + top) 2^-40` per element, `top` the head's
  largest `|y|`.
- The fixture `zig/src/families/inkling2/fixtures/attn_tiny.bin` (SHA-256
  `96aadd48918923fe5a4d165740bc337340476aeeacc5c3ee63600d8dcd7d0cde`) comes from `python3
  zig/src/families/inkling2/oracle/attn.py write <model_dir>`, run on a local copy of
  `tiny-random/inkling-2@0a57c72e4e5b68aafe1d983dbb62980344a5fcd7` (`model.safetensors` SHA-256
  `ac7f0eaa84e50b53c4afce3ea7eecdc49ccca21bd467edb16f3d052d9099a3de`) with `transformers` 5.23.0 and `torch`
  2.15.0+cpu (`attn.py:13-16`).

## Falsification

Each single change below was made alone to the host code (the fp32 model in `attn_order.zig`, the window to `first` in
`attn.zig`, and the bound's terms and domain checks in `attn_bound.zig`), `zig build test-inkling2-host` was run on
it, and the change was reverted. The test that turns red is named (test names as in `attn_order_hand_test.zig` and
`attn_bound_test.zig`; where more than one goes red, the count is given). The unchanged tree is green: 81 tests passed
and 4 skipped.

| Single change to the host model | A test that turns red (N red in all) |
|---|---|
| table: no bf16 rounding | `table: bf16 once, a tie to even` (2 red) |
| dot rounded to bf16 (as eager) | `score: the dot and the scale stay fp32` (2 red) |
| scaled dot rounded to bf16 | `score: the dot and the scale stay fp32` (2 red) |
| logit rounded to bf16 | `sub-chunk seam: starts from its own maximum` (5 red) |
| scale missing | `score: the dot and the scale stay fp32` (6 red) |
| scale 1/hd | `score: the dot and the scale stay fp32` (5 red) |
| bias not added | `bias: distances under rext only` (8 red) |
| P not rounded (numerator and denominator) | `P: bf16 once, in the products and the denominator` (7 red) |
| P rounded for P.V, denominator sums unrounded P | `P: bf16 once, in the products and the denominator` (8 red) |
| P.V sums unrounded P, denominator rounded | `P: bf16 once, in the products and the denominator` (7 red) |
| output not rounded | `output: bf16 once, after the gate` |
| output rounded twice (before the gate too) | `P: bf16 once, in the products and the denominator` (10 red) |
| value sum rounded to bf16 (as eager) | `bias: distances under rext only` (4 red) |
| sink term rounded to bf16 | `sink: at the merge, unrounded` (2 red) |
| sink at half its weight | `sink: at the merge, unrounded` (3 red) |
| sink left out | `sink: at the merge, unrounded` (4 red) |
| maximum at the merge without the sink | `NaN: live keys and the sink` |
| tile of 32 keys | `tile: a weak key rounds against a strong one after it` |
| tile of 128 keys | `tile seam: P rounds against its tile's maximum, then rescales` |
| sub-chunk of 512 keys | `sub-chunk: summed alone, then folded` |
| sub-chunk of 2048 keys | `sub-chunk seam: starts from its own maximum`; `sub-chunk: summed alone, then folded` (3 red) |
| sliding chunk of 128 keys | `sliding chunk: summed alone, then merged` |
| sliding chunk of 512 keys | `chunk seam at 256: sliding starts afresh, global does not`; `sliding chunk: summed alone, then merged` (3 red) |
| global chunk of 8192 keys | `global chunk: its sub-chunks fold first` |
| global chunk of 32768 keys | `global chunk: its sub-chunks fold first`; `global chunks: merge in order` (2 red) |
| no rescale between tiles (alpha 1) | `tile seam: P rounds against its tile's maximum, then rescales` (9 red) |
| denominator not rescaled between tiles | `tile seam: P rounds against its tile's maximum, then rescales` (9 red) |
| values not rescaled between tiles | `tile seam: P rounds against its tile's maximum, then rescales` (5 red) |
| fold scales swapped (old by the new part's factor) | `tile seam: P rounds against its tile's maximum, then rescales` (36 red) |
| chunks merged in reverse order | `global chunks: merge in order`; `sliding chunks: merge in order` (2 red) |
| masked key scored 0 instead of -inf | `` window: exactly `window` keys `` (29 red) |
| block loads only the row's own keys (NaN rule) | `NaN: a non-finite V at a masked key a block loads` |
| block starts at the row's own window (NaN rule) | `NaN: below the window, a key another row's block loads` |
| mu without the all-masked guard | `bound: sliding seams` |
| gate sigmoid with the wrong sign | `NaN: a masked key no block loads is harmless`; `NaN: a non-finite V at a masked key a block loads` (34 red) |
| gate missing | `gate: its sigmoid` (11 red) |
| key limit of 8192 kept in the fp32 model in the kernels' order | `shapes: over 8192 keys is accepted, a refusal stays a refusal` (4 red) |
| block of one pair (NaN rule) | `NaN: a non-finite V at a masked key a block loads`; `NaN: below the window, a key another row's block loads` (2 red) |
| window one key too wide | `` window: exactly `window` keys `` (6 red) |
| bias one distance too far | (the table read runs past the last column and `index out of bounds` aborts the test binary: 32 tests terminate, none names a rounding) |
| bias one distance too short | `bias: distances under rext only` (3 red) |
| wrong key-head group (h mod kv_heads) | `group: the group's key head` (3 red) |
| table taps summed in reverse | `table: taps in order` |
| dot summed in reverse (all dims, one chain) | `score: one chain across k16 blocks` |
| bound: no absolute term 2^-149 for the last product | `bound: an output below the normal range` |
| domain: gate bound not checked | `bound: the domain ends where fp32 overflows or the sigmoid leaves the normal range` |
| domain: dot magnitude not checked | `bound: the domain ends where fp32 overflows or the sigmoid leaves the normal range` |
| domain: live keys not limited to 2^20 | `bound: the domain ends at 2^20 live keys` |
| bound: not infinite where exp(L) overflows | `bound: where exp(L) overflows the bound is infinite, never NaN` |
| domain: infinite value at a loaded key not checked | `bound: the domain ends where fp32 overflows or the sigmoid leaves the normal range` |

These changes leave the host tests green. They are the model's choices or equivalent forms, and their effect is one
rounding, below what a float64 hand value or the bound can separate:
- fusing the score `dot * scale + bias` into one `fma` (an open site above), or the fold's product and add (the source
  pins the fold with `__fmul_rn` and `__fadd_rn`, so the model's separate roundings follow the source, not a test);
- the order of the products a tile's `P.V` adds: the model adds them in one fp32 chain, so descending key order over
  the 64 keys reverses the four chained mma and the products inside each; no test pins it (the order inside an mma is
  the hardware's, the order of the four is stated above); and the dot's k16 blocks summed apart and then added in
  order;
- which of the four lanes holds a key of `l` and how the four lane sums add (one lane, or a chain instead of
  `(l0 + l1) + (l2 + l3)`): one rounding in `l`;
- the sink folded in before the chunks instead of at the merge: the same operations in another association;
- the exponent in natural-log units with `exp` instead of log2 units with `exp2f`: the same values except where the
  other fp32 rounding of the score `X` moves a P across a bf16 rounding boundary; the tests stay green with it, so
  none of them pins it;
- the bound's score floor at `2^-140` instead of `2^-133`: the error it covers is below `2^-133`, while `L` is at least
  `lambda(u_bf)`, about `2^-8`, so no input can tell the two apart; the value rests on the derivation above.

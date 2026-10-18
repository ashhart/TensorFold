# Flash Next rotary probe allowances

This is a conditional numerical-contract change, not a kernel change or a passed device gate.
The normalizer, phases and final reference remain independently evaluated in FP64.
The allowance compares the device output with that reference after its FP32-to-BF16 rounding.
The original nonfinite, repeatability, guards, row-count invariance and resource checks remain required.

## Scalar accuracy required before qualification

Let u = 2^-24. The candidate requires relative rsqrt error at most 4u, relative exp2 error at most 8u,
and absolute sine/cosine error at most 8u at the actual FP32 argument.
These are explicit scalar accuracy requirements, not guarantees about Metal fast functions.
For finite normal results, two and four FP32 ulps are no larger than 4u and 8u relative, respectively;
for sine/cosine, four ulps over [-1, 1] are no larger than 8u absolute.
They are chosen independently of any observed output-to-bound ratio.
Device component traces must establish these caps before the candidate can qualify.
If they fail, do not increase them to fit the final output.

The prepared fixture values are multiples of 1/8 in [-1, 1]; their four-row means are multiples of 1/32.
The allowance rejects other input grids, weights outside [0.5, 2] in magnitude, nonnormal/nonpositive epsilon,
and positions outside [0, 1048576), instead of claiming an unchecked universal input bound.
The numerical probe's separate nonfinite gate retains its original exact class checks.

## Normalization and intermediate BF16 values

For this fixture grid, every square is an integer multiple of 1/1024, at most one.
At most 256 such terms total at most 262144 units, requiring only 19 significant bits.
All FP32 products and additions are therefore exact in any order.
This proof does not assume a particular SIMD reduction tree or weaken the accumulator declaration.
The registered generic reduction depth remains unchanged.

Let S be that exact sum and q = S/n + epsilon.
Division by n is a power-of-two scale, so only the epsilon addition contributes to the denominator radius:

    Eq = u |q| + guard(q).

Enclose 1/sqrt(q_device) by evaluating q +/- Eq, then apply the rsqrt accuracy cap.
Multiply each signed input and weight by both inverse endpoints.
Expand them for two FP32 products with gamma(2, u), then round both endpoints through FP32 and BF16.
Monotone rounding encloses midpoint crossings instead of charging their error to a cancelled final result.
This gives the already rounded reference pair a, b and absolute radii Ea, Eb.
Unrotated dimensions need only their own BF16 interval radius.

Both pooling probes use an exact four-row FP32 mean in these fixtures.
The checker verifies every addition and the power-of-two scale is exact before its BF16 store.
An inexact pooling fixture is refused and needs its own propagated mean interval, not an implicit larger tolerance.
The subsequent normalization interval is identical to the preparation probes'.

## FP32 phase construction

The plain reference frequency is f = 2^x with x = -(i/32) log2base.
The device exponent x32 is the FP32 rounding of that exact product; i/32 is exact.
Evaluate 2^x32 independently in FP64 and enclose the exp2 result with its required relative cap.
For YaRN, both sides read the same stored FP32 frequency, so this interval has zero frequency radius.
Supported integer positions are exactly representable in FP32.
Round the position-times-frequency interval endpoints to FP32 and compare them with the ideal angle t = p f.
Let h be the largest resulting absolute angle discrepancy.

Sine and cosine are globally 1-Lipschitz and bounded by one.
Taylor's remainder also gives a local enclosure:

    Dcos = min(2, h, |sin(t)| h + h^2/2)
    Dsin = min(2, h, |cos(t)| h + h^2/2).

Add the absolute trig accuracy cap, multiply by the stored YaRN scale and propagate that FP32 multiply.
This yields coefficient errors EC and ES for C = scale cos(t), S = scale sin(t).
Phase error is weighted by the unrotated inputs; it cannot disappear when a rotated term approaches zero.

## Rotation and the already rounded reference

For each relabelled pair, let r = a C +/- b S.
A pre-output error allowance is

    E = |C| Ea + |S| Eb
      + (|a| + Ea) EC + (|b| + Eb) ES
      + gamma(2,u) [(|a| + Ea)(|C| + EC) + (|b| + Eb)(|S| + ES)].

The depth-two term covers separate products and addition; contraction needs no larger allowance.
The final absolute comparison to w = BF16(FP32(r)) uses

    E + 2^-8 (|r| + E) + |r - w| + guard(r).

The reference's own rounding is explicit. A relative term based only on |w| would miss cancellation.
The implementation adds conservative FP64 evaluation guards and a smallest-normal FP32 floor for underflow/flush.
These guards are independent of measured errors.

The generic kernel bound is unchanged. Only these ten numerical probes provide an absolute-allowance callback;
the mask keeps its exact check. Both the shape and accumulation gates consume the same callback and reject invalid allowances.
CPU witnesses and real callback tests establish the stated arithmetic and wiring only.
All eleven probes, component caps, compiler/resource receipts, model equality and long-context fidelity must be rerun before release.

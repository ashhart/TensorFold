# Shared M5 attention precision

`TF_ATTENTION_PV_FP32=1` selects explicit float FMA products for the shared `attention_nax` P x V on M5.
Unset or `0` keeps the existing tensor product, which truncates a float probability to ten fraction bits.
Other values are refused when a library using this attention source is opened.
Set the variable before starting the process and keep it fixed for that process.
The default has not changed. This is an experimental arithmetic mode, not a family qualification.

The full mode retains Q x K, softmax, masks, the normalizer and the final BF16 output.
It preserves every normal FP32 probability bit in its products, with FP32 FMA accumulation.
It does not promise exact real sums or bit equality with the cut mode.
The extra scalar products and lane shuffles can cost throughput; measure the selected checkpoint and prompt size.
Older GPUs keep the previous simdgroup-matrix source even when the switch is set.
Nemotron disk prompt states include the mode in their identity and cannot cross it.

The shared helper serves Nemotron prompt attention, including its paged twin, and Qwen Image/MOSS Image attention.
It does not change the 27B text path's explicit half probabilities, general float GEMM or another family's attention.
The 27B's Qwen-VL tower also uses a separate float GEMM.
A teacher-forced 27B comparison is an unaffected control, not proof of a quality change in that model.
Image denoisers have velocity outputs rather than top tokens; their teacher-forced velocities need their own case.

## Checks

```sh
zig build test-kernel-contract tf-kernel-contract -Dcpu=apple_m1
./zig-out/bin/tf-kernel-contract --only 'attention PV'
```

The contract registers cut operands against a cut-operand binary64 formula and full operands against full binary64.
Both require FP32 sums, guarded buffers, repeats, exceptional classes and matrix tile edges.
Its planted control runs the actual cut helper against the full-operand bound and must fail the shapes gate.
The bound is the FP32 chain length and term magnitude, not a percentage fitted to the observed result.
Separate launch records name both probes and all four production full-mode alignments.
M1/M2 compilation records and an M5 device receipt are distinct evidence; unsupported hardware is not a pass.

The template probe injects exact dyadic probabilities into the real attention source, including a float-output copy.
It checks causal/full visibility, query/key tails, key-tile seams, five repeats, input bytes and guards:

```sh
swiftc -O tools/zig/attention_pv.swift -o attention-pv
TF_ATTENTION_PV_FP32=0 ./attention-pv . check red
TF_ATTENTION_PV_FP32=1 ./attention-pv . check
TF_ATTENTION_PV_FP32=0 ./attention-pv . bench 2048
TF_ATTENTION_PV_FP32=1 ./attention-pv . bench 2048
```

Run device work only on an isolated GPU under the machine's existing queue and lock.
A compile failure cannot stand in for the cut mode's numerical red witness.
Teacher-force identical histories and report every top-token difference and both margins; do not judge diverging replies.
No default should change before the affected family has its own accepted correctness and fidelity receipts.

# Exact Qwen NVFP4 physical SK1 experiment

`TF_QWEN_NVFP4_SK1=1` opts Qwen ModelOpt CUDA into one CTA per QMMF output tile at
N=6144, K=5120 on sm_120 or sm_121. The default is off. Only Qwen's loader reads it;
other families keep their lane. Invalid values are refused by the environment registry.
The option refuses `TF_QMMF_HALF_BN=1` and `TF_Q35_GROUP_PROJ=1` so another experiment
cannot change the compared tile or bypass its launch.

This is a physical scheduling differential. The arithmetic remains SK2: each CTA
starts two FP32 partial sums from positive zero, visits each original half of K
in order, adds the second root to the first, multiplies the global scale once and
stores the original output type. Setting `Weight.sk=1` would change rounding and
is a planted control, not the candidate. No codes, scales or activations are repacked.

The candidate uses main's existing fused-slice body, newly instantiated for FP4
at row tiles16 and32, with its existing64-row instance beyond32 rows. Tile width64,
128 threads, four K64 stages, operand mapping, MMA instructions and logical SK2
arguments remain. The physical grid has one block per tile instead of a two-block
cluster. Main's original split-K rule remains unchanged.

Only ordinary decode/verify projections enter this branch. A prompt marked as such
with more than16 rows retains prompt16, including its existing tile selection and
rounding. Other N/K pairs, formats, pins and main's fused path from256 rows remain.
With the option unset or0 the candidate loads no functions and emits no launch.
The normal Qwen adapter then executes its existing path.

The source and host checks are an experiment, not GPU qualification. The new
instances' launch records remain pending until actual nvcc records are measured.
The byte contract is registered in `tf-cuda-test kernel-contract`; its focused
entry is `tf-cuda-test qmmf-sk1-contract`. It compares BF16 and FP32 outputs with
main on identical packed inputs at every verify width1..32, wider seams and prompt
rows through2048. Nonfinite bytes/classes, five repeats, guards and inputs are
checked. Controls change the logical split or the global scale and must fail.

On an authorized GB10, rebuild the qmmf fatbin with the qualified compiler and
embed it with the same complete fatbin set used for the server. Then run:

```sh
zigb test-qmmf-sk1-host
./zig-out/bin/tf-cuda-test qmmf-sk1-contract
./zig-out/bin/tf-cuda-test kernel-contract 'Qwen NVFP4 physical SK1'
./zig-out/bin/tf-cuda-test kernel-contract-limits "$FATBINS"
```

The focused contract does not replace the complete backend gate. Retain all
`SUMMARY` and `RECORD` lines, compiler version, SASS profile and binary hashes.
Missing compiled resource usage is a failed qualification check, not a pass.

For served A/B checks, use one GB10 and the identical locally available
`nvidia/Qwen3.8-27B-NVFP4` and `z-lab/Qwen3.8-27B-DFlash2` checkpoint revisions.
Run one server process at a time. Alternate off/on/on/off at one and four streams,
keeping the binary, prompt IDs, thinking, seed, KV precision, graph settings,
cache budget, clocks and request parameters fixed. Compare exact token hashes,
drafted/plain, concurrent/alone and resumed/fresh outputs before interpreting timing.
Keep cold prefill and warm decode separate. A kernel timing sum is not complete-round
latency, and CTA count is not a measured occupancy or wave count.

Stop on any output, logit or state difference. Do not retain a candidate with a
layer regression over2%, complete-step median gain below3%, warm/cold reversal or
concurrency regression. Reopen the projection ranking if its measured share is
below60% of complete GPU-active round time. RTX5090 results are unmeasured here;
GB10 measurements do not establish its170-SM scheduling behavior.

# EXL3 projection scheduling

`Pipelines.pairs_up_to_groups` selects the two-column SIMD-group product when
the corresponding sixteen-column grid has at most that many groups. Zero
disables the path. `TF_EXL3_PAIRS_UP_TO_GROUPS` overrides the threshold.
Both kernels retain the same result bits.

The earlier M5 Max measurements recorded beside this setting used one row
and the best of 20 runs. At K of 4096 to 5120, the recorded ratios were 2.15
for N=512, 1.34 for N=1024 and 1.23 for N=1280. N=2304 (144 groups) gave 0.73.
These historical tuning measurements do not qualify another GPU.

`encodeJobs` dispatches every job at one stage before inserting the stage
barrier. A concurrent encoder may overlap independent jobs; a serial encoder
runs them in order. Each job computes the same product as `encode` or
`encodeIndexed`. An output or scratch region must not overlap any other job's
read or write region. The entire group is checked before encoding starts.

The dense path retains both rotation stages and replaces the trellis product
with a dense product. It accumulates in f32 into the scratch sums consumed by
`rotate_out`, traversing K sequentially as the fp64 reference does.

The concurrent GPU test chains four rounds of two products. The plain product
writes the next round's input and the routed product writes a separate output.
The serial control uses `encode` and `encodeIndexed`. Missing stage or round
barriers, or overlapping scratch, can expose unfinished rows and change the
compared bits. All output and intermediate input comparisons are retained.

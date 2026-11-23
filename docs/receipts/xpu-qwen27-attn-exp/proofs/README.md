# Proofs: vector exp and fmax equal scalar on the card

Whether a vector `exp` or `fmax` gives the scalar one's bits depends on the compiler, so these are the versions the two
runs used, on 2026-11-22, one Arc Pro B70 (bmg-g31), Ubuntu 26.04.1, Linux 7.0.0-43-generic:

- Compiler: IGC 2.41.5 (`intel-igc-core-2` and `intel-igc-opencl-2` 2.41.5, `/usr/local/lib/libigc.so.2` and
  `libigdfcl.so.2`).
- Compute-runtime: 26.35.39758.21 (`intel-opencl-icd`, `libze-intel-gpu1`, `intel-ocloc`).
- The probes build their kernel through the OpenCL runtime (`clBuildProgram`, `-cl-std=CL3.0
  -cl-intel-256-GRF-per-thread`). The host also has Ubuntu's IGC 2.41.16
  (`libigc2`, `libigdfcl2`), which neither the runtime nor `ocloc` loads here: `LD_DEBUG=files` shows both resolve
  `libigc.so.2` and `libigdfcl.so.2` to `/usr/local/lib`, the 2.41.5 build.

| File | Script | What it checks |
|---|---|---|
| `expprobe_all.out` | `../scripts/expprobe_all.py` | all 2^32 float bit patterns, 16-lane vector `exp` against scalar `exp`: 0 mismatches |
| `fmaxprobe.out` | `../scripts/fmaxprobe.py` | 536,870,912 pairs (all special-value pairs, nearby and random patterns), vector `fmax` against scalar: 0 mismatches |

A different IGC can give a different answer: rerun both scripts after a driver or compiler change before relying on
the kernel's bits.

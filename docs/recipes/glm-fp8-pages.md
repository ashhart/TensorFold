# GLM CUDA paged FP8 compatibility

`TF_KV_PAGES=on` with `--kv-cache fp8` uses the existing E4M3 row quantization and attention arithmetic through
the stream's page table. Each layer's page part holds 512 rows of codes, their FP32 scales, then BF16 pooled keys.
The existing automatic FP8 selection keeps contiguous windows. FP8 with host-KV gather remains refused.

The change covers page geometry, constructors, write/read launch arguments, logical sparse token positions and
snapshot code/scale spans, including shared-prefix skipping. It retains the existing MTP page part.
It introduces no new setting or default.

## Reproduction

Use the Zig version in `.zig-version`. These checks use synthetic inputs and recording CUDA calls:

```sh
zig build test-glm-startup-host test-glm-fp8-kv-host test-glm-pages-host \
  test-glm-page-contract-host test-glm-snapshots-host test-glm-admission-host \
  compile-glm-native-open compile-glm-fp8-gate-host
```

These checks inspect startup refusal before construction, page geometry, constructors, launch ABIs, logical sparse
token units, shuffled snapshot scales, numerical-case registration and the three paged-FP8 resource records.

## Synthetic device reference

The existing CUDA runner's `glm-ref` command calls `glm_ref_fp8_pages.zig` through `glm_ref_fp8.zig`.
Build the runner with real CUDA images using the repository's normal CUDA build before running it on a free device.
`compile-glm-fp8-gate-host` compiles the same reference with inert images and performs no GPU execution.

The paged reference uses 513 keys, page table `[2, 0]`, a nonzero layer offset and an untouched physical-page guard.
It compares code and scale bytes with contiguous E4M3, then compares dense and sparse attention at both 16-head and
32-head dispatch widths. A sparse row with zero selected keys preserves its output.

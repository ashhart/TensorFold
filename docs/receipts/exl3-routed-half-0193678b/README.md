# Routed EXL3 half-product rotation

The source measured here is `0193678bfae7e7c4da02ec53784957048662b477`, on main
`077998f7d9b14adebec3a2acc2caa47ff7ae6ec7`. This is one shared operator, with no model loader or serving caller.
It rounds each input-times-scale product to FP16, uses the existing FP32 128-element Walsh transform,
and writes FP16. The existing wider-product EXL3 rotation is unchanged.

## Environment

Two independent NVIDIA GB10 devices, each in a 128 GB DGX Spark; Linux `7.0.0-1019-nvidia`,
NVIDIA driver `580.178.04`, nvcc `13.0.88`, Zig `0.17.0`.
The runner was cross-compiled on x86-64 Linux for `aarch64-linux-gnu`, CPU `baseline`.
No checkpoint, model server or drafter was loaded by the runner. Each GPU was idle before its run.

## Reproduce

From the source root with Zig 0.17.0 and nvcc 13.0.88 on PATH:

```sh
zig build -Dtarget=aarch64-linux-gnu -Dcpu=baseline tf-exl3-routed-half-contract --prefix out-arm64
zig build -Dfixed-cpu test-exl3-routed-half-host
mkdir -p operator-fatbins
set --
for sm in 80 86 89 120 121; do
  set -- "$@" -gencode "arch=compute_$sm,code=sm_$sm"
done
nvcc --fatbin -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true "$@" \
  zig/kernels/cuda/exl3_routed_half.cu -o operator-fatbins/exl3_routed_half.fatbin
nvcc --fatbin -O3 --expt-relaxed-constexpr -Izig/kernels "$@" \
  zig/kernels/cuda/exl3_experts.cu -o operator-fatbins/exl3_experts.fatbin
nvcc --fatbin -O3 -Izig/kernels "$@" \
  zig/kernels/cuda/norms.cu -o operator-fatbins/norms.fatbin
```

On each idle GB10, run the ARM64 executable with those three source-built images:

```sh
CUDA_VISIBLE_DEVICES=0 out-arm64/bin/tf-exl3-routed-half-contract operator-fatbins
```

The runner checks its complete operator registration, all compiler records, gates 1–4 and live launch limits.
`--offline operator-fatbins` runs only the compiler-record audit; it is not a device qualification.
The same fixture also joins the existing full CUDA backend contract runner.

## Results

Both runs exited 0, with the identical complete summary:

```text
SUMMARY 4 kernels: 4 pass gates 1-4, 0 fail, 0 known findings; 2 controls: 2 red as planted, 0 missed; 0 limit problems
```

Raw gate output: [GB10 A](gb10-a.txt), [GB10 B](gb10-b.txt).
The four cases cover shared-input and per-pair input with 192 and 384 experts.
Each covers eight served/edge shapes, including 4608/2304 widths and up to 2048 routed pairs.
Invalid negative and upper-bound picks must preserve a nonzero output sentinel.
The independent reference uses direct binary64 Walsh sums over separately rounded FP16 products.
All four accumulation stress fixtures reject FP16 and BF16 sums without changing the gate's bound.
Both controls use the unchanged wider-product kernel and fail byte equality at the intended rounding boundary.

The five compiler records are current: registers `26/26/26/20/20` on SM80/86/89/120/121,
zero static shared bytes, launch bound 32. Only SM121 was executed; SM80/86/89/120 remain compile-audited.
Metal, HIP and Windows were not run; no implementation for those backends changes here.
Peak host RSS was 424284 KiB on A and 424392 KiB on B; device/pinned peak telemetry was not collected.
This is not a model memory, performance, token-equality or fidelity receipt.

## Measured artifact identities

```text
358a2f0172a65d5be3ff12028814d7ea0d1e705ad9ccd425c2d790785b1dc6d2  tf-exl3-routed-half-contract
26165c1bde708ad103ba2cddc4b39f13d5ba1594afa9ff9e13ca0d61e9d3e594  exl3_experts.fatbin
d4e31b6c6dc2d7b2277dc5671b2fe98c7e9704f5a77f11fdafa4edcdc3bec171  exl3_routed_half.fatbin
9ab9f475ccce2eeaea5b0a9f84a4d5b925227dc898a51b08f87b6f4e2e0fdc14  norms.fatbin
```

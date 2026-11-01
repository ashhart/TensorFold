# Idol's denoiser on CUDA

Idol's denoiser (see `idol.md`) on the CUDA runtime, beside the Metal family and driven by the same case driver
(`zig/tests/idol_run.zig`): `tf-idol` on Metal and on CUDA reads the same case and runs the same gates. The image
encoders, the decoders and the PLY writer are not ported, and nothing registers with the lane engine or the server.

**Not run on a GPU yet.** It compiles for sm_120 and sm_121 and its host tests pass; no CUDA kernel of this family has
run on a device. Until a GB10 or an RTX 50 card has run the kernel contract and a case, treat it as unqualified.

`Model.load` admits sm_121 (GB10) and sm_120 (RTX 50, RTX PRO Blackwell) and refuses any other GPU by name.

## Precision

The same contract as on Metal, in CUDA's own terms:

- fp16 block projections on cuBLASLt (`CUBLAS_COMPUTE_32F`, fp16 inputs, fp32 output); `idol.cu` adds the bias to
  those fp32 sums and rounds once to fp16, GELU after it.
- RePo's maps, the condition embedding and the output projection in fp32 on cuBLASLt with `CUBLAS_COMPUTE_32F` and
  fp32 inputs: true fp32 (no TF32), so no hi/lo split is needed here.
- LayerNorms in fp32, rounded to fp16; the modulation in fp16 ops.
- Attention: `idol_attention.cuh`, head 64, fp16 q, k and v, fp32 scores, softmax and sums, scale 1/8, every query row
  to every key row. It is Idol's own kernel: no shared attention template gains an instance.
- `idol.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root). `F.normalize`'s clamp keeps a
  NaN norm, as on Metal.

## Kernel contract

`tf-cuda-test idol-contract` runs gates 1-4 over every `idol.cu` kernel and the attention, with three controls that
each drop one fp16 rounding point; the full contract runs it among its family sets. Gate-5 records
(`zig/src/core/contract/idol_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121. No SUMMARY line yet:
it needs a GPU.

## Run a case

```sh
zig build idol-cuda
zig-out/bin/tf-idol WEIGHTS case/
zig-out/bin/tf-idol WEIGHTS case/ blocks 0
```

The case comes from `tools/zig/idol_case.py`, as in `idol.md`.

## Background removal (BiRefNet)

BiRefNet's forward (Swin-L twice, the ASPP-deformable decoder) runs on CUDA in fp32, with its input and output
resizes. The host's preprocessing (`preprocess.zig`, `photo.zig`: decode, Pillow's resizes, erosion, crop and paste) is
the same code as on Metal, and `tf-idol-birefnet` on Metal and on CUDA reads the same case and runs the same gates
(`zig/tests/idol_birefnet_run.zig`).

**It compiles and passes host tests; it has never run on a GPU; GB10 qualification is pending.** `Net.load` admits
sm_121 and sm_120 and refuses any other GPU by name.

Precision, in CUDA's terms:

- Every conv and linear is `tf_br_conv`, an fp32 kernel with fp32 inputs. No TF32 and no cuBLASLt: cuBLASLt's fp32
  kernels do not document their summation order. In practice each output is one running sum per thread, or a split-K
  the heuristic picks, and one running sum over the squeeze's 51,840 terms drifts, as it did on Metal. So each output's
  sum runs in blocks of 256 terms. Each block is a sequential fp32 FMA chain from zero, and the blocks are added in
  order into the total.
- LayerNorm, BatchNorm (folded in float64 at load), window attention (scores, softmax and P V in fp32), the pooling and
  the gates use the same orders as on Metal.
- `idol_birefnet.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root). GELU uses `erff` and the
  sigmoids use `expf`, each within 2 ulps.
- The deformable conv is `tf_br_deform`, an fp32 kernel with torchvision's `deform_conv2d` sampling: bilinear over the
  four neighbours, zero outside the picture, the modulation 2 sigmoid(m), and a weighted sum that is not fused.

`tf-cuda-test idol-birefnet-contract` runs gates 1-4 over every kernel in `idol_birefnet.cu`. Seven controls each add
an fp16 rounding point or sum, or drop a NaN, and the full contract runs this set among its family sets. The gate-5
records (`zig/src/core/contract/idol_birefnet_cuda_records.zig`) are nvcc 13.0's for sm_80, 86, 89, 120 and 121.

```sh
zig build idol-birefnet-cuda
zig-out/bin/tf-idol-birefnet WEIGHTS case/ photo.png
```

The case comes from `tools/zig/idol_birefnet_case.py`, as in `idol.md`.

JPEG photos need libjpeg-turbo 3.1.4.1 (`libturbojpeg.so.0` on Linux); other versions are refused. A distribution's
package is usually another version, so build it from the official release tarball into a private prefix and point
`TENSORFOLD_LIBTURBOJPEG` at it:

```sh
curl -LO https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
echo 'ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  libjpeg-turbo-3.1.4.1.tar.gz' | sha256sum -c
tar xf libjpeg-turbo-3.1.4.1.tar.gz
cmake -S libjpeg-turbo-3.1.4.1 -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/opt/libjpeg-turbo-3.1.4.1"
cmake --build build -j && cmake --install build
export TENSORFOLD_LIBTURBOJPEG="$HOME/opt/libjpeg-turbo-3.1.4.1/lib/libturbojpeg.so.0"
```

The build needs CMake and a C compiler; with NASM installed it also builds the SIMD paths. The prefix may install to
`lib64/` instead of `lib/` on some distributions.

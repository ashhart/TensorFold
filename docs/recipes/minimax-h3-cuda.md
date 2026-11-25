# MiniMax H3 / FastH3 CUDA blocks, one device

This library contains @drowzeys' #958 BF16 transformer blocks and kernel contracts, with the maintainers' boundary,
shared-context, complete-block and allocation repairs. The later shared copy-source owner protects temporary uploads.
Weights and activations retain BF16 storage and products use the runtime's cuBLASLt Linear with FP32 sums.

The host supplies text encoding, sampling, projections, video/audio encoding and decoding. This is a block library,
not a complete native clip-generation endpoint. No weights are included. Code is Apache-2.0; MiniMax H3 / FastH3
weights have the MiniMax H3 Community License and its territorial limits.

Build the real library on a CUDA build machine:

```sh
zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast
```

This stage has no rank inputs, transport source, exchange kernel or libibverbs dependency. The C interface is
zig/src/families/h3/cuda_lib.h. Check tf_h3_abi_version() == 1 before passing its 96-byte tf_h3_inputs.
All tiles and rows belong to one device. The library retains the current primary CUDA context through Context.shared;
a host framework's device addresses must belong to that context. If no context is current, the runtime selects zero.

The same entry points support device addresses (tf_h3_forward), host addresses (tf_h3_forward_host), mapped host
allocation, profiling and BF16 projection replacement. The host owns pointer extents and serialized calls on a model.
The fixed 16,384-row cuBLASLt algorithm-selection shape is retained to preserve the original arithmetic policy.

Host tests and compile-only ABI libraries do not load a GPU image:

```sh
zig build test-h3-boundaries test-cuda-host test-cuda-copy-source compile-h3-library
```

Compile-only artifacts are installed under zig-out/host-checks and contain an empty image. They are verification
artifacts, not runnable inference libraries. The Linux library is cross-compiled and is not executed on this Mac.
The numerical contracts, actual kernels, model fidelity, real context interoperability and performance need CUDA hardware.

The original #958 thread reports one/two/three-GB10 results. Those measurements predate this split and do not qualify
this exact source. Sequence-parallel execution is a separate follow-up with an extended ABI and its own transport gates.

The Linux CUDA runner and all H3 contract adapters also have a compile-only gate:

```sh
zig build compile-cuda-tests -Dtarget=x86_64-linux-gnu
```

This step has no NVCC or Metal compiler dependency and does not execute the resulting binary.

## MiniMax-H3-Image's prompt path and image decoder on these blocks

The image pipeline's two CPU-bound stages run on the same CUDA blocks as the clip transformer, each writing the
file its Metal sibling writes, so a stage swaps on a Spark with no Metal and no PyTorch:

```sh
zig build tf-h3-image-prompt-cuda tf-h3-image-decoder-cuda -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt-cuda TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image-decoder-cuda MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
zig-out/bin/tf-h3-image-decoder-cuda reference MODEL_DIR out/case.latent.f32 128 128 out/case-gpu.png out/case-host.png 2048x2048
```

`tf-h3-image-prompt-cuda` runs Qwen3-VL to layer 50 and the token refiner with one layer's weights uploaded at a
time. `tf-h3-image-decoder-cuda` runs the VAE in bf16 over batched tiles. The CPU tools
(`tf-h3-image-prompt-host`, `tf-h3-image-decoder-host`) are independent reference implementations of the same case
writer and decoder chain. Both GPU tools have a synthetic selftest against the host's float64 reference: `selftest`
on the prompt tool, the default run on the decoder tool. `reference` decodes one latent on the GPU and on the host
reference decoder and fails past the 3e-2 full-chain relative bound.

The decoder's selftest gates each of the 24 blocks, on the GPU's own input, at 90% of values identical to the float64
reference (worst 8 ULPs, relative 2e-3), and the picture after two blocks at 80% of bytes identical and at most 2
apart, the Metal decoder's bounds (its recipe reports the measured at most 1 apart, not its bound). Its RMS norm sums
the row's squares in f64, so the norm's mean rounds as the reference's does, at no measurable cost (43.4 ms a tile
against 43.2 ms). The prompt's norm and head kernels sum their squares in f64 too, so their inverse roots are the host
reference's bits, and the head kernel rounds each normed value before the turn, as the host does. The prompt's selftest
gates its text, refiner and vision rows against the same float32-noise bounds.
On the GB10 both pass every check: the decoder's 37 checks pass and its lowest block (9) reads 92.037% identical.
[The run's lines](../receipts/h3-image-cuda-gb10-0bfee52e/README.md) are each block's. Other served SMs compile but
were not run.

Same prompt, seed 42, six distilled steps, BF16 throughout, one GB10:

| Stage | CPU reference | CUDA | Gate |
|---|---:|---:|---|
| Prompt, 2048² | 88 s | 18.0 s cold, 20.1 s warm mean | selftest passes |
| Prompt, 4096² | 98 s | 18.3 s cold, 18.5 s warm mean | selftest passes |
| Decoder, 2048² | 218 s | 5.6 s (5.25 s on the GPU) | selftest passes |
| Decoder, 4096² | 771 s | 20.4 s (20.3 s warm mean) | selftest passes |

The 2048² latent decoded on both paths is 56.65 dB PSNR from the host reference (86.180% of bytes identical, 99.923%
within 1, at most 5 apart, relative error 3.15e-3 against the 3e-2 bound). CPU times are the host reference tools on
the same case; the GPU times are warm means over three repeats after one cold run.

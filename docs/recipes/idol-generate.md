# Idol end to end on Metal

See [Idol on Metal](idol.md) for the checkpoint, encoders and decoders.

## End to end

`tf-idol-generate` (`zig/tests/idol_generate.zig`) chains the pieces above, from one preprocessed picture to a PLY:

1. DINOv3's feature1;
2. the VAE's feature2;
3. the denoiser's condition and its 50 guided steps;
4. the octree and Gaussian decoders;
5. the PLY and the camera.

Each model is loaded when its stage starts and freed when it ends. Background removal and cropping (BiRefNet, the SDK's
`preprocess`) are not ported, so the input is the picture `preprocess` returns, as float32 `pixels` [3, 1024, 1024] or
`image` [1024, 1024, 3] that `idol_case.py dino` or `vae` writes; either name in the other layout is refused.

```sh
zig build tf-idol-generate -Dcpu=apple_m1
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO case/vae.safetensors out/   # seed 42, 50 steps, cfg 3, 262,144
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO case/vae.safetensors out/ --noise case/vae.safetensors
zig-out/bin/tf-idol-generate WEIGHTS KLEIN/vae DINO - out/ --features dino.safetensors --features vae.safetensors
```

Noise:

- **The starting noise** is `torch.Generator("cpu").manual_seed(seed)`'s randn: [8192, 16], then [5], as
  `idol_case.py case` draws it. The tool reimplements torch's mt19937 and its Box-Muller. The camera's 5 values are
  bit-equal to torch's. 28 of the 131,072 latent values are 1 ulp off, from torch's float cos and sin.
- **The VAE posterior's noise and the octree's draws** come from native seeded generators.
- **`--noise FILE`** gives the posterior's noise (`noise`, as `idol_case.py vae` records it). If FILE also holds
  `latent` and `camera`, they replace the starting noise.

It writes `generate.ply`, `camera.json`, `start.f32`, `sample.f32` and `tf-idol-generate.json`, each stage's seconds.

### Measured

Setup:

- The avocado example and the same M5 Ultra and torch 2.13.
- Seed 42, 50 steps, guidance 3, 262,144 Gaussians.
- The SDK runs its own `IdolPipeline.__call__` on MPS once (`optimize` is CUDA-only), each stage timed by wrapping
  the SDK's methods with `torch.mps.synchronize()`. Its PLY is byte-identical to an earlier run.
- Ours runs twice, each in a fresh process. The weights were in the page cache both times, so cold and warm differ
  only by the process.

| stage | ours, cold | ours, warm | SDK, MPS |
| --- | ---: | ---: | ---: |
| load (all models) | 0.67 s | 0.67 s | 2.11 s (and 2.75 s of imports) |
| preprocess (BiRefNet) | not ported | not ported | 0.51 s |
| DINOv3 | 0.158 s | 0.155 s | 0.275 s |
| VAE | 0.245 s | 0.241 s | 0.119 s |
| condition | 0.072 s | 0.071 s | in the sampling |
| denoise, 50 steps x 2 | 27.49 s | 27.50 s | 93.89 s |
| octree | 0.145 s | 0.143 s | 0.612 s |
| Gaussian decoder (+ activation) | 0.222 s (+ 0.001) | 0.222 s (+ 0.001) | 0.491 s (+ 0.013) |
| PLY and camera | 0.004 s | 0.004 s | 0.018 s |
| total | 29.09 s | 29.07 s | 100.81 s (95.41 s generating, less preprocess) |

Memory, the whole process under `/usr/bin/time -l`:

| | peak footprint | max RSS | GPU |
| --- | ---: | ---: | --- |
| ours | 2.71 GB | 1.94 GB | 2.40 GB Metal allocated at most |
| SDK | 12.88 GB | 6.20 GB | 5.01 GB MPS allocated, 12.64 GB driver (sampled) |

### Quality

The PLYs cannot be equal:

- **Starting noise:** the SDK draws it from MPS's generator after the VAE's draw, which cannot be reproduced natively.
- **VAE posterior noise:** MPS's generator too, drawn in bf16.
- **The octree's draws:** torch's multinomial and jitter in the SDK, a native generator here. One count's difference
  changes the topology from that level down.
- **The denoiser:** fp16 over 50 guided steps, and our encoders are not bit-equal to the SDK's.

The starting noise matters most. With the same features, the SDK's own sample from MPS noise and its sample from the
CPU-seeded noise have cosine 0.479. Their PLYs are as far apart as ours is from the SDK's.

So the PLYs are compared as distributions. Statistics are per PLY: xyz in PLY units, opacity and scale activated. The
NN columns, against the SDK's pipeline PLY, are:

- the mean symmetric nearest-neighbour distance between 4,096 points sampled from each;
- the same over all 262,144 points.

Two independent 4,096-point samples of the SDK's own PLY lie 0.00896 apart.

| run | xyz std | opacity mean (q50) | scale q50 | NN, sampled | NN, all points |
| --- | --- | --- | --- | ---: | ---: |
| SDK pipeline | 0.189, 0.078, 0.304 | 0.885 (0.9986) | 0.00096 | 0.00906 | 0 |
| ours, native noise | 0.176, 0.073, 0.320 | 0.423 (0.181) | 0.00113 | 0.04623 | 0.0365 |
| ours, SDK's VAE noise | 0.176, 0.074, 0.320 | 0.425 (0.186) | 0.00113 | 0.04627 | 0.0366 |
| ours, SDK's VAE and starting noise | 0.189, 0.079, 0.304 | 0.887 (0.9986) | 0.00096 | 0.00922 | 0.00166 |
| SDK, same features, CPU-seeded noise | 0.176, 0.074, 0.319 | 0.422 (0.179) | 0.00113 | 0.04617 | 0.0368 |

The table shows:

- **Every PLY:** 262,144 Gaussians, all finite.
- **The SDK's VAE noise alone** changes little: the starting noise still differs.
- **With both of the SDK's noises:** our sample has cosine 0.99835 to the SDK's. The PLYs are as close as the sampling
  can resolve: 0.00922 against the 0.00896 baseline, 0.0017 over all points. The camera is yaw 1.6956, pitch 0.5397
  and fov 0.1094, against 1.7006, 0.5413 and 0.1096.
- **Repeat:** our cold and warm PLYs are byte-identical.

### Parity chain

This run gives the SDK's own recorded feature1 and feature2 (`--features`) and the CPU-seeded noise. The SDK's
reference is the same features and noise: the `idol_case.py case` sample, and its `decode` PLY.

- **Sample:** cosine 0.999991 to the SDK's.
- **Camera:** its latent's 5 values within 0.0022 of the SDK's.
- **PLY** against the SDK's decode of its own sample:
  - mean NN 0.00981 between 4,096-point samples (the baseline 0.00944);
  - 0.0018 over all points;
  - xyz mean 6e-5 apart.

From the picture, our encoders move the sample to cosine 0.99437 with native VAE noise. With the SDK's noise, they
move it to 0.99953. The PLY stays 0.0021 over all points from the SDK's decode.

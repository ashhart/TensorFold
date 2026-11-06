# MOSS-Image on CUDA

`zig/src/families/moss_image` also runs MOSS-Image's transformer on the CUDA runtime (`cuda.zig`, `cuda_model.zig`):
the same denoiser as on Metal ([moss-image.md](moss-image.md)), from the same released bf16 checkpoint, driven by the
same `tf-moss-image` tool and checked against the same publisher pipeline.

- A prompt's rows once a prompt: the input projection and the two refiner blocks.
- The image and prompt rows together at each step: every joint block under one shared modulation, full attention
  over both.
- The card's guidance: two passes a step, the prompt and the empty negative prompt with block 22 skipped.

The prompt encoder and the image decoder are not ported; `tools/zig/moss_image_case.py` runs them in torch, as on
Metal. It is a denoiser, not a token family: it does not register with the lane engine, the server or
`families.detect`.

The config, the rotary tables and the timestep's modulation (`config.zig`, `host.zig`) are shared with Metal. The
weights are checked as on Metal: the released geometry only, and no tensor left unread.

It runs on the two GPUs it was checked on, sm_120 (RTX 5090) and sm_121 (GB10): `Model.load` refuses any other.

## Precision

The Metal family's rounding, step for step:

- bf16 activations, rounded where the publisher's torch code rounds them.
- fp32 sums in every projection and in attention. Projections are cuBLASLt bf16 GEMMs with fp32 accumulation
  (`vit.Gemms`, as the vision towers use); the biased ones add the bias to the fp32 sums and round once
  (`vit.linear`).
- fp32 for norms and softmax; the rotary tables in bf16; the time MLP and modulation on the host, once a timestep.
- No int8, no fast math: `moss_image.cu` builds with the glue flags (`--fmad=false`, IEEE division and square root).

Attention is `prefill_attention.cu`'s full (non-causal) `vision_kernel`, which the vision towers use at head dims 72
and 80, here instantiated at 128 (eight warps, 69,632 bytes of shared memory). The new instance leaves the existing
ones as they were: their SASS is identical before and after (sm_80, 86, 89, 120 and 121; nvcc 13.0, 13.2 and 13.3).

`moss_image.cu` holds the family's own steps, each a port of `moss_image.metal`'s kernel:

- `tf_moss_norm`: RMSNorm, then optionally the modulation's scale and shift;
- `tf_moss_qk`: each q and k head's RMSNorm and rotary, q, k and v packed into attention's input rows (both the
  joint blocks' separate projections and the refiner's packed one);
- `tf_moss_situ`: the situ gate, over the MLP's (hidden | gate) rows or two buffers;
- `tf_moss_gate_add`: the gated residual add (ones for the refiner's plain residual).

## Kernel contract

`zig/tests/cuda/contract_moss_image*.zig` registers the four kernels in each form the family launches, and the head-128
attention:

- each output is checked against float64 at the kernel's bf16 points, in the range its fp32 steps allow (IEEE division
  and square root, two ulps for `expf` and `tanhf`);
- shapes are the served ones (4,106 to 4,188 joint rows, 10 to 63 prompt rows, widths 3,072 and 8,192) and the tiles'
  edges; the attention runs all 24 heads at the prompt lengths and two at the joint lengths;
- five controls each drop one bf16 rounding point (the norm's, the modulation's, the rotary's, the sigmoid's and the
  gate's), and gate 3 must catch each;
- the attention is checked against `contract_attn.zig`'s float64 reference; its gate 4 uses P x V sums that fp16 or bf16
  sums would miss.

Gate 5's records are in `moss_image_cuda_records.zig` (sm_80, 86, 89, 120 and 121; the same on nvcc 13.0, 13.2 and
13.3, so no later release needs a set of its own).

`tf-cuda-test moss-contract` on a GB10 (nvcc 13.0 and 13.2) and an RTX 5090 (nvcc 13.3):

```
SUMMARY moss image: 8 kernels, 8 pass gates 1-4, 0 fail; 5 controls: 5 red as planted, 0 missed
```

## Run a case

```sh
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 --local-dir MODEL_DIR
python3 tools/zig/moss_image_case.py case MODEL_DIR case/ --prompt-file prompt.txt --device cuda
zig build install -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121   # -Dsm=120 for an RTX 5090
zig-out/bin/tf-moss-image MODEL_DIR/transformer case/case.safetensors case/
python3 tools/zig/moss_image_case.py decode MODEL_DIR case/latents.f32 case/zig.png --device cuda
python3 tools/zig/moss_image_case.py compare case/reference.png case/zig.png case/side.png
```

The case tool needs diffusers 0.38 or later: the checkpoint's decoder config names `decoder_block_out_channels`.

## Measured

Setup:

- The Metal doc's two cases, their prompts and seeds: a fox at 1024x1024 (4,096 image rows, 34 prompt rows, seed 42)
  and the card's Chinese sticky note at 1200x880 (4,125 image rows, 63 prompt rows, seed 7); the negative prompt has
  10 rows. Each case holds a float32 run of the same transformer at all 50 steps (`--fp32-steps 50`).
- Reference: the checkpoint's `pipeline_moss_image.py` on torch 2.11.0 (CUDA, bf16), diffusers 0.38.0, transformers
  5.12.1, on the same GB10.
- The card's defaults: 50 Euler steps, guidance 3.

Results on a DGX Spark (GB10, sm_121, 121 GB unified memory, nvcc 13.2):

- **Velocity**, both branches of each of the 50 steps on the pipeline's own input:
  - Against torch bf16: cosine 0.99972 to 0.99998 (fox) and 0.99981 to 0.99998 (sticky note).
  - Against the float32 run, mean 1 - cosine: fox prompt 1.598e-4 (torch bf16's own 1.596e-4), negative 1.693e-4
    (1.707e-4); sticky note prompt 1.602e-4 (1.602e-4), negative 1.759e-4 (1.748e-4). Ours is the closer of the two at
    23 and 29 of the fox's 50 steps, 24 and 27 of the sticky note's.
- **Image:** our latents through the pipeline's decoder are 45.93 dB PSNR from the pipeline's own picture for the fox
  and 32.76 dB for the sticky note (its handwritten text the same; fine detail such as cabinet handles differs, as on
  Metal: 50 guided steps magnify bf16 rounding).
- **Repeat:** the same step twice, another between, gives identical bytes; two runs of a case give identical latents.
- **Speed**, the GPU otherwise idle, the median of a case's 100 forwards:
  - torch bf16: 1.402 s a forward for the fox, 1.477 s for the sticky note.
  - Ours: 0.690 s and 0.700 s, 2.03x and 2.11x torch's (69.6 s and 70.6 s for 50 guided steps). 0.2 s for both
    prompts' refiner once.

Results on an RTX 5090 (sm_120, 32 GB, driver 615.71, nvcc 13.3), on the same two cases (their inputs, torch bf16
velocities and float32 runs as recorded on the GB10):

- **Velocity:** against torch bf16, cosine 0.99967 to 0.99998 (fox) and 0.99978 to 0.99998 (sticky note). Against the
  float32 run, mean 1 - cosine: fox prompt 1.607e-4, negative 1.694e-4; sticky note prompt 1.612e-4, negative
  1.762e-4 (torch bf16's own as above). Ours is the closer at 23 and 21 of the fox's steps, 25 and 29 of the sticky
  note's.
- **Image:** 44.01 dB PSNR from the pipeline's picture for the fox, 34.51 dB for the sticky note. The latents differ
  from the GB10's in their low bits: cuBLASLt picks different GEMM kernels on each GPU.
- **Repeat:** identical bytes, as on the GB10.
- **Speed**, the GPU otherwise idle:
  - torch bf16 (torch 2.13.0, the pipeline's own forward on this card): 0.307 s a forward for the fox, 0.309 s for the
    sticky note.
  - Ours: 0.243 s and 0.245 s, 1.26x torch's (24.6 s and 24.8 s for 50 guided steps).
- **Memory:** the transformer's bf16 weights (9.5 GB) on the device, plus scratch sized at load for one image and the
  longest prompt: 0.51 GB at 1024x1024 with a 34-row prompt (from the allocations, not measured), and cuBLASLt's
  32 MiB workspace.

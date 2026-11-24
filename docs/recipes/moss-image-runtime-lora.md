# MOSS-Image: runtime LoRAs, never merged

`tensorfold-native serve` on a MOSS-Image checkpoint runs LoRAs a request names, on top of the checkpoint's own
bf16 weights. Nothing is merged and no second copy of the transformer is written: an adapter loads to the GPU the
first time a request names it and stays there, so requests pick, stack and re-weight them freely.

## Use

Put adapters in a folder: `<checkpoint>/loras/`, or the folder `TF_LORA_DIR` names. An adapter is either
`NAME.safetensors` (kohya keys `<module>.lora_down.weight` / `lora_up.weight` / `alpha`, or PEFT
`<module>.lora_A.weight` / `lora_B.weight`) or a PEFT folder `NAME/` with `adapter_model.safetensors` and
`adapter_config.json`. Common prefixes (`diffusion_model.`, `transformer.`, `base_model.model.`, `lora_unet_` with
underscores) are accepted.

`core.runtime_lora` reads and checks every adapter: each factor must name one of the projections below with the
projection's shape, values must be finite, and a PEFT config may not ask for DoRA, rsLoRA, per-module ranks or
alphas, trained biases or `modules_to_save`. A kohya file is renamed to PEFT names first and needs one rank and one
alpha for all its modules (alpha = rank, scale 1, when it has none). The factors are checked in f32 and keep the
file's values exactly on the device: B as f32, and A as bf16 pieces (as the rows it multiplies are) whose sum is A
exactly, one piece for a bf16 adapter, two for most f16 values, three for f32. Each piece is cut toward zero, so an
f32 value past bf16's largest stays finite. A piece is never subnormal (the GPU may flush those), so an A value
that would need one fails the request: a subnormal value, or an f32 one below about 2^-103 with low bits set.

```sh
TF_LORA_DIR=~/models/moss-loras tensorfold-native serve --host 127.0.0.1 --port 8020 --name moss-image ~/models/moss-image-base
curl -s localhost:8020/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "moss-image", "prompt": "a red fox in a birch grove", "steps": 8, "guidance": 1,
  "loras": ["moss-8step-dmd"] }'
# stacked, each with its own strength (scale 1 when left out, -16 to 16)
  "loras": [{"name": "moss-8step-dmd", "scale": 0.5}, {"name": "my-style", "scale": 0.8}]
```

A request without `loras` (or with `[]`, or a LoRA at scale 0) runs the plain weights. An adapter stays on the device
until the server stops, and its file is read once: restart the server after changing a file in the folder. Up to 8 adapters a request. A name is a file name
in the folder (letters, digits, `.`, `_`, `-`). A missing adapter, or one that cannot be read or fails a check, fails
that request alone with a sentence saying which; the server log names the path, module or reason.

## Arithmetic

Every projection `cuda_model.zig` runs through `linear` / `linearBias` (the 42 joint blocks' q, k, v, o,
ffn_up_gate, ffn_down; the refiner's qkv, proj, ffn_up, ffn_gate, ffn_down; the text, video and final projections)
can carry adapters. A projection no picked adapter targets runs exactly as without LoRAs. For one that is targeted,
the sums stay fp32 until one rounding to bf16 (with the bias, for `linearBias`):

`sums = x Wᵀ + Σ scale · (x Aᵀ) Bᵀ`, where `x Aᵀ` is one bf16 GEMM per A piece into the same fp32 sums (bf16
products are exact in fp32), kept in fp32 and multiplied by B in fp32 (cuBLASLt, fp32 compute, no TF32, beta 1 onto
the sums). There is no other mode.

`tf-moss-image-lora TRANSFORMER_DIR WORK_DIR [LORA_DIR NAME]` (`zig build moss-image-lora-cuda`) checks this
against FP64 on real weights, per projection class, and prints torch's unmerged bf16 order beside it.

Merging instead rounds `W + scale·B A` to bf16 once per weight, which drops any change smaller than the weight's
own bf16 step. The MOSS 8-step DMD adapter is exactly that case (its README says to apply it unmerged): a merged
bf16 copy kept ~66% of its change by projection.

## Measured (GB10, 1024², 8 steps, guidance 1, seed 42, warm)

| | picture |
|---|---|
| no LoRA (the same bytes as without this feature) | 10.0 s |
| runtime LoRA (the DMD2 adapter: f32, rank 128 on all 252 block projections, A in three pieces) | 14.5 s |

First use of an adapter: about 2.2 s to read, check, cut and upload (that adapter: 1375 MiB on the device). Switching
between cached adapters, scales or stacks costs nothing extra. On every projection class the LoRA outputs sit at
bf16's own rounding of the FP64 sum taken from the file's values (relative RMS 1.65e-3 to 1.67e-3, worst element 0.50
of a bf16 step); torch's unmerged order is at 1.71e-3 to 2.36e-3 on the same rows.

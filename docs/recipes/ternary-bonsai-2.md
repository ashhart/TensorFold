# Ternary Bonsai 2 27B

Prism ML's Bonsai 2 27B is Qwen3.8-27B with ternary weights stored in a rotated basis. Its MLX pack
(`model_type: prism_hadamard_qwen35`) keeps every projection as 2-bit codes in groups of 128. The weights
are ternary, so each group's codes decode to -s, 0 or +s. The pack's `hadamard.json` names the transform
each projection's input needs: fixed ±1 signs, then a normalized Walsh-Hadamard transform over 1024-wide
blocks. The embedding stores its rows in the same basis and turns them back on lookup.

```bash
tensorfold pull prism-ml/Ternary-Bonsai-2-27B-mlx-2bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
```

The `bonsai` family runs the pack on the Qwen3.8-27B lane engine:

- Every projection's rows go through a row-exact rotation kernel (the signs, then the transform, then one
  rounding to bf16) before the lane matmul. Projections that share an input rotate it once.
- A lookup dequantizes the 2-bit row and rotates it back.
- The recurrent layers' unquantized fp32 gates (`in_proj_a`, `in_proj_b`) run through a fixed-order dense
  kernel.
- Norms, convolution and state parameters are held in bf16, as in the Qwen3.8-27B checkpoint.
- The vision tower in the pack is not loaded. TensorFold serves the text model.

Only the version-1 Prism contract is read: blocks of 1024, explicit signs, grouped DeltaNet activations and
no MTP head. Any other pack is refused from its configuration files before its weights download.

## Chips

- M5 tensor-unit GPUs read the 2-bit codes through the lane kernels. Each group-128 scale and bias covers two
  groups of 64 there, and each fp16 scale rounds to bf16 once at load.
- M1 through M4 run the row decoder on the codes widened exactly to 4 bits, which takes 13.9 GiB rather than
  7.2 GiB.
- Where the widened codes don't fit beside 12 GiB for the drafter and a prompt chunk (Macs with 32 or 36 GB), the
  row decoder reads the pack's 2-bit codes and fp16 scales as stored instead, through the packed affine row kernel
  ([quantized checkpoints](../quantization.md)). A forward then takes 1.3x as long at one row and 2.1x at 16 rows
  on an M3 Ultra.
- Between the two, the first layers widen as far as the budget allows: a 36 GB Mac's default 25.2 GiB widens 60 of
  64. Outputs are exact within a server; how many layers widen depends on the memory budget, so servers with
  different budgets can differ in their bits (the startup line says how many).
- With the drafter, the model needs about 20 GiB of budget, so 16 and 24 GB Macs can't serve it.

Drafting uses Qwen3.8-27B's DFlash2 model and context copies. The target verifies every draft against
its own serial sample, so drafted output equals `"draft": false` output.

Against mlx_lm running the pack's own runtime (paired runs on the same machine, the range over two runs a chip):

| Chip | Decode, code | Decode, chat | Prompt processing |
|---|---|---|---|
| M5 Max | 3.7-5.8x | 1.7-2.2x | 1.4-1.7x (2k to 32k) |
| M3 Ultra | 2.1-2.9x (94-111 vs 38-46) | 1.3-1.7x (60-67 vs 39-46) | 1.15-1.28x (2k to 64k) |

Replies are exact on both: drafted equals serial, resumed equals fresh, and concurrent streams equal their solo
runs. On the M3 Ultra the packed 2-bit rows (a 32 GB Mac's budget) are exact too, at 40-80 tok/s.

Bonsai 2 27B is by Prism ML (Apache-2.0).

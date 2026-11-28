# Gemma 4 E4B, native Metal

This recipe covers text input with the E4B of the Gemma 4 E-series (`model_type` `gemma4`, 42 layers, a per-layer input
of 256 values a layer, and the last 18 layers reading the cache of an earlier one), in the revision NVIDIA ships as
Nemotron-3.5-Bolt-E4B, converted to MLX affine 4-bit in groups of 64. It runs on the native Gemma 4 family that serves
[the 31B](gemma-4-native.md) and [the 12B](gemma-4-12b-native.md). Build with Zig **0.17.0**, as described in
[the native preview](../../ZIG-PREVIEW.md#build).

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 --revision 49843eed8db1f1c616c47af75cfd36e07aec0667 \
  --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
python tools/zig/gemma4_e4b_convert.py --dense-ple "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16" "$HOME/models/bolt-e4b"   # mlx installed for --check-mlx
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/bolt-e4b" --port 8888 --parallel 4 --context 32768
```

`--dense-ple` keeps the three per-layer linears in bf16, as the engine reads them; a conversion without it is refused
(`UnsupportedGemmaQuantization`). The converter names the tensors as mlx-lm does (`language_model.model.`).

## Serving

| Checkpoint / platform | Qualification |
| --- | --- |
| The converted E4B, M4 Max, macOS 27 | Greedy and sampled decode, four streams, prompt reuse, the server. |
| M5 | Refused at load: a chip with tensor units would take the lane projection, and nothing has run there. |
| M1 to M3 | Not qualified; the row path is the one the M4 Max runs, and nothing has run on these chips. |
| Drafts | None: `--drafter` is refused for the E4B (the MTP assistant drafts for the 12B, DFlash for the 31B). Copy drafts work. |
| Images, audio | Not read: the converter drops both towers. |
| Other widths | The attention and MLP linears and the tables at MLX affine 2-6 or 8 bits in groups of 64; the per-layer table must be 4-bit, the per-layer linears bf16. |

On an M4 Max, the whole prompt runs in decode windows of up to 128 rows, its last row alone (the rows before it skip
the 18 layers that read an earlier layer's cache), so a prompt's logits are bit-equal to decoding it token by token:
`tf-gemma4-e4b-prompt-logits`' served last row equals the windows of `tf-gemma4-e4b-logits` bit for bit on the 16
reference prompts and six prefixes of the 6,422-token one, and each served greedy reply is the windows' argmax at every
position. A prompt resumes every 128 tokens.
Measured with `tools/bench_openai.py`, `tools/bench_cache.py --require-cold` and `tools/bench_concurrent.py --alone
--serial`: decode 100 tokens/s on one chat stream; a cold prompt prefills at about 1,735 tokens/s at 2k, 1,667 at 8k and
1,368 at 32k; every reply equals its run alone and a resumed prompt, a cancelled pass's included, gives the fresh pass's
reply. The server keeps 4.1 GiB wired.

`tf-gemma4-e4b-logits` writes the teacher-forced logits of a reference's prompts, and `tf-gemma4-e4b-ple` with
`tf-gemma4-e4b-ple-host` compares the GPU's per-layer inputs with `layer_input.zig`'s.

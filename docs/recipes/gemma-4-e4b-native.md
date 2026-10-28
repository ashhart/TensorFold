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

On an M4 Max, a prompt's first 128 rows run in decode windows (a short prompt answers sooner), the rest in chunks of 512 on
affine_mm's tiles over the row path's packed weights, with the per-layer inputs in the chunk. The tiles sum in another
order than a decode step, so a prompt's logits differ from decoding it token by token: with `tf-gemma4-e4b-logits`
(`E4B_TILE=1` against its windows) the largest logit difference is 2.9 over 617 positions, top-1 agrees on 98.7% (the 8
misses are near ties) and the mean KL is 0.0007; against the BF16 reference the windows give 93.8% top-1 and KL 0.073,
the tiles 93.0% and 0.075. `GEMMA_E4B_WINDOWS=1` takes the windows for the whole prompt.
Measured with `tools/bench_openai.py`, `tools/bench_cache.py --require-cold` and `tools/bench_concurrent.py --alone
--serial`: decode 99 tokens/s on one chat stream, 212 tokens/s over four; a cold prompt prefills at about 2,400
tokens/s; every reply equals its run alone and a resumed prompt gives the fresh pass's reply. The server keeps 4.2 GiB wired.

`tf-gemma4-e4b-logits` writes the teacher-forced logits of a reference's prompts, and `tf-gemma4-e4b-ple` with
`tf-gemma4-e4b-ple-host` compares the GPU's per-layer inputs with `layer_input.zig`'s.

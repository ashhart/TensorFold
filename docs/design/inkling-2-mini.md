# Inkling 2 Mini on CUDA and Metal: a plan in claimable slices

The maintainer agreed this plan and its gate on 2026-11-21 in pull request #4274's discussion. The maintainers are building the host slices and all three backends themselves for the next release; a contributor who wants a slice says which one in that discussion before starting, so nothing is built twice.
It splits the work into slices small enough for one contributor each.
#4265 proposes the same model on HIP; the host slices below are shared with it and should have one owner.

## Decisions

Answered by the maintainer in #4274's discussion on 2026-11-21; answer 2 carries the later correction.

1. Inkling 2 Mini is wanted on CUDA, Metal and HIP, as one plan with #4265, with the host slices shared.
2. The contract for the routed experts is the reference pull request's arithmetic: the stored NVFP4 and MXFP8 weights decoded in registers, bf16 activations, FP32 sums, with the per-expert NVFP4 factor rounded to bf16 as it loads. The activation quantization `hf_quant_config.json` declares is not pinned down by the file itself and can come later as an opt-in with its own quality gate.
3. sm_12x first, with the stored FP4 and FP8 arithmetic. Older chips can follow as their own pull request.
4. Metal starts from the original checkpoint (BF16 dense weights, the stored experts, the drafter file), so all three backends run the same weights.
5. The loop lives in the `inkling2` family.
6. The gate under "Gate" is agreed. The speed bar: decode no slower than a standard server on the same machine, with and without the drafter.
7. No separate issues for now: the maintainers are building the host slices and all three backends in parallel themselves for the next release. A contributor who wants a slice says which one in the discussion before starting.

## What main has

On main `72ebe5f6e`:

- `zig/src/families/inkling2` is Inkling 2 Small's host package (#3578): config, weight names, layer plan, chat template and tokenizer tests. It accepts `inkling2` only, so Mini's `inkling2_loop` is refused.
- CUDA has Inkling 2's attention kernels (#4125) and glue kernels (#4126) under the kernel contract.
- Open for Small on CUDA: expert and vision kernels (#4127), drafter kernels (#4128), the reply split (#4124) and the EXL3 loader (#3612).
- Metal has no Inkling 2 kernel. It has NVFP4 code for Drama 3 and an attention residual for Kimi K3; whether either fits Inkling 2 is not established here.
- No backend serves Inkling 2 end to end.

## What Mini is

From `thinkingmachines/Inkling-2-Mini` (Apache-2.0), `config.json` and the model card as published on 2026-11-20:

- 26.4B parameters, text and images in, text out. One 19.5 GB safetensors file and a 0.64 GB drafter file.
- 24 layers of weights, hidden 2,304. Layers 6 to 17 run three passes, so a token makes 48 layer visits.
- Layers 5, 11, 17 and 23 see the whole context; the other 20 slide over 1,024 tokens. 24 query heads and 8 key heads of 128.
- 128 routed experts of 1,152 a layer, 8 a token, no shared expert. NVFP4 in layers 0 to 19 and MXFP8 in layers 20 to 23; everything else is BF16 apart from a few FP32 tensors.
- A vision tower of 23 layers, 1,024 wide, patch 20 with a 2 by 2 merge.
- A DSpark drafter in `mtp.safetensors`: 3 layers, blocks of 8 tokens, taps at visits 5, 23, 41 and 48.
- The tokenizer and chat template are the same files as Small's, as #4265 reports.

What Mini adds over Small is the loop: a cache slot per visit, rank-64 gates on the core layers' inputs in passes 2 and 3, and learned cross-pass queries over the previous pass.

Memory, estimated from the config and not measured:

- The global layers are visited 8 times. With 8 key heads of 128 in BF16 that is 32 KiB a token: 4 GiB at 128K and 32 GiB at 1M.
- The sliding layers are visited 40 times; their rings hold about 160 MiB a stream.
- So one GPU or one Mac holds the model, and no slice below needs more than one rank.

## Upstream references

- The loop's reference code is not released: it is in Transformers' open pull request huggingface/transformers#51050 (head `6173a2f6` on 2026-11-20), on top of the released `inkling2` model.
- That pull request dequantizes the NVFP4 and MXFP8 experts on load and marks this as temporary. Its arithmetic is the experts' contract: stored weights decoded in registers, bf16 activations, FP32 sums.
- SGLang's open pull request sgl-project/sglang#51926 runs the experts with the activation quantization `hf_quant_config.json` declares, which is not the contract and is at most a later opt-in comparison. As reported in this plan's review, its rounding differs between its SM100-series and SM120 paths; neither point is checked here. If the opt-in is built, H3, C3 and M3 name the path and its rounding points.
- llama.cpp merged the model on 2026-11-20 (ggml-org/llama.cpp#32694), so it is one candidate for the standard server on both backends. vLLM is another on NVIDIA.
- No speed for a DGX Spark or for Apple silicon was found published; the receipt's baseline has to be measured on the claimant's machine.

## Slices

Each slice is one pull request with its own tests. "Needs" names what must land first.

### Host, shared by CUDA, Metal and HIP

| Slice | What | Needs |
|---|---|---|
| H1 | `families/inkling2` accepts `inkling2_loop`: parses `loop_config`, refuses `is_varloop`, builds the visit plan (layer, pass, cache slot, message sources) and counts drafter taps by visit. Small's tests pass unchanged. This is #4265's first slice. | none |
| H2 | Names for the controller and cross-pass tensors, and each layer's expert precision read from Mini's headers, with fixtures from the published config and a tiny checkpoint. | H1 |
| H3 | The loop's arithmetic written down: where each value rounds and the order of every sum for the gates and the cross-pass message, a float64 model, and single-change probes that turn the host tests red. It covers a message that reads two blocks of the previous pass, which Mini's core has and the tiny checkpoint's does not. | H1 |
| H4 | The drafter's host side: `mtp.safetensors` names and shapes against Mini's header, and its four taps mapped to visits and cache slots. Taps 5, 23 and 41 are the inputs to zero-based decoder visits 5, 23 and 41; tap 48 is the post-final-norm output, not a cache slot. | H1 |

### CUDA

| Slice | What | Needs |
|---|---|---|
| C1 | Mini's shapes on Small's kernels: served shapes, tiles and limit records for hidden 2,304 and experts of 1,152. | #4126 to #4128 |
| C2 | The loop kernels (gates and cross-pass message attention) under the kernel contract. | H3 |
| C3 | The routed experts as stored, NVFP4 and MXFP8, on sm_12x, with the stored weights decoded in registers, bf16 activations and FP32 sums. Older chips follow as their own pull request. | none |
| C4 | The engine at one rank, text only: 48 KV and conv slots in the pool and in prompt-cache snapshots, the lane adapter, admission and the family registration. Images are refused until C6. | H2, C1 to C3 |
| C5 | DSpark: the checkpoint's drafter on #4128's kernels, with taps by visit, through the shared lane rounds. Drafted output equals plain decode, and decode with it is no slower than without. | C4, H4, #4128 |
| C6 | Images on #4127's vision kernels at Mini's tower shape. | C4, #4127 |
| C7 | The served receipt, the recipe page and the family's optimization ledger cells. | C4 |

### Metal

| Slice | What | Needs |
|---|---|---|
| M1 | Sliding and global attention with the sink, the relative bias, log scaling and the L2 clips, on the arithmetic reference #4125 follows. A prompt path for M5 and one for earlier chips. | none |
| M2 | Glue kernels: embedding, the block attention residual, the short convolutions (static and input-dependent), the router, soft-capped SwiGLU and the pick. | none |
| M3 | The routed experts as stored: NVFP4 and MXFP8 decoded in registers, bf16 activations, FP32 sums. | none |
| M4 | The loop kernels under the kernel contract. | H3 |
| M5 | The engine, text only: 48 slots, the lane adapter, admission by memory class and the family registration. Starts from the original checkpoint. | H2, M1 to M4 |
| M6 | DSpark: the drafter's kernels under the kernel contract, then the drafter through the shared lane rounds. MiniMax M3 has the one DSpark drafter on Metal to adapt (`families/minimax_m3/dspark.zig`); Kimi K3's Metal adapter refuses external draft heads. Same checks as C5. | M5, H4 |
| M7 | Images: the vision tower and projector. | M5 |
| M8 | The served receipt, the recipe page and the ledger cells. | M5 |

H1 to H3, M1 and M2 can start now. C1 waits on Small's stack; C3 and M3 run the bf16-activation arithmetic and have no other dependency.

## DSpark

DSpark support is part of the plan on both backends, not an extra: H4, C5 and M6 carry it, and C7 and M8 fill the family's DSpark cell in the optimization ledger.
The engine slices C4 and M5 land first without it, so plain decode exists as the reference the drafter is checked against.
The drafter taps by zero-based decoder visit: 5, 23 and 41 are the inputs to those visits, and 48 is the output after the final norm, not a cache slot.

## Gate

The gate is agreed; the speed bar below is its threshold.

- **Kernels:** each backend's full kernel-contract `SUMMARY` with every planted control red, on the chips named in the pull request.
- **Same-engine exactness,** on the tiny checkpoint and on the released one, greedy and sampled: drafted equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. Seams are checked at every pass boundary, for KV, conv state and final logits.
- **Fidelity:** teacher-forced histories against the reference huggingface/transformers#51050, with every top-token difference and its margin recorded.
- **Speed:** a served receipt as CONTRIBUTING describes (decode with and without the drafter, cold first token at one and four streams, cold 2k, 8k, 32k and 64k prompts, peak memory), against a standard server on the same machine. The bar: decode no slower than that server, with and without the drafter.
- **Refusals:** `is_varloop`, `--loops` (Dory's pass-count override), audio parts, and images before the vision slice, each with a test.

The tiny checkpoint `hf-internal-testing/tiny-inkling2-mini` at revision `ee242a48af34ff0729518f4ac28cfcb847e470ab` has 18 decoder layers, a loop core of layers 6 to 11, and no drafter: its config declares no `mtp_config` and its files ship no `mtp.safetensors`. It covers plain, resumed and concurrent decode. It cannot cover drafted-versus-plain, and with one core block it cannot cover the two-source message case. Draft parity therefore runs on the original checkpoint with its bundled `mtp.safetensors`, recording proposal and acceptance counts.

## Hardware

Each claimant states the chips they can run. The plan needs at least one sm_12x GPU (the model card names one DGX Spark, sm_121) and one Mac with 32 GB or more; an M5 and an earlier chip are both needed for M1.
The maintainers run device tests on a GB10, M5 and M3 Ultras, and a gfx1151.
The author of this plan can run one GB10 (DGX Spark, sm_121) and one M4 Max with 128 GiB, and has asked in #4274's discussion for H3, with C3 as the fallback; neither starts before the maintainer confirms.
That covers the CUDA slices on sm_121 and the earlier-chip side of the Metal slices. It leaves two gaps for other testers: an M5 for the tensor-unit prompt path, and any CUDA chip other than sm_121.

## Related

- #3476: Inkling 2 Small on CUDA, with the maintainer's answers on the family's shape and on the attention reference.
- #4265: Inkling 2 Mini on HIP.
- #3578, #4124 to #4128, #3612: Small's host package, kernels, reply split and EXL3 loader.
- #3723: a looped model on the qwen3_5 CUDA engine.

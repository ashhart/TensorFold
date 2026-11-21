# Inkling 2 Mini on CUDA and Metal: a plan in claimable slices

This is a proposal, not an agreed gate. Nothing here is claimed, and no slice should start before a maintainer's decision.
It splits the work into slices small enough for one contributor each, so that several people can take parts of it through ATC.
#4265 proposes the same model on HIP; the host slices below are shared with it and should have one owner.

## Decision requested

1. Is Inkling 2 Mini wanted on CUDA and on Metal, as one plan with #4265, sharing the host slices?
2. Which reference defines the routed experts? `hf_quant_config.json` asks for per-token activation quantization in front of the NVFP4 and MXFP8 experts, and CONTRIBUTING keeps the checkpoint's activation precision. The same question is open for Small on #3476 and for HIP on #4265; one answer would serve every backend.
3. Which CUDA chips come first: sm_12x alone, with the stored FP4 and FP8 arithmetic, or also older chips through expert tiles that decode to BF16 in registers?
4. Which checkpoint comes first on Metal: the original (BF16 dense weights, the stored experts, the drafter file), or `mlx-community/Inkling-2-Mini-4bit` as #4265 describes it (4-bit dense weights, the same experts, no drafter)?
5. Does the loop live in the `inkling2` family, or does it follow #3723's execution schedule (one weight copy, one cache slot per execution)?
6. Is the gate under "Proposed gate" the right one?
7. Once the plan is agreed, should each slice below become its own issue, so that it can be checked and claimed with ATC?

## What main has

On main `523d58cbe`:

- `zig/src/families/inkling2` is Inkling 2 Small's host package (#3578): config, weight names, layer plan, chat template and tokenizer tests. It accepts `inkling2` only, so Mini's `inkling2_loop` is refused.
- CUDA has Inkling 2's attention kernels under the kernel contract (#4125).
- Open for Small on CUDA: glue kernels (#4126), expert and vision kernels (#4127), drafter kernels (#4128), the reply split (#4124) and the EXL3 loader (#3612).
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
- That pull request dequantizes the NVFP4 and MXFP8 experts on load and marks this as temporary. So it runs the stored experts with float activations, not with the activation quantization `hf_quant_config.json` asks for. Decision 2 is whether that is the reference.
- llama.cpp merged the model on 2026-11-20 (ggml-org/llama.cpp#32694), so it is one candidate for the standard server on both backends. vLLM is another on NVIDIA.
- No speed for a DGX Spark or for Apple silicon was found published; the receipt's baseline has to be measured on the claimant's machine.

## Slices

Each slice is one pull request with its own tests. "Needs" names what must land or be decided first.

### Host, shared by CUDA, Metal and HIP

| Slice | What | Needs |
|---|---|---|
| H1 | `families/inkling2` accepts `inkling2_loop`: parses `loop_config`, refuses `is_varloop`, builds the visit plan (layer, pass, cache slot, message sources) and counts drafter taps by visit. Small's tests pass unchanged. This is #4265's first slice. | Decision 1 and 5 |
| H2 | Names for the controller and cross-pass tensors, and each layer's expert precision read from Mini's headers, with fixtures from the published config and a tiny checkpoint. | H1 |
| H3 | The loop's arithmetic written down: where each value rounds and the order of every sum for the gates and the cross-pass message, a float64 model, and single-change probes that turn the host tests red. | H1 |
| H4 | The drafter's host side: `mtp.safetensors` names and shapes against Mini's header, and its four taps mapped to visits and cache slots. | H1 |

### CUDA

| Slice | What | Needs |
|---|---|---|
| C1 | Mini's shapes on Small's kernels: served shapes, tiles and limit records for hidden 2,304 and experts of 1,152. | #4126 to #4128 |
| C2 | The loop kernels (gates and cross-pass message attention) under the kernel contract. | H3 |
| C3 | The routed experts as stored, NVFP4 and MXFP8, on sm_12x. A second pull request covers older chips if decision 3 asks for them. | Decision 2 and 3 |
| C4 | The engine at one rank, text only: 48 KV and conv slots in the pool and in prompt-cache snapshots, the lane adapter, admission and the family registration. Images are refused until C6. | H2, C1 to C3 |
| C5 | DSpark: the checkpoint's drafter on #4128's kernels, with taps by visit, through the shared lane rounds. Drafted output equals plain decode, and decode with it is no slower than without. | C4, #4128 |
| C6 | Images on #4127's vision kernels at Mini's tower shape. | C4, #4127 |
| C7 | The served receipt, the recipe page and the family's optimization ledger cells. | C4 |

### Metal

| Slice | What | Needs |
|---|---|---|
| M1 | Sliding and global attention with the sink, the relative bias, log scaling and the L2 clips, on the arithmetic reference #4125 follows. A prompt path for M5 and one for earlier chips. | Decision 1 |
| M2 | Glue kernels: embedding, the block attention residual, the short convolutions (static and input-dependent), the router, soft-capped SwiGLU and the pick. | Decision 1 |
| M3 | The routed experts as stored: NVFP4 and MXFP8 decoded in registers, FP32 sums. | Decision 2 |
| M4 | The loop kernels under the kernel contract. | H3 |
| M5 | The engine, text only: 48 slots, the lane adapter, admission by memory class and the family registration. | H2, M1 to M4, decision 4 |
| M6 | DSpark: the drafter's kernels under the kernel contract, then the drafter through the shared lane rounds. MiniMax M3 and Kimi K3 have DSpark drafters on Metal to adapt. Same checks as C5. | M5 |
| M7 | Images: the vision tower and projector. | M5 |
| M8 | The served receipt, the recipe page and the ledger cells. | M5 |

H1 to H3, M1 and M2 can start as soon as the plan is agreed. C1 waits on Small's stack, and C3 and M3 wait on decision 2.

## DSpark

DSpark support is part of the plan on both backends, not an extra: H4, C5 and M6 carry it, and C7 and M8 fill the family's DSpark cell in the optimization ledger.
The engine slices C4 and M5 land first without it, so plain decode exists as the reference the drafter is checked against.
If decision 4 picks the 4-bit checkpoint, which ships no drafter file, M6 needs the original's `mtp.safetensors` beside it; whether that pairing is admitted is part of that decision.

## Proposed gate

- **Kernels:** each backend's full kernel-contract `SUMMARY` with every planted control red, on the chips named in the pull request.
- **Same-engine exactness,** on the tiny checkpoint and on the released one, greedy and sampled: drafted equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. Seams are checked at every pass boundary, for KV, conv state and final logits.
- **Fidelity:** teacher-forced histories against the reference decision 2 names, with every top-token difference and its margin recorded. #4265 names `hf-internal-testing/tiny-inkling2-mini` as a token-equal fixture.
- **Speed:** a served receipt as CONTRIBUTING describes (decode with and without the drafter, cold first token at one and four streams, cold 2k, 8k, 32k and 64k prompts, peak memory), against a standard server on the same machine. The threshold is the maintainer's to set; `zig/gate.json` has none.
- **Refusals:** `is_varloop`, audio parts, and images before the vision slice, each with a test.

## Hardware

Each claimant states the chips they can run. The plan needs at least one sm_12x GPU (the model card names one DGX Spark, sm_121) and one Mac with 32 GB or more; an M5 and an earlier chip are both needed for M1.
The author of this plan can run one GB10 (DGX Spark, sm_121) and one M4 Max with 128 GiB, and would take one group of slices after the decision; which group is not chosen.
That covers the CUDA slices on sm_121 and the earlier-chip side of the Metal slices. It leaves two gaps for other testers: an M5 for the tensor-unit prompt path, and any CUDA chip other than sm_121.

## Related

- #3476: Inkling 2 Small on CUDA, with the maintainer's answers on the family's shape and on the attention reference.
- #4265: Inkling 2 Mini on HIP.
- #3578, #4124 to #4128, #3612: Small's host package, kernels, reply split and EXL3 loader.
- #3723: a looped model on the qwen3_5 CUDA engine.

# Native GLM arithmetic before the M5

Single-rank GLM on Metal selects its arithmetic from the device, independently of `GLM_PROMPT` and `GLM_CHUNK`. On M1-M4, standalone dense projections use the shared `simd_qmm_bits` tree for supported g64 members. Other group sizes and narrow tails use the same row operator in both callers. A scalar group that exceeds the reflected thread limit also uses the tile.

KDA windows, chunks and accepted-prefix replay share the chunk prep/scan/post encoder. The recurrence state stays FP32 at block boundaries. Each block reads the original convolution history and writes the final tail from the whole supplied projection span. Window scratch is included in both admission and actual allocation. The changed shader source and arithmetic policy enter the saved learned-state identity.

MLA keeps its existing per-head projection, index selection and attention operators at bounded window heights, with chunk-sized standalone projections. Both shared and routed MoE experts keep the existing window operators. These choices remove the remaining changes in operator selection between prompt modes. They add window work to a large prompt; prompt cost and decode speed must be measured on the device.

This is an intentional arithmetic change for single-Mac GLM before the M5. Plain and drafted replies can change from either earlier prompt mode, for text and image turns and every affected checkpoint width. No particular old token string is promised. The intended invariant is identical tokens for the same rendered prompt and settings at every supported cut, draft width, resume and concurrent schedule. M5, paired-Mac and CUDA GLM keep their previous arithmetic. Other families keep their projection arithmetic; the shared SIMD input-pitch argument defaults to the previous contiguous addressing.

## Checks

`zig build test-glm-ordered-host` executes the real projection and KDA encoders against a recording boundary. It checks strides, split members, row fallback, state carry, convolution history and counted admission. It also tests the independent FP64 recurrence oracle with a planted wrong-output control. The regular GLM window and kernel-contract host tests remain required.

On a pre-M5 Mac with the Metal toolchain, build the model-free device gate with:

```sh
zig build build-glm-ordered-check-host -Dcpu=apple_m1 -j1
```

Run `zig-out/bin/tf-glm-ordered-check` under the shared GPU lock. It exercises the actual encoder, dense classes against FP64, every window height 1..17 and representative 33/2,048-row cuts, strides and padding. It checks partial-acceptance replay and compares all KDA output, recurrent-state and convolution-tail bytes across cuts, with independent FP64 scan and post references at their recorded input points. It does not qualify the nonlinear prep, full checkpoint or served engine by itself. Run the ordinary kernel contract for all six `simd_qmm_bits` widths and compile the new resource records for M1/M2; zero compiled limits mean unchecked, not measured support.

Before landing, obtain full-checkpoint receipts on the reporter's original pinned 2k/8k/32k ids for windows, 2,048-row chunks and 16,384-row chunks. Compare the complete token arrays and corresponding margins. `tf-glm-run`'s `GLM_VS` only prints differences, so compare its `GLM_OUT` JSON files explicitly. Qualify independent-reference fidelity, plain/drafted, resumed/fresh, concurrent/solo and cancellation, and show that decode is no slower. Keep parent/candidate cold-prompt samples and peak resident memory at the same settings. Host compilation and recording tests do not establish those device results.

The full GLM-5.3-Flash 4-bit checkpoint does not fit a 256 GB Mac whole. `GLM_LAYERS=40` can exercise the actual weight prefix and identify its first differing operator, state seam or resource refusal. Its logits and replies describe a truncated network, even when its real head is retained. That run cannot qualify full-model tokens, fidelity, MTP acceptance, resume/concurrency, memory fit or end-to-end speed. Full qualification requires a fitting device at the agreed load limit; do not raise a limit as part of this arithmetic change.

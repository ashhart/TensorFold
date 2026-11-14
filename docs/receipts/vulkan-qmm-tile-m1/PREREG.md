# qmm_q4_tile pre-registration (written 2026-11-03 before any GPU run of this change)

## Question

Does storing each projection's weights, scales and biases in 32-column tiles (a load-time repack) make a Qwen3.5-2B run shorter
on the 13-inch M1, with byte-identical logits? The kernel is `qmm_q4_vec4` with only its address formulas changed, so every sum is the same.

## Arms (two ELFs of `tf-qwen35-model`, ReleaseSafe, aarch64)

- old: the vec4 build, commit `e742e023`, ELF `683fcf46...`.
- new: the same plus the tiled image: weights of every projection with n a multiple of 32 are repacked at load; the lm head is a second,
  tiled copy of the embedding table (the embedding keeps the row layout). Projections with n = 16 keep the vec4 image.

## Gates (every process, both arms)

1. Exit 0 and a complete receipt (the house receipt check).
2. Generated ids equal the pinned ids.
3. Logits SHA-256 of every process equals one digest per case, the same digest the earlier vec4 runs gave (unicode-24 `e5447d9a...`,
   document-256 `147a2d2e...`, document-1024 `7f9d8e65...`).
4. Before the pairs: the stage 1 probe on the GPU runs the tiled image on every tileable projection case and compares bytes to the default image.

## Metric and design

Run 2 duration from the probe's stderr milestones (prefill plus 63 decode steps plus the per-step logits readback), as before.
Three pairs per case, order old,new,new,old,old,new. Cases: unicode-24, document-256, document-1024.

## Decision rule

Speedup: median pair gain of at least 3 percent and all three pair gains positive and all gates hold. Otherwise reported as no speedup and not proposed.

## Forecast (a prediction; I do not know the answer)

Gain of 20 to 45 percent of run 2 on unicode-24 and similar on the documents. Why not more: qmm is about 63 percent of the run net of the submit
cost, the multi-row result showed qmm is not simply load-count limited, and the hardware cache may already hold part of the 1 KiB stride.
I give 20 percent to a gain under 3 percent, and 10 percent to a regression.

## Not claimed

Other chips, other models, other quantizations, the lm head on devices whose storage range is below the table size (they keep the row layout).
The software-device pass is not GPU evidence. Memory: the tied lm head adds about 320 MB of device memory.

## Addendum, written after the final pairs ran (2026-11-03)

The first contract run on the GPU refused the served shape 1x16x64: a registered kernel must serve every shape the contract serves, and the build
above kept the n = 16 projections on the vec4 image. The design changed before the final runs: the repack pads the last tile with zero columns, so the tiled
image serves every n, the n = 16 projections are tiled too, and the arms read "old: the vec4 build, new: the padded tile build". The gates, metric and
decision rule did not change. This addendum was written after the final pairs; the three earlier pairs on the unpadded build are kept in the receipts as an
earlier measurement and are not the result. The forecast above was for the unpadded build; it was not revised.

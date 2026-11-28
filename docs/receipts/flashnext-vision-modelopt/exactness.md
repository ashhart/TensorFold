# flashnext-vision-modelopt: exactness

The branch `ac20700003` on main `077998f7d9`, one image (a photograph, sha256 `596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb`, 240512 bytes), greedy requests (temperature 0, seed 1234, thinking off), fresh server an arm, `--parallel 8 --context 32768 --kv-cache bf16`, `--vision` where the leg is vision.

## The three contracts, with vision on

| contract | result |
|---|---|
| drafted output equals `"draft": false` (eight prompts, `served_exactness.py drafts`) | 8 of 8 equal, 0 unequal, replies of 256 tokens |
| concurrent equals solo (`bench_concurrent.py --alone --serial`, levels 1, 2, 4, temperatures 0 and 1.0) | 52 equal, 0 unequal, 0 failed requests |
| resumed equals fresh (`served_exactness.py turns`, then `fresh` on a `--prompt-cache-gib 0` boot) | 7 of 7 equal on both a plain and a drafted fresh request, 0 problems |

The turns recorded cached prompt tokens [13, 16, 16, 13, 24, 18, 3406] and every fresh request reported 0 cached tokens; the equality basis is token_ids. Problems: {}.

## The image reply, and the pairs it must equal

- the reply, once: `A close-up portrait captures a tabby cat with striking green eyes, gazing intently at the viewer against a softly blurred background.`
- prompt tokens 147, completion tokens 28, drafted 38, accepted 20, wall 1.141 s, text sha256 `f4f12377f25c105d841b7f23a84cb516504b928edd1fe14aac5268ccdbd280f0`
- the same request twice in one boot: equal=True by token_sha256
- the same request with `"draft": false`: equal=True by token_sha256
- the same request streamed, twice: equal=True by token_sha256
- the same request streamed again: equal=True by token_sha256
- the same request on the second boot (no prompt cache): equal=True by token_sha256

## The default path, text only (vision off)

- drafted equals `"draft": false` on the eight prompts: 32 of 32 equal across the four boots (main two, branch two), 0 unequal
- the greedy hashes: 8 of 8 prompts carry the same token ids on all four boots

## The checkpoint's tower: main refuses, the branch serves

- main, `--vision` on the FP4 checkpoint: [2026-11-17T00:30:32.520841803Z error: model.visual.blocks.0.attn.qkv.weight: f8_e4m3 { 3456, 1152 }, expected bf16 { 3456, 1152 }](the log line, quoted)
- the branch: the same request served, reply above; the tower's line in the boot is in the README's memory table.

## The floating-point tower (CHECKPOINT=int4)

- both images were refused before any weight was read, `error: flashnext: config.json's quantization_config.quant_method differs from the supported Flash Next CUDA layout`, so this leg has no result on this engine.

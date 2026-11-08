# XPU Qwen3.8-27B decode attention grid order: receipts

The change dispatches `attn_decode` as `(rows, heads, slices)` so the tree rows of one key slice share it in L2.
Per-work-group arithmetic is unchanged; outputs are bit-identical to `1a5644892`. Gate agreed in #2543.

## Machine and builds

- One Intel Arc Pro B70 (BMG-G31, PCI id 0xe223, 32 GB), 230 W cap, Ubuntu 26.04, kernel 7.0, xe, oneAPI 2026.1.
- Zig 0.17.0, ocloc 26.05.37020. Both binaries carry one local build change that drops
  `-cl-intel-128-GRF-per-thread` from the Gemma 4 prompt-rows kernels, which this ocloc rejects.
- Checkpoint `TensorFold/Qwen3.8-27B-MLX-4bit` @ `22d8d538`, drafter `z-lab/Qwen3.8-27B-DFlash2`, bf16 KV.
- Serving: `--parallel 1 --context 139264 --prompt-cache-gib 0 --no-thinking`, greedy, 512 tokens, one stream.

## Files

- `contract-attention.txt`: the attention lines, the `SUMMARY` and the exit status of `tf-xpu-contract-run --device 0`.
- `token-sha.json`: server `token_sha` per prompt length for main and branch, drafted and `"draft": false`. The kept
  server logs cover 46 of the 54 timed requests: main drafted has n=2 below 130K tokens and n=1 at 130K tokens,
  while its speed medians use n=3 and n=2. The other 8 main requests have text hashes only.
- `decode-depth.json`: streamed decode tokens/s, TTFT and prefill per cell (code and prose prompts).
- `bench-openai.json`: `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`, two alternated rounds.

## Results

Contract: 10 kernels pass gates 1 to 4, 6 controls red as planted, 0 limit problems; the attention pair is
bit-equal to `1a5644892` at 20 shapes (tree widths 1, 2, 4, 8, 12, 16; 33 to 32,770 visible keys).

Token hashes are equal across main and branch, drafted and `"draft": false`, at 52 to 130,856 prompt tokens.

| Prompt tokens (code) | main tok/s | branch tok/s |
|---|---|---|
| 68 | 128.2 | 128.1 |
| 7,976 | 85.1 | 90.1 |
| 32,552 | 37.3 | 45.9 |
| 130,856 | 11.4 | 15.1 |

## Not run

`XPU_KV_INT8=1`, `--parallel` above 1, prompt-cache resumes, sampled decoding, chips other than BMG-G31.
`xpu-qwen27-tests` means-test fails identically on main and branch (worst error 5.12e-5 prompt, 1.86e-5 decode
against a 2e-6 bound, then exit 139); seam and close pass on both.

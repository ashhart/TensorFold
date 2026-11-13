# XPU Qwen3.8-27B prefix decode attention (`XPU_ATTN_PREF`): receipts

An opt-in, off by default (#2543). With `XPU_ATTN_PREF=256` a round reads the committed prefix once: `attn_pref`
scores chunks of 256 keys on bf16 DPAS for 8 (row, head) queries at a time, `attn_tail` keeps `attn_decode`'s
arithmetic for a row's last keys, and `attn_merge2` merges both in key order. Unset, decode runs `attn_decode` and
`attn_merge` as before and token hashes equal main's.

## Machine and builds

- Two Intel Arc Pro B70 (BMG-G31, PCI id 0xe223, 32 GB): #1 at a 230 W cap, #2 at 180 W. Ubuntu 26.04, kernel 7.0,
  xe, oneAPI 2026.1. One server at a time on a card.
- Zig 0.17.0, ocloc 26.05.37020. Every binary carries one local build change that drops
  `-cl-intel-128-GRF-per-thread` from the Gemma 4 prompt-rows kernels, which this ocloc rejects.
- `measured`: `a1d896a88` (#2595 before its two contract commits) with this change's kernels inside `decode.cl`.
  `head`: this pull request, where the same kernels are in `decode_prefix.cl` and `attn_decode` keeps its signature.
- Checkpoint `TensorFold/Qwen3.8-27B-MLX-4bit` @ `22d8d538`, drafter `z-lab/Qwen3.8-27B-DFlash2`, bf16 K/V.
- Serving: `--parallel 1 --context 139264 --prompt-cache-gib 0 --no-thinking`, greedy, 512 tokens, one stream.
- Prompts: the code and prose prompts of #2595's receipt (68 / 7,976 / 32,552 / 130,856 and 52 / 7,963 / 32,540 /
  130,846 prompt tokens).

## Files

- `requests.json`: every timed request: path, switch, build, card, prompt tokens, decode tokens/s, TTFT, the
  server's `token_sha`, rounds and accepted drafts, and the sha256 of the answer text.
- `battery.json`: 329 graded items, default path against `XPU_ATTN_PREF=256`, with both text hashes an item.

## Results

Default path (switch unset), `measured` and `head` builds: `token_sha` equals #2595's at every prompt length,
drafted and `"draft": false`.

With the switch, drafted equals `"draft": false` equals a second drafted run at every cell (0, 8K, 32K twice;
131K once), and on the 211 battery items run with `"draft": false` (211 of 211; the other 118 were not run that
way). A second card returns the first card's drafted text on 329 of 329 items. `head` returns the `measured` build's hashes on both paths (8K and
32K, B70 #2).

Decode tokens/s on B70 #1, drafted, code / prose, default against the switch in one session:

| Prompt tokens (code) | default | `XPU_ATTN_PREF=256` | ratio |
|---|---|---|---|
| 68 | 127.4 / 100.7 | 127.4 / 100.7 | 1.00 |
| 7,976 | 90.0 / 71.6 | 114.7 / 91.5 | 1.27 |
| 32,552 | 45.9 / 33.9 | 90.1 / 66.5 | 1.96 |
| 130,856 | 15.1 / 13.6 | 50.7 / 45.6 | 3.35 |

`"draft": false`: 30.4 / 23.7 / 13.0 (default) and 30.4 / 27.3 / 18.8 (switch) at 8K / 32K / 131K. The 68-token row
of the default column is the switch run's: no row there reaches a prefix chunk, so both run the same kernels.
On B70 #2 at 180 W with the `head` build, while the other card served a battery: 85.0 / 67.8 to 109.0 / 87.0 at
8K and 42.3 / 31.2 to 84.5 / 62.5 at 32K.

The switch changes tokens. Against the default path the answer text differs at 4 of 6 depth cells (8K prose, 32K
code and prose, 131K prose) and on 67 of 329 battery items. `XPU_ATTN_PREF=512` against 256 differs again at 3
of 4 cells, so the text follows the order of the sums.

Battery passes, default / switch: HumanEval even 78 / 78 of 82, odd 76 / 76 of 82, knowledge 40 / 42 of 55, JSON
schema 55 / 55 of 55, instruction following 33 / 33 of 55. Two items gained, none lost.

## Not run

The kernel contract for `attn_pref`, `attn_tail` and `attn_merge2` (FP64 reference, planted control, records);
teacher-forced error and margins; `tools/bench_openai.py`; `XPU_KV_INT8=1` and tree rounds of several streams
(both keep the default path under the switch); prompt-cache resumes; sampled decoding; `--parallel` above 1;
three repeats at 131K; chips other than BMG-G31.

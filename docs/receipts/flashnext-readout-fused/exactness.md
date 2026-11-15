Main `41a72fd257` and the branch on it at `ee13555c8e`: unset, and `TF_FLASHNEXT_READOUT_FUSED=1`. Each arm booted
fresh with the prompt cache on and again with `--prompt-cache-gib 0`; `--parallel 4`, bf16 KV, a 131,072-token window.

| check | measure | main | branch, unset | branch, fused |
|---|---|---|---|---|
| 1 | bench_concurrent --serial, code+chat, greedy | 2/2 equal (token_sha256, [256] tok) | 2/2 equal | 2/2 equal |
| 1 | bench_concurrent --serial, code+chat, temperature 1.0 | 8/8 equal (token_sha256, [256] tok) | 8/8 equal | 8/8 equal |
| 1 | bench_concurrent --serial, six golden, greedy | 6/6 equal (token_sha256, [256] tok) | 6/6 equal | 6/6 equal |
| 1 | served_exactness drafts, 8 prompts, cache on | 8/8 equal (token_ids, [256] tok) | 8/8 equal | 8/8 equal |
| 1 | served_exactness drafts, 8 prompts, cache off | 8/8 equal (token_ids, [256] tok) | 8/8 equal | 8/8 equal |
| 1 | serve_check drafted == serial (greedy, t0.8), cache off | 2/2 ok | 2/2 ok | 2/2 ok |
| 2 | served_exactness turns (on) against fresh (off), 5 chats | 7/7 equal; resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens | 7/7 equal | 7/7 equal |
| 2 | bench_cache repeated/concurrent == first, cache on | 4 equal, 0 unequal; cached [0, 2570, 2570, 2570] | 4 equal, 0 unequal | 4 equal, 0 unequal |
| 3 | bench_concurrent --alone, identical code+chat, 1/2/4, greedy | 42/42 equal, 0 failed | 42/42 equal, 0 failed | 42/42 equal, 0 failed |
| 3 | bench_concurrent --alone, code+chat, 1/2/4, temperature 1.0 | 42/42 equal, 0 failed | 42/42 equal, 0 failed | 42/42 equal, 0 failed |
| 3 | bench_concurrent --alone, six golden distinct, 1/2/4/6, greedy | 13/13 equal, 0 failed | 13/13 equal, 0 failed | 13/13 equal, 0 failed |
| 3 | serve_check together/among-decoders/twice, cache off | 12/12 ok | 12/12 ok | 12/12 ok |

| check | main | branch, unset | branch, fused |
|---|---|---|---|
| drafted == draft:false | PASS | PASS | PASS |
| resumed == fresh | PASS | PASS | PASS |
| concurrent == solo | PASS | PASS | PASS |

Replies against main's (same request, same boot kind), unset and fused alike: concurrent alone+serial hashes 16/16;
golden alone+serial hashes 6/6; drafts with the cache on, plain token ids 8/8; with it off 8/8; turns, first token ids
7/7; resumed token ids 7/7. serve_check: 15 ok, 0 FAIL in every arm.

The golden and drafts prompts are short, so their passes stay under 768 rows and take the chain in every arm. The
fused tiles run on these:

- Three long prompts (about 1.5k, 5k and 15k tokens), `bench_concurrent --alone --serial` at 1 and 4 streams, greedy,
  128 tokens: 20 reply hashes an arm, 3 distinct, the same 20 in main, unset and fused, with the cache on and off.
- The same long prompts with prompt passes of 2,048 rows and of 3,072 rows (`TF_FLASHNEXT_CHUNK`), 1 and 4 streams:
  main and fused record the same 20 hashes each, 0 failed requests.
- With passes of 1,000 rows, every request fails on main and on the branch alike (`CudaFailed`, after
  `cuMemsetD8Async: CUDA_ERROR_INVALID_VALUE`), so that size gives no comparison.
- `bench_cache.py --require-cold`'s 2,571-token prompt in the six speed arms: the same 4 reply hashes in all six.
- The speed arms' own replies, greedy and at temperature 1.0 with seed 1234, code and chat, 1, 2 and 4 streams: all 168
  `token_sha256` values of each invocation equal M1's in every arm.

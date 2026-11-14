Main `06bcb989` (image `um15`) and the branch on it (`um15x`): unset, and `mxfp8`. Each arm booted fresh with the prompt
cache on and again with `--prompt-cache-gib 0`; `--parallel 4`, bf16 KV, a 131,072-token window.

| check | measure | main | branch, unset | branch, mxfp8 |
|---|---|---|---|---|
| 1 | bench_concurrent --serial, code+chat, greedy | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, code+chat, temperature 1.0 | 8/8 equal (token_sha256, [256] tok) | 8/8 equal (token_sha256, [256] tok) | 8/8 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, six golden, greedy | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) |
| 1 | served_exactness drafts, 8 prompts, cache on | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | served_exactness drafts, 8 prompts, cache off | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | serve_check drafted == serial (greedy, t0.8), cache off | 2/2 ok | 2/2 ok | 2/2 ok |
| 2 | served_exactness turns (on) against fresh (off), 5 chats | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] |
| 2 | bench_cache repeated/concurrent == first, cache on | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570] | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570] | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570] |
| 3 | bench_concurrent --alone, identical code+chat, 1/2/4, greedy | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests |
| 3 | bench_concurrent --alone, code+chat, 1/2/4, temperature 1.0 | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests |
| 3 | bench_concurrent --alone, six golden distinct, 1/2/4/6, greedy | 13/13 equal, 0 failed requests | 13/13 equal, 0 failed requests | 13/13 equal, 0 failed requests |
| 3 | serve_check together/among-decoders/twice, cache off | 12/12 ok | 12/12 ok | 12/12 ok |

| check | main | branch, unset | branch, mxfp8 |
|---|---|---|---|
| drafted == draft:false | PASS | PASS | PASS |
| resumed == fresh | PASS | PASS | PASS |
| concurrent == solo | PASS | PASS | PASS |

Replies against main's (same request, same boot kind):
- branch, unset: concurrent alone+serial hashes 16/16; golden alone+serial hashes 6/6; drafts with the cache on, plain
  token ids 8/8; with it off 8/8; turns, first token ids 7/7; resumed token ids 7/7
- branch, mxfp8: concurrent alone+serial hashes 16/16; golden alone+serial hashes 6/6; drafts with the cache on, plain
  token ids 8/8; with it off 8/8; turns, first token ids 7/7; resumed token ids 7/7

The speed arms' own replies (`cells.csv`'s runs, greedy and at temperature 1.0 with seeds 1234, 2234 and 3234): every
`token_sha256` the bench files record, 672 an arm, is equal to main's first arm in the second main arm, both unset
arms, the blank arm (U2, the container's environment holding `TF_FLASHNEXT_MTP_WEIGHTS=` empty) and both `mxfp8` arms.

serve_check lines (cache off; its resumed == fresh line compares two cold requests on that server):
- main: 15 ok, 0 FAIL []
- branch, unset: 15 ok, 0 FAIL []
- branch, mxfp8: 15 ok, 0 FAIL []

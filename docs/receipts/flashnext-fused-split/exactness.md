# Exactness: main, and the branch with TF_FLASHNEXT_FUSED_SPLIT set to 1, unset and blank

Each arm boots once with the prompt cache on (concurrent, golden, drafts, turns, cache) and once with
`--prompt-cache-gib 0` (fresh, drafts, serve_check). Check 1 is drafted against `"draft": false`, check 2 resumed against
fresh, check 3 concurrent against solo.

| check | measure | main | fs-on | fs-unset | fs-blank |
|---|---|---|---|---|---|
| 1 | bench_concurrent --serial, code+chat, greedy | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, code+chat, temperature 1.0 | 16/16 equal (token_sha256, [256] tok) | 16/16 equal (token_sha256, [256] tok) | 16/16 equal (token_sha256, [256] tok) | 16/16 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, six golden, greedy | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) |
| 1 | served_exactness drafts, 8 prompts, cache on | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | served_exactness drafts, 8 prompts, cache off | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | serve_check drafted == serial (greedy, t0.8), cache off | 2/2 ok | 2/2 ok | 2/2 ok | 2/2 ok |
| 2 | served_exactness turns (on) against fresh (off), 5 chats | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] |
| 2 | bench_cache repeated/concurrent == first, cache on | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} |
| 3 | bench_concurrent --alone, identical code+chat, 1/2/4, greedy | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests |
| 3 | bench_concurrent --alone, code+chat, 1/2/4, temperature 1.0 | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests | 90/90 equal, 0 failed requests |
| 3 | bench_concurrent --alone, six golden distinct, 1/2/4/6, greedy | 21/21 equal, 0 failed requests | 21/21 equal, 0 failed requests | 21/21 equal, 0 failed requests | 21/21 equal, 0 failed requests |
| 3 | serve_check together/among-decoders/twice, cache off | 12/12 ok | 12/12 ok | 12/12 ok | 12/12 ok |

| check | main | fs-on | fs-unset | fs-blank |
|---|---|---|---|---|
| drafted == draft:false | PASS | PASS | PASS | PASS |
| resumed == fresh | PASS | PASS | PASS | PASS |
| concurrent == solo | PASS | PASS | PASS | PASS |

Replies against main's (same request, same boot kind):
- fs-on: conc alone+serial hashes 32/32; gold alone+serial hashes 8/8; d_on plain token ids 8/8; d_off plain token ids 8/8; turn first token ids 7/7; turn resumed token ids 7/7
- fs-unset: conc alone+serial hashes 32/32; gold alone+serial hashes 8/8; d_on plain token ids 8/8; d_off plain token ids 8/8; turn first token ids 7/7; turn resumed token ids 7/7
- fs-blank: conc alone+serial hashes 32/32; gold alone+serial hashes 8/8; d_on plain token ids 8/8; d_off plain token ids 8/8; turn first token ids 7/7; turn resumed token ids 7/7

serve_check lines (cache off; its resumed == fresh line compares two cold requests on that server):
- main: 15 ok, 0 FAIL []
- fs-on: 15 ok, 0 FAIL []
- fs-unset: 15 ok, 0 FAIL []
- fs-blank: 15 ok, 0 FAIL []

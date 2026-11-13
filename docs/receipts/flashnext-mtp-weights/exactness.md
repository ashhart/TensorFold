| check | measure | main | PR, unset | PR, mxfp8 |
|---|---|---|---|---|
| 1 | bench_concurrent --serial, code+chat, greedy | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) | 2/2 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, code+chat, temperature 1.0 | 8/8 equal (token_sha256, [256] tok) | 8/8 equal (token_sha256, [256] tok) | 8/8 equal (token_sha256, [256] tok) |
| 1 | bench_concurrent --serial, six golden, greedy | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) | 6/6 equal (token_sha256, [256] tok) |
| 1 | served_exactness drafts, 8 prompts, cache on | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | served_exactness drafts, 8 prompts, cache off | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] | 8/8 equal (token_ids, [256] tok); accepted 13-237 a reply, plain drafted ['0'] |
| 1 | serve_check drafted == serial (greedy, t0.8), cache off | 2/2 ok | 2/2 ok | 2/2 ok |
| 2 | served_exactness turns (on) against fresh (off), 5 chats | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] | 7/7 equal (token_ids, [14, 43, 25, 70, 256, 120, 12] tok); resumed from [13, 16, 16, 13, 24, 18, 3406] of [112, 84, 99, 208, 306, 25, 3459] prompt tokens; fresh cached [0] |
| 2 | bench_cache repeated/concurrent == first, cache on | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} | 4 equal, 0 unequal, 0 unverified; states {'cold': 1, 'reused': 3}; cached [0, 2570, 2570, 2570]; {'qualified': True} |
| 3 | bench_concurrent --alone, identical code+chat, 1/2/4, greedy | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests |
| 3 | bench_concurrent --alone, code+chat, 1/2/4, temperature 1.0 | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests | 42/42 equal, 0 failed requests |
| 3 | bench_concurrent --alone, six golden distinct, 1/2/4/6, greedy | 13/13 equal, 0 failed requests | 13/13 equal, 0 failed requests | 13/13 equal, 0 failed requests |
| 3 | serve_check together/among-decoders/twice, cache off | 12/12 ok | 12/12 ok | 12/12 ok |

| check | main | PR, unset | PR, mxfp8 |
|---|---|---|---|
| drafted == draft:false | PASS | PASS | PASS |
| resumed == fresh | PASS | PASS | PASS |
| concurrent == solo | PASS | PASS | PASS |

Replies against main's (same request, same boot kind):
- pr-default: conc alone+serial hashes 16/16; gold alone+serial hashes 6/6; d_on plain token ids 8/8; d_off plain token ids 8/8; turn first token ids 7/7; turn resumed token ids 7/7
- pr-mxfp8: conc alone+serial hashes 16/16; gold alone+serial hashes 6/6; d_on plain token ids 8/8; d_off plain token ids 8/8; turn first token ids 7/7; turn resumed token ids 7/7

serve_check lines (cache off; its resumed == fresh line compares two cold requests on that server):
- main: 15 ok, 0 FAIL []
- pr-default: 15 ok, 0 FAIL []
- pr-mxfp8: 15 ok, 0 FAIL []

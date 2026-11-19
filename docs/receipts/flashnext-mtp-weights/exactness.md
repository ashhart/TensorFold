# The three exactness contracts, every state

drafted equals `"draft": false` (served_exactness drafts, cache on and off), resumed equals fresh (turns on against fresh off), concurrent equals solo (bench_concurrent --alone over pass A's greedy cells), the golden prompts' own alone/serial checks, and every reply's token_sha against main's.

| state | drafted on | drafted off | resumed == fresh | golden alone | golden serial | concurrent == solo (greedy) | concurrent == solo (sampled) |
|---|---|---|---|---|---|---|---|
| M | 8/8 | 8/8 | 7/7 | 5/0 | 8/0 | 252 equal, 0 unequal | 252 equal, 0 unequal |
| U | 8/8 | 8/8 | 7/7 | 5/0 | 8/0 | 252 equal, 0 unequal | 252 equal, 0 unequal |
| X | 8/8 | 8/8 | 7/7 | 5/0 | 8/0 | 252 equal, 0 unequal | 252 equal, 0 unequal |

Replies against main's, same request, same boot kind:

- U: pass A every reply's token_sha 552 equal, 0 unequal; coldhash 8 equal, 0 unequal
- X: pass A every reply's token_sha 552 equal, 0 unequal; coldhash 8 equal, 0 unequal

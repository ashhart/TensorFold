# Exactness on the served server

Each contract on both arms with the prompt cache on and off (`bench_concurrent.py --alone --serial` at temperatures 1.0 and 0 and on the six golden prompts; `served_exactness.py drafts`, `turns`, and `fresh` against the cache-on turns), then the greedy replies compared across the arms.

- B cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- B cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- B cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B cache off fresh: 7 of 7 resumed replies equal the fresh ones
- M cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- M cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- M cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- M cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B vs M: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal

- the cold cells: 8 of 8 at four slots and 4 of 4 at one slot carry the same token ids and reply text on all four boots (`coldhash.py`, the same prompt bytes on every arm)
- drafts off: 42 replies equal, 0 unequal

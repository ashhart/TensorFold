| contract | M | B0 | B1 |
|---|---|---|---|
| concurrent equals solo (pass A, 1 to 8 streams, greedy) | 252/252 equal | 252/252 equal | 252/252 equal |
| drafted equals `"draft": false` (pass A, 1 to 8 streams, greedy) | 20/20 equal | 20/20 equal | 20/20 equal |
| resumed equals fresh (pass C, 7 chats, cache on against cache off) | 7/7 equal | 7/7 equal | 7/7 equal |
| cancel and reset leave the next request valid (pass C, the cancel route) | 3/3 contracts | 3/3 contracts | 3/3 contracts |
| every greedy reply equals main's (token_sha over pass A's cells), B0 | 528/528 equal | | |
| every greedy reply equals main's (token_sha over pass A's cells), B1 | 528/528 equal | | |

- B0 cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- B0 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B0 cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- B0 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B0 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B1 cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- B1 cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B1 cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- B1 cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B1 cache off fresh: 7 of 7 resumed replies equal the fresh ones
- M cache on golden (6 prompts, 1 and 4 streams): concurrent equals solo 5/0; drafted equals `"draft": false` 8/0
- M cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache on turns: 7 chats, resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- M cache on cancel: 3 of 3 contracts, reply after the cancel True equal to the reference (token_sha256)
- M cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B0: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal
- B1 against B0: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal

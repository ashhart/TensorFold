These JSON files are synthetic recorded-format unit fixtures, authored for the
harness. They are not captures from any model or hardware. `recording.json` has
eight greedy tokens and a BF16-near tie at generated index 2. `profiles.json`
contains three placeholder endpoints with configured caps 1, 4 and 8.

`equal.json` records identical serial/drafted token streams. `divergent.json`
changes only the depth-4 token at index 2 from 12 to 99, reversing the two
synthetic contenders in that run's raw-logit capture. Both margins are within
one BF16 spacing, but exact token equality must fail.

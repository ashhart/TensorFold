# Verified copies on CUDA lane hosts

Gemma 4, MiniMax M3, Kolibri-1 and Spark use the lane core's existing root and context copies with a four-token match, as Nemotron does. The existing verification widths and model arithmetic are unchanged. Copies come from request history, then the ordinary target verifies them; a mismatched token stops the accepted prefix.

The serving drafts switch enables these copies by default. `--no-drafts` disables them, and a request with `"draft": false` takes its plain path. Copies do not need a neural drafter. When a drafter is loaded, the core can prefer a matched copy before asking the head for more drafts. Head depth and target verification capacity remain separate; MiniMax's seven-draft DSpark proposal does not limit the target's variable-width copy windows.

`zig build test-family-copies-host` runs the real family facts through the lane planner with an instrumented target: disabled copies, the four-token entry threshold, greedy/seeded plain equality, shared streams, resumed prefixes, a rejected copy, cancellation and the next request. These host fixtures do not establish whole-checkpoint exactness or speed. Qualify each checkpoint with its existing CUDA contract and matched plain/drafted device cells before claiming a gain.

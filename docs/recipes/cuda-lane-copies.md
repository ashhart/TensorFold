# Verified copies on CUDA lane hosts

Gemma 4, MiniMax M3, Kolibri-1, Spark and Dory can opt into the lane core's root and context copies with a four-token match, as Nemotron does. The existing verification widths and model arithmetic are unchanged. Copies come from request history, then the ordinary target verifies them; a mismatched token stops the accepted prefix.

The default retains each family's existing suffix-copy policy and eight-token match. Set the matching load-time option to 1 to try root and context copies: `TF_GEMMA4_COPIES`, `TF_MINIMAX_COPIES`, `TF_KOLIBRI_COPIES`, `TF_SPARK_COPIES` or `TF_DORY_COPIES`. Unset or 0 keeps the existing policy. Other values refuse startup by name. Keep MiniMax at its default policy until a runnable CUDA checkpoint has a qualifying cell.

`--no-drafts` and a request with `"draft": false` take plain rounds. Copies do not need a neural drafter. When a drafter is loaded, the core can prefer a matched copy before asking the head for more drafts. Head depth and target verification capacity remain separate; MiniMax's seven-draft DSpark proposal does not limit the target's variable-width copy windows.

`zig build test-family-copies-host` runs the real family facts through the lane planner with an instrumented target. It pins main's default and drafts-off facts, the four-token entry threshold, greedy/seeded plain equality, shared streams, resumed prefixes, a rejected copy, cancellation and the next request. These host fixtures do not establish whole-checkpoint exactness or speed.

Qualify each checkpoint with its CUDA contract and matched main, policy-off, policy-on and plain controls. Retain all repeats in fresh OFF/ON/ON/OFF runs for chat and code, greedy and seeded, one and eight streams. Every token must agree, and no cell may be slower than main or copies off beyond the repeat spread. A missing or slower cell keeps the existing default policy. No device gain is claimed for these opt-in policies.

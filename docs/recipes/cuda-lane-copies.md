# Verified copies on CUDA lane hosts

Gemma 4, MiniMax M3, Kolibri-1, Spark and Dory can opt into the lane core's root and context copies with a four-token match, as Nemotron does. The existing verification widths and model arithmetic are unchanged. Copies come from request history, then the ordinary target verifies them; a mismatched token stops the accepted prefix.

The default retains each family's existing suffix-copy policy and eight-token match. Set the matching load-time option to 1 to try root and context copies: `TF_GEMMA4_COPIES`, `TF_MINIMAX_COPIES`, `TF_KOLIBRI_COPIES`, `TF_SPARK_COPIES` or `TF_DORY_COPIES`. Unset or 0 keeps the existing policy. Other values refuse startup by name. Keep MiniMax at its default policy until a runnable CUDA checkpoint has a qualifying cell.

`--no-drafts` and a request with `"draft": false` take plain rounds. Copies do not need a neural drafter. When a drafter is loaded, the core can prefer a matched copy before asking the head for more drafts. Head depth and target verification capacity remain separate; MiniMax's seven-draft DSpark proposal does not limit the target's variable-width copy windows.

`zig build test-family-copies-host` runs the real family facts through the lane planner with an instrumented target. It pins main's default and drafts-off facts, the four-token entry threshold, greedy/seeded plain equality, shared streams, resumed prefixes, a rejected copy, cancellation and the next request. These host fixtures do not establish whole-checkpoint exactness or speed.

Qualify each checkpoint with its CUDA contract and matched main, policy-off, policy-on and plain controls. Retain all repeats in fresh OFF/ON/ON/OFF runs for chat and code, greedy and seeded, one and eight streams. Every token must agree, and no cell may be slower than main or copies off beyond the repeat spread. A missing or slower cell keeps the existing default policy. No device gain is claimed for these opt-in policies.

## GLM on CUDA

`TF_GLM_COPIES=1` on rank 0 opts GLM into the same four-match root/context copies, within its 32-row verify window.
Unset or 0 keeps GLM's current suffix policy and head policy. Other values refuse by name
before weights load. `--no-drafts` and request-local `"draft": false` stay plain. Followers execute rank 0's existing
commands, including partial keeps and zero-depth head absorption; no CUDA kernel or target arithmetic changes.

Copies use the shared cost allocator. With MTP they are priced by the windows timed at load
([window prices](glm-5.3-flash-cuda.md#window-prices)); with DFlash2, or when the scratch context is too small to time,
they use GLM's existing two-GB10 EXL3 target-window and shared-row tables as seeds. The stream's own accepted/rejected copies update their landing estimate, and an expensive offer can receive zero rows.
With the opt-in, `--mtp-confidence` caps the head's probability prefix using the shared confidence rule while copied
rows remain costed. The allocator can choose plain instead of a head proposal. The default confidence path is unchanged.
No row-cost learner is added here; live learned target costs belong to the lane core (#2111).

`TF_GLM_COPY_SHARE=1`, with `TF_GLM_COPIES=1`, lets a copy in a shared round take the lone ramp too (`2w+1` after a
copy kept whole, back to the first copy's three tokens after a miss), capped at its stream's share of the round's 32
rows less its pending row: 15 tokens at two streams, 7 at four, and three from seven streams on. Without it every copy
in a shared round stays three tokens wide, and only a stream alone ramps. The cap leaves the other streams' rows open;
the prices still decide what fills them. Unset or 0 keeps the facts above, and 1 without `TF_GLM_COPIES=1` refuses at
the open. With `TF_GLM_SHARED=0` each stream's copy keeps three tokens, and Recall keeps its width.

Load timing covers short contexts and one stream, and the seed tables came from an older DFlash2 profile, so neither
establishes total round cost at a new checkpoint, long context, rank layout or high concurrency. Keep the default off until exact TP=2 whole-model replies and measured cost
pass: baseline/OFF/ON/ON/OFF, greedy and seeded chat/code, one and eight streams, short/32K/64K context, cold prompts
and separately labelled resumed turns. Retain exact rendered prompt IDs, emitted token arrays/hashes, actual verify,
partial-keep and head-absorption times by rows, complete round/request times, proposals/acceptance/zero grants, load
seconds and resident/pinned/device memory. Compare `"draft": false` before speed, and retain every repeat.

Spark's copy experiment lost 3.8 to 22.7% in long-context cells on 3 November, so acceptance alone is insufficient.
A slower or missing GLM cell keeps the default policy. No GLM speed gain or device qualification is claimed here.
`zig build test-glm-copies-host` exercises real GLM facts through the shared planner with an instrumented target,
including expensive and repeatedly rejected offers reduced to plain; it does not execute the checkpoint on CUDA.

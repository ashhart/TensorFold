# Grammar timing

Lane rounds measure grammar work on the host when a constrained stream builds its window. The decision log records `grammar_window` with the step and stream id, then `grammar_round` with that step's sum. Both carry `windows`, `fills`, `window_ns` and `fill_ns`. A stream's round total resets on its first window in a new step.

`fill_ns` covers the rules fill, optional completion-budget mask and dead-end check. `window_ns` covers the whole grammar traversal, including fills, rollback and allocations. It excludes reading held GPU drafts and preparing the window's token and parent arrays. Errors retain work already timed. The windows include their fills, so their durations must not be added.

Finished replies expose lifetime sums as `tensorfold.grammar_windows`, `grammar_fills`, `grammar_window_seconds` and `grammar_fill_seconds`. These fields are absent when no grammar window was measured. They use a separate host clock; planner costs and GPU timings are unchanged. Unconstrained windows perform no timing reads or metric allocations. A clock without a host timing hook leaves these metrics absent.

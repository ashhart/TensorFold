//! Counters reported by an engine when one reply ends.

pub const Stats = struct {
    rounds: u64 = 0,
    grammar: ?@import("lanes").grammar_time.Sum = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// The GPU-busy seconds of the forward passes the reply ran in, where profiling times them (TF_Q27_PROFILE).
    gpu_seconds: ?f64 = null,
    /// The folded-memory version the request routed against, fixed when it was opened (null: no memory served).
    memory_version: ?u64 = null,
    /// Seconds from engine submission to first admission; an engine that admits immediately leaves zero.
    queued_seconds: f64 = 0,
    /// Why the prompt resumed no further (#3686); null: the engine's store does not say.
    prompt_cache: ?@import("prompt_cache_why.zig").Why = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

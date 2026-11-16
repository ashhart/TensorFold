//! Counters reported by an engine when one reply ends.

/// What a remote prefill cost one reply (#1594): the router's last verdict on its prompt, and a pull's phases.
pub const RemotePhases = struct {
    /// What a remote prefill cost one reply (#1594): the route's last verdict on its prompt, and a pull's phases.
    route: []const u8 = "",
    /// at the gate, behind another pull or the breaker's trial
    queue_s: f64 = 0,
    /// the producer's prefill: the trigger's exchange
    trigger_s: f64 = 0,
    /// the export's transfer, landing included
    pull_s: f64 = 0,
    /// of the pull, the time spent landing parts in the Mac's caches
    land_s: f64 = 0,
    /// the Mac's own pass after the pull (the prefill_seconds the reply reports)
    tail_s: f64 = 0,
    /// payload bytes the pull moved
    bytes: u64 = 0,
};

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
    /// Seconds from engine submission to first admission; an engine that admits immediately leaves zero.
    queued_seconds: f64 = 0,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
    /// The restored prefix's tokens when it stands on another machine's state (#1594 step 2b): outside exactness.
    remote_prefix: u32 = 0,
    /// remote_prefix again when this reply pulled its prefix over the link; 0 when a kept remote state was restored.
    pulled_prefix: u32 = 0,
    /// The route's verdict and, for a pulled prefix, its phases.
    remote: RemotePhases = .{},
};

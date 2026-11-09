//! The lone driver and its handoff callbacks, shared by native family adapters and the lane host.
const lanes = @import("lanes");

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    plain: bool = false, // it drives streams with drafts off too
    logprobs: bool = false, // it gives each token its logprob row too
    structures: bool = false, // it masks each window's rows by the stream's grammar too
    images: bool = true, // it takes image requests too (false: they stay in the lane core)
    // thinking budgets above this and loop guards: it hands the stream over before a forced close (0: never)
    min_budget: u32 = 0,
    resumes: bool = false, // it takes the lane core's stream back between rounds (LoneHooks.resumed); else only opens
    /// Prefills (or resumes) `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core.
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
    resumed: bool = false, // the stream comes from the lane core between rounds: no prompt pass, its drafts the core's
    // After a completed nonfinal prompt call, true transfers the unfinished pass to the shared fill queue.
    prefill_yield: ?*const fn (ctx: *anyopaque) bool = null,
};

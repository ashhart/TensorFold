//! Engine readiness and status values, read without entering a device context.

/// Thread-safe backend readiness; no GPU context or blocking device work is allowed on HTTP threads.
pub const HealthSource = struct {
    ctx: *anyopaque,
    read: *const fn (*anyopaque) bool,
};

pub const Status = struct {
    healthy: bool = true,
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// Filling prompts' total tokens and those still to prefill (null: no prompt is going in).
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    /// A prompt pass is running whose backend reports no `fill_at`: the totals stay null rather than guessed.
    fill_unknown: bool = false,
    /// Changes whenever the filling set does: a reader pairs its drain samples with one fill.
    fill_run: u64 = 0,
};

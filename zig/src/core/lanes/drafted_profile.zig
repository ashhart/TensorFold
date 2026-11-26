//! The drafted wrapper's counters and round-profile state: each gate read once, the phases fused per round.
const std = @import("std");

pub const DraftStats = struct {
    rounds: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    draft_rounds: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // windows with held > 0
    offered: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kept: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    held_max: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kept_max: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};
pub var draft_stats: DraftStats = .{};
pub var draft_stats_env: ?bool = null; // the TF_DRAFT_STATS gate, read once (the hot path must not getenv per round)
pub var draft_debug_env: ?bool = null; // the TF_DRAFT_DEBUG gate, read once (the hot path must not getenv per round)

/// TF_ROUND_PROFILE: one line a drafted round, its phases in host-wall ms - held (the drafts'
/// DtoH waiting on the last hold), fwd (the verify window), keep, absorb, hold (the drafter's
/// launch). A phase a round skips rides into the next line; the wall's remainder = the lane
/// host's own time. The depth buckets accumulate across the run (kept/offers by held depth).
pub var profile_env: ?bool = null;
pub var profile_round: u64 = 0;
pub var profile: struct {
    held: f64 = 0,
    fwd: f64 = 0,
    keep: f64 = 0,
    absorb: f64 = 0,
    hold: f64 = 0,
    rows: usize = 0,
    offers: usize = 0,
    kept: usize = 0,
} = .{};
pub var profile_depths: [9]u64 = @splat(0);
pub var profile_offers_at: [9]u64 = @splat(0);

pub fn profileOn() bool {
    if (profile_env == null) profile_env = std.c.getenv("TF_ROUND_PROFILE") != null;
    return profile_env.?;
}

pub fn profileNow() f64 {
    const c = @import("sys").c;
    var t: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}

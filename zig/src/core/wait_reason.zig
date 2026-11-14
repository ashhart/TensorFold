//! Fixed reasons: a queued request's wait (the first three are server/wait_reason.zig's codes), refusals, failures.
const std = @import("std");

/// Why a queued request has not started: the hold its last admission attempt met, one reason a queued request.
pub const Wait = enum {
    /// No lane is free, or nothing more specific held it yet.
    engine_busy,
    /// The device is stale: a put-back has not reached it, so no reply starts.
    engine_unavailable,
    /// Memory pressure holds admission to the streams already running.
    memory_pressure,
    /// The family's planner waits for KV pool pages that running requests hold.
    kv_pool,
    /// The family's planner waits for the memory floor (TF_DSV41_FLOOR_*); never counted as `memory_pressure`.
    memory_floor,
    /// The family's planner passes over a long prompt while others prefill (TF_DSV41_PREFILL_CONCURRENT).
    prefill_concurrent,
    /// It waits for a cut another prompt is filling (lane_host_prefix.zig).
    prefix_cut,
    /// It waits for its disk state's read ahead (lane_host_fetch.zig).
    state_read,
    /// Its caches' bytes do not fit the host's KV ledger now (lane_host_ledger.zig).
    ledger_bytes,
    /// The family's own pool has no room for it now (Info.admit).
    pool_room,
};

/// A refusal by the code its client sees: the capacity codes, the engine's 400, the admission flags' 429s.
pub const Refusal = enum {
    engine_busy,
    engine_unavailable,
    memory_pressure,
    invalid_request,
    max_inflight_exceeded,
    max_inflight_per_key_exceeded,
    admission_capacity_exceeded,
    requests_per_second_exceeded,
};

/// Ended failed with its own words, no refusal code: the host's four, its planner's refusal, a stale device at submit.
pub const Failure = enum { backend_prompt, backend_round, lane_memory_alone, prompt_too_large, planner_refused, engine_stale };

pub const waits = std.enums.values(Wait).len;
pub const refusals = std.enums.values(Refusal).len;
pub const failures = std.enums.values(Failure).len;

test "wait reason ledger: the client codes lead the set, each tag at its own index" {
    try std.testing.expectEqual(@as(usize, 10), waits);
    for (std.enums.values(Wait), 0..) |r, i| try std.testing.expectEqual(i, @backingInt(r));
    try std.testing.expectEqualStrings("memory_pressure", @tagName(std.enums.values(Wait)[2]));
}

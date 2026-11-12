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

/// What a client sees as a refusal: the three capacity codes and the engine's 400 for a request it cannot take.
pub const Refusal = enum { engine_busy, engine_unavailable, memory_pressure, invalid_request };

/// A request the host took and then ended failed, with its own words: not a refusal a client can retry by code.
pub const Failure = enum { backend_prompt, backend_round, lane_memory_alone, prompt_too_large };

pub const waits = std.enums.values(Wait).len;
pub const refusals = std.enums.values(Refusal).len;
pub const failures = std.enums.values(Failure).len;

test "wait reason ledger: the client codes lead the set, each tag at its own index" {
    try std.testing.expectEqual(@as(usize, 10), waits);
    for (std.enums.values(Wait), 0..) |r, i| try std.testing.expectEqual(i, @backingInt(r));
    try std.testing.expectEqualStrings("memory_pressure", @tagName(std.enums.values(Wait)[2]));
}

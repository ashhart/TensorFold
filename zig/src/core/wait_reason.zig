//! One fixed code for every way a request waits or is refused: bounded values, no request data (issue #2135).
const std = @import("std");

/// Why a queued request is not running. Every lane host wait path maps to exactly one.
pub const Wait = enum(u8) {
    engine_busy, // every lane holds an admitted request
    prefix_cut, // another prompt is filling a cut this request would resume
    state_read, // its disk state is still being read ahead
    memory_pressure, // admission holds under the memory governor
    engine_unavailable, // a Living Weights put-back has not reached the device
    ledger_bytes, // the byte ledger promises no room yet
    pool_room, // the family's pool has no room for its caches
    kv_pool, // DeepSeek-V4.1's round planner waits for KV pool pages
    memory_floor, // DeepSeek-V4.1's round planner waits for the memory floor
    prefill_concurrent, // DeepSeek-V4.1's round planner waits behind a long prompt prefilling
};

/// Why a request was refused. Every lane host refusal path maps to exactly one.
pub const Refusal = enum(u8) {
    engine_unavailable, // the host is closing, unhealthy or waiting for a device put-back
    engine_busy, // no job or queue slot could be allocated
    invalid_request, // the family's request checks refused it before admission
    memory_pressure, // critical memory pressure (a 503)
    prompt_too_large, // the prompt is past the whole ledger alone
    lane_memory_alone, // no lane memory for its caches with nothing else running
    backend_prompt, // its prompt pass failed
    backend_round, // a shared round failed
    restore_refused, // its kept state could not be restored: it prefills from 0
};

pub const wait_count = @typeInfo(Wait).@"enum".field_names.len;
pub const refusal_count = @typeInfo(Refusal).@"enum".field_names.len;

/// The code a wait reports: its tag name, fixed at compile time.
pub fn waitCode(w: Wait) []const u8 {
    return @tagName(w);
}

/// The code a refusal reports: its tag name, fixed at compile time.
pub fn refusalCode(r: Refusal) []const u8 {
    return switch (r) {
        .engine_busy => waitCode(.engine_busy),
        .engine_unavailable => waitCode(.engine_unavailable),
        .memory_pressure => waitCode(.memory_pressure),
        else => @tagName(r),
    };
}

/// Queued requests by wait reason; the sum is the host's `waiting` count.
pub const Waits = struct {
    counts: [wait_count]u32 = @splat(0),

    pub fn note(w: *Waits, r: Wait) void {
        w.counts[@backingInt(r)] +|= 1;
    }

    pub fn count(w: Waits, r: Wait) u32 {
        return w.counts[@backingInt(r)];
    }

    pub fn total(w: Waits) u32 {
        var n: u32 = 0;
        for (w.counts) |c| n +|= c;
        return n;
    }
};

/// Refused requests by refusal reason; a counter, so it survives a host detach.
pub const Refusals = struct {
    counts: [refusal_count]u64 = @splat(0),

    pub fn note(r: *Refusals, why: Refusal) void {
        r.counts[@backingInt(why)] +|= 1;
    }

    pub fn count(r: Refusals, why: Refusal) u64 {
        return r.counts[@backingInt(why)];
    }

    pub fn add(a: Refusals, b: Refusals) Refusals {
        var out = a;
        for (b.counts, 0..) |c, i| out.counts[i] +|= c;
        return out;
    }
};

test "every wait and refusal code is a fixed tag name within the label bound" {
    var waits: [wait_count][]const u8 = undefined;
    var refusals: [refusal_count][]const u8 = undefined;
    for (std.enums.values(Wait), 0..) |w, i| waits[i] = waitCode(w);
    for (std.enums.values(Refusal), 0..) |r, i| refusals[i] = refusalCode(r);
    for ([_][]const []const u8{ &waits, &refusals }) |codes| for (codes, 0..) |code, i| {
        try std.testing.expect(code.len > 0 and code.len <= 32);
        for (code) |c| try std.testing.expect((c >= 'a' and c <= 'z') or c == '_');
        for (codes[0..i]) |prior| try std.testing.expect(!std.mem.eql(u8, prior, code));
    };
}

test "wait counts add by reason and keep their total" {
    var w: Waits = .{};
    w.note(.engine_busy);
    w.note(.engine_busy);
    w.note(.prefix_cut);
    try std.testing.expectEqual(@as(u32, 2), w.count(.engine_busy));
    try std.testing.expectEqual(@as(u32, 1), w.count(.prefix_cut));
    try std.testing.expectEqual(@as(u32, 0), w.count(.kv_pool));
    try std.testing.expectEqual(@as(u32, 3), w.total());
}

test "refusal counts add across a host detach" {
    var a: Refusals = .{};
    a.note(.backend_prompt);
    var b: Refusals = .{};
    b.note(.backend_prompt);
    b.note(.memory_pressure);
    const sum = Refusals.add(a, b);
    try std.testing.expectEqual(@as(u64, 2), sum.count(.backend_prompt));
    try std.testing.expectEqual(@as(u64, 1), sum.count(.memory_pressure));
}

test "capacity codes share the same names for queued and refused requests" {
    try std.testing.expectEqualStrings("engine_busy", waitCode(.engine_busy));
    try std.testing.expectEqualStrings("engine_unavailable", waitCode(.engine_unavailable));
    try std.testing.expectEqualStrings("memory_pressure", waitCode(.memory_pressure));
    try std.testing.expectEqualStrings(waitCode(.engine_busy), refusalCode(.engine_busy));
    try std.testing.expectEqualStrings(waitCode(.engine_unavailable), refusalCode(.engine_unavailable));
    try std.testing.expectEqualStrings(waitCode(.memory_pressure), refusalCode(.memory_pressure));
}

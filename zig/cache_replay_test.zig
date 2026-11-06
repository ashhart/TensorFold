//! Measurement controls: the adapter must match native main, and eviction must preserve borrowed and live state.
const std = @import("std");
const pc = @import("cache");
const run = @import("cache_replay_run.zig");
const trace = @import("cache_replay_trace.zig");
const model = @import("cache_replay_model.zig");
const t = std.testing;
fn put(r: *run.Replay, tokens: []const u32, at: u32) !void {
    var live: model.State = .{};
    live.advance(tokens, at);
    try t.expect(r.store.keep(tokens, at, &live, &.{}, &.{}));
}
fn sequence(comptime n: usize, seed: u32) [n]u32 {
    var result: [n]u32 = undefined;
    for (&result, 0..) |*v, i| v.* = seed + @as(u32, @intCast(i));
    return result;
}
test "cache replay: reservation adapter matches native main on every request at all three budgets" {
    for (std.enums.values(trace.Shape)) |shape| {
        var requests = try trace.make(t.allocator, shape, 61, 7);
        defer requests.deinit();
        for ([_]u64{ 4, 8, 16 }) |gib| {
            var native: run.Replay = undefined;
            native.init(t.allocator, .main, gib << 30, .{});
            defer native.deinit();
            var adapter: run.Replay = undefined;
            adapter.init(t.allocator, .main, gib << 30, .{});
            adapter.adapter_main = true;
            defer adapter.deinit();
            for (requests.requests) |req| {
                const actual = try native.request(req);
                const substituted = try adapter.request(req);
                try t.expectEqualDeep(actual, substituted);
                try t.expectEqualDeep(native.store.counts, adapter.store.counts);
            }
        }
    }
}
test "cache replay: real main prefers a superseded continuation over an older unrelated session" {
    var replay: run.Replay = undefined;
    replay.init(t.allocator, .main, 4 << 30, .{});
    defer replay.deinit();
    const other = sequence(32, 1000);
    const conversation = sequence(64, 1);
    const incoming = sequence(64, 2000);
    try put(&replay, &other, 16);
    try put(&replay, conversation[0..32], 16);
    try put(&replay, conversation[0..48], 32);
    try t.expectEqual(@as(?usize, 1), run.policy.victim(&replay.store, .main, null, &incoming));
    try t.expectEqual(@as(?usize, 0), run.policy.victim(&replay.store, .recency, null, &incoming));
    try t.expectEqual(@as(?usize, 0), run.policy.victim(&replay.store, .saved_seconds_per_byte, null, &incoming));
}
test "cache replay: borrowed snapshots cannot be evicted and dropping copies cannot change live state" {
    const old = sequence(32, 1);
    const incoming = sequence(64, 2000);
    for (std.enums.values(run.policy.Policy)) |p| {
        var replay: run.Replay = undefined;
        replay.init(t.allocator, p, 4 << 30, .{});
        defer replay.deinit();
        try put(&replay, &old, 16);
        const entry = replay.store.entries.items[0];
        entry.borrowers += 1;
        var live: model.State = .{};
        try replay.store.family.vtable.restore(replay.store.family.ptr, &live, entry.saved);
        const original = live;
        replay.store.budget = replay.family.model.bytes(32);
        try t.expect(run.policy.victim(&replay.store, p, null, &incoming) == null);
        run.policy.reserve(&replay.store, p, null, &incoming, &.{32});
        try t.expect(!run.policy.roomForKeep(&replay.store, &incoming, 32));
        try t.expectEqual(@as(usize, 1), replay.store.entries.items.len);
        try t.expectEqualDeep(original, live);
        replay.store.release(entry);
        run.policy.reserve(&replay.store, p, null, &incoming, &.{32});
        try t.expectEqual(@as(usize, 0), replay.store.entries.items.len);
        try t.expectEqualDeep(original, live);
        try t.expectEqual(@as(u64, 0), replay.family.live);
    }
}
test "cache replay: explicit restoring entry is protected under every selector" {
    const tokens = sequence(64, 1);
    for (std.enums.values(run.policy.Policy)) |p| {
        var replay: run.Replay = undefined;
        replay.init(t.allocator, p, 4 << 30, .{});
        defer replay.deinit();
        try put(&replay, &tokens, 32);
        const e = replay.store.entries.items[0];
        try t.expect(run.policy.victim(&replay.store, p, e, &tokens) == null);
    }
}
test "cache replay: zero budget recomputes every token and ample budget reuses each session's prefix" {
    var requests = try trace.make(t.allocator, .steady, 3, 7);
    defer requests.deinit();
    for (std.enums.values(run.policy.Policy)) |p| {
        var empty: run.Replay = undefined;
        empty.init(t.allocator, p, 0, .{});
        defer empty.deinit();
        var ample: run.Replay = undefined;
        ample.init(t.allocator, p, 1 << 50, .{});
        defer ample.deinit();
        var previous: [100]u32 = @splat(0);
        for (requests.requests) |req| {
            const cold = try empty.request(req);
            const hot = try ample.request(req);
            try t.expectEqual(req.tokens.len, cold.recompute);
            try t.expectEqual(previous[req.session], hot.from);
            try t.expectEqual(req.tokens.len, @as(u64, hot.from) + hot.recompute);
            previous[req.session] = req.history;
        }
        try t.expectEqual(@as(usize, 0), empty.store.entries.items.len);
    }
}
test "cache replay: fixed traces contain growing long histories, unique one-offs and an idle return" {
    for (std.enums.values(trace.Shape)) |shape| {
        var requests = try trace.make(t.allocator, shape, 61, 7);
        defer requests.deinit();
        var again = try trace.make(t.allocator, shape, 61, 7);
        defer again.deinit();
        try t.expectEqual(requests.digest, again.digest);
        var last_round: u32 = 0;
        var returning = false;
        var reached_80k = false;
        for (requests.requests) |req| switch (req.kind) {
            .agent => {
                try t.expect(req.history >= 20000 and req.history <= 80000);
                reached_80k = reached_80k or req.history == 80000;
            },
            .one_off => try t.expect(req.history >= 256 and req.history <= 8192),
            .idle_seed => last_round = req.round,
            .idle_return => {
                if (!returning) try t.expectEqual(@as(u32, 44), req.round - last_round);
                returning = true;
            },
        };
        try t.expect(returning and reached_80k);
    }
}
test "cache replay: cost model matches the declared geometry and quantiles use nearest rank" {
    const bytes: model.Model = .{};
    try t.expectEqual(@as(u64, 153944064), model.Model.recurrent_bytes);
    try t.expectEqual(@as(u64, 65536), model.Model.kv_bytes_per_token);
    const raw: model.Model = .{ .padded = false };
    try t.expectEqual(model.Model.recurrent_bytes + 80000 * 65536, raw.bytes(80000));
    try t.expectEqual(@as(u64, 0), bytes.bytes(80000) % (32 << 20));
    try t.expect(bytes.bytes(80000) >= raw.bytes(80000));
    var s: run.Stats = .{};
    defer s.deinit(t.allocator);
    try s.costs.appendSlice(t.allocator, &.{ 80, 0, 40, 20 });
    try t.expectEqual(@as(u32, 20), s.percentile(50));
    try t.expectEqual(@as(u32, 80), s.percentile(95));
    try t.expectEqual(@as(u32, 80), s.percentile(100));
}

test "cache replay: density can evict an existing planned mark before a later keep needs its room" {
    var requests = try trace.make(t.allocator, .steady, 61, 7);
    defer requests.deinit();
    var replay: run.Replay = undefined;
    replay.init(t.allocator, .saved_seconds_per_byte, 8 << 30, .{});
    defer replay.deinit();
    for (requests.requests) |req| _ = try replay.request(req);
    try t.expect(replay.store.held <= replay.store.budget);
}

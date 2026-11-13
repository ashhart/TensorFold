//! Fresh-seed native equivalence checks use the frozen traces without changing the candidate.
const std = @import("std");
const traces = @import("cache_agent_trace.zig");
const run = @import("cache_replay_run.zig");
const seeds = @import("cache_confirm_replay.zig").seeds;
const t = std.testing;
const pc = @import("cache");
const model = @import("cache_replay_model.zig");

fn same(a: *run.Replay, b: *run.Replay) !void {
    try t.expectEqualDeep(a.store.counts, b.store.counts);
    try t.expectEqualDeep(a.family.dropped.items, b.family.dropped.items);
    a.family.dropped.clearRetainingCapacity();
    b.family.dropped.clearRetainingCapacity();
    try t.expectEqual(a.store.entries.items.len, b.store.entries.items.len);
    for (a.store.entries.items, b.store.entries.items) |x, y| {
        try t.expectEqualSlices(u32, x.tokens, y.tokens);
        try t.expectEqualSlices(u32, x.last, y.last);
        const xs: *const model.State = @ptrCast(@alignCast(x.saved));
        const ys: *const model.State = @ptrCast(@alignCast(y.saved));
        try t.expectEqualDeep(xs.*, ys.*);
        inline for (.{ "at", "bytes", "born", "used", "shared", "rewind", "branched", "intermediate", "expired", "borrowers" }) |field|
            try t.expectEqual(@field(x, field), @field(y, field));
    }
}

fn compare(candidate: bool) !void {
    for ([_]u64{ 7, seeds[0], seeds[1] }) |seed| for (std.enums.values(traces.Shape)) |shape| {
        var trace = try traces.makeSeeded(t.allocator, shape, seed);
        defer trace.deinit();
        for (traces.budgets) |budget| {
            var native: run.Replay = undefined;
            native.init(t.allocator, .main, budget, .{});
            native.family.record_drops = true;
            defer {
                native.deinit();
                native.family.dropped.deinit(t.allocator);
            }
            if (candidate) native.store.eviction = .decayed_resume;
            var adapter: run.Replay = undefined;
            adapter.init(t.allocator, if (candidate) .decayed_resume else .main, budget, .{});
            adapter.adapter_main = !candidate;
            adapter.family.record_drops = true;
            defer {
                adapter.deinit();
                adapter.family.dropped.deinit(t.allocator);
            }
            for (trace.requests) |request| {
                try t.expectEqualDeep(try native.request(request), try adapter.request(request));
                try same(&native, &adapter);
            }
        }
    };
}

test "cache confirm: default choices and owned snapshot bytes equal frozen main on seeds 7, 11 and 23" {
    try compare(false);
}

test "cache confirm: native opt-in choices equal the frozen E25 candidate on seeds 7, 11 and 23" {
    try compare(true);
}

test "cache confirm: environment selects only the explicit policy and invalid values are refused by name" {
    const env = @import("sys").env;
    const settings = @import("sys").environment;
    const saved = if (std.c.getenv("TF_CACHE_EVICT")) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (saved) |v| {
            _ = env.setenv("TF_CACHE_EVICT", v, 1);
            t.allocator.free(v);
        } else _ = env.unsetenv("TF_CACHE_EVICT");
    }
    for ([_][:0]const u8{ "", "main", "decayed_resume" }) |value| {
        try t.expectEqual(@as(c_int, 0), env.setenv("TF_CACHE_EVICT", value, 1));
        var family: model.Family = .{ .gpa = t.allocator, .model = .{} };
        var store = pc.Store.init(t.allocator, family.snapshots(), .{}, 4 << 30);
        defer store.deinit();
        try t.expectEqual(if (std.mem.eql(u8, value, "decayed_resume")) pc.Eviction.decayed_resume else .main, store.eviction);
        try settings.check("TF_CACHE_EVICT", value);
    }
    try t.expectEqual(@as(c_int, 0), env.unsetenv("TF_CACHE_EVICT"));
    var family: model.Family = .{ .gpa = t.allocator, .model = .{} };
    var store = pc.Store.init(t.allocator, family.snapshots(), .{}, 4 << 30);
    defer store.deinit();
    try t.expectEqual(pc.Eviction.main, store.eviction);
    for ([_][]const u8{ "1", "frequency_probation", "decayed", "DECAYED_RESUME", " decayed_resume" }) |bad|
        try t.expectError(error.InvalidEnvironment, settings.check("TF_CACHE_EVICT", bad));
}

test "cache confirm: native opt-in preserves borrowers reservations skips keep-others and clear boundaries" {
    var r: run.Replay = undefined;
    r.init(t.allocator, .main, 4 << 30, .{});
    defer r.deinit();
    r.store.eviction = .decayed_resume;
    var a: [64]u32 = undefined;
    var b: [64]u32 = undefined;
    var incoming: [64]u32 = undefined;
    for (&a, &b, &incoming, 0..) |*x, *y, *z, i| {
        x.* = @intCast(i + 1);
        y.* = @intCast(i + 1000);
        z.* = @intCast(i + 2000);
    }
    for ([_][]const u32{ &a, &b }) |tokens| {
        var state: model.State = .{};
        state.advance(tokens, 32);
        try t.expect(r.store.keep(tokens, 32, &state, &.{}, &.{}));
    }
    const first = r.store.entries.items[0];
    const second = r.store.entries.items[1];
    first.borrowers += 1;
    const Guard = struct {
        fn needed(ctx: *anyopaque, entry: *const pc.Entry) bool {
            return @intFromPtr(ctx) == @intFromPtr(entry);
        }
    };
    r.store.reservation_guard = .{ .ctx = second, .needed = Guard.needed, .exact = true };
    try t.expect(r.store.victimBut(null, &incoming, 64) == null);
    r.store.reservation_guard = null;
    try t.expect(r.store.victimBut(second, &incoming, 64) == null);
    r.store.release(first);
    r.store.keep_others = true;
    try t.expect(pc.others.victim(&r.store, null, &incoming, 64, r.family.model.bytes(32), 1) == null);
    r.store.resume_carry = .{ .at = 32, .n = 32, .key = pc.Store.sharedKey(a[0..32]), .resumes = 2 };
    r.store.clear();
    try t.expectEqual(@as(u2, 0), r.store.resume_carry.resumes);
    try t.expectEqual(@as(u64, 0), r.family.live);
}

//! Semantic controls for bounded resume preference and turnover, separate from the frozen measurement gate.
const std = @import("std");
const run = @import("cache_replay_run.zig");
const model = @import("cache_replay_model.zig");
const traces = @import("cache_agent_trace.zig");
const t = std.testing;

fn sequence(comptime n: usize, seed: u32) [n]u32 {
    var result: [n]u32 = undefined;
    for (&result, 0..) |*v, i| v.* = seed + @as(u32, @intCast(i));
    return result;
}

fn put(r: *run.Replay, tokens: []const u32, at: u32) !void {
    var live: model.State = .{};
    live.advance(tokens, at);
    try t.expect(r.store.keep(tokens, at, &live, &.{}, &.{}));
}

test "cache decay: recently resumed session survives a newer probation state but old popularity is bounded" {
    var r: run.Replay = undefined;
    r.init(t.allocator, .decayed_resume, 4 << 30, .{});
    defer r.deinit();
    const old = sequence(64, 1);
    const fresh = sequence(64, 1000);
    const incoming = sequence(64, 2000);
    try put(&r, &old, 16);
    try put(&r, &fresh, 16);
    r.store.entries.items[0].used = 10;
    r.store.entries.items[0].seen.hits = 2;
    r.store.entries.items[1].used = 20;
    r.store.clock = 22;
    try t.expectEqual(@as(?usize, 0), run.policy.victim(&r.store, .main, null, &incoming));
    try t.expectEqual(@as(?usize, 1), run.policy.victim(&r.store, .decayed_resume, null, &incoming));
    r.store.entries.items[0].seen.hits = 100;
    r.store.entries.items[1].used = 30;
    r.store.clock = 32;
    try t.expectEqual(@as(?usize, 0), run.policy.victim(&r.store, .decayed_resume, null, &incoming));
}

test "cache decay: redundant continuations precede unique states even when repeatedly resumed" {
    var r: run.Replay = undefined;
    r.init(t.allocator, .decayed_resume, 4 << 30, .{});
    defer r.deinit();
    const other = sequence(64, 1000);
    const path = sequence(64, 1);
    const incoming = sequence(64, 2000);
    try put(&r, &other, 16);
    try put(&r, path[0..32], 16);
    try put(&r, path[0..48], 32);
    r.store.entries.items[1].seen.hits = 100;
    try t.expectEqual(@as(?usize, 1), run.policy.victim(&r.store, .decayed_resume, null, &incoming));
    r.store.links_lost = true;
    try t.expectEqual(@as(?usize, 1), run.policy.victim(&r.store, .decayed_resume, null, &incoming));
}

test "cache decay: active rewind follows other unborrowed states and explicit skip still protects them" {
    var r: run.Replay = undefined;
    r.init(t.allocator, .decayed_resume, 4 << 30, .{});
    defer r.deinit();
    const path = sequence(64, 1);
    const other = sequence(64, 1000);
    try put(&r, &path, 16);
    try put(&r, &other, 16);
    r.store.active_rewind_at = 16;
    r.store.entries.items[1].seen.hits = 100;
    try t.expectEqual(@as(?usize, 1), run.policy.victim(&r.store, .decayed_resume, null, &path));
    try t.expectEqual(@as(?usize, 0), run.policy.victim(&r.store, .decayed_resume, r.store.entries.items[1], &path));
}

test "cache decay: replacement cohort restarts histories and cannot resume the old system prefix" {
    var trace = try traces.make(t.allocator, .agent_cohorts);
    defer trace.deinit();
    try t.expectEqual(@as(usize, 864), trace.requests.len);
    const first = trace.requests[0];
    const replacement = trace.requests[432];
    try t.expectEqual(@as(u32, 1), first.session);
    try t.expectEqual(@as(u32, 7), replacement.session);
    try t.expectEqual(first.history, replacement.history);
    try t.expect(!std.mem.eql(u32, first.tokens[0..8192], replacement.tokens[0..8192]));
    var r: run.Replay = undefined;
    r.init(t.allocator, .decayed_resume, 16 << 30, .{});
    defer r.deinit();
    _ = try r.request(first);
    try t.expectEqual(@as(u32, 0), (try r.request(replacement)).from);
    try t.expectEqual(@as(u32, 8192), (try r.request(trace.requests[435])).from);
    const latest = r.store.find(trace.requests[435].tokens, &.{}, &.{}).?;
    try t.expectEqual(@as(u64, 0), latest.seen.hits);
}

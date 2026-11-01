//! Host tests of the EXL3 graph cache: sequence retirement, pending deadlines and the TP template.

const std = @import("std");
const cuda = @import("cuda");
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gtogether = @import("cuda_exl3_graph_together.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gdriver = @import("cuda_exl3_graph_test_driver.zig");
const Key = gkey.Key;
const Entry = gcache.Entry;
const Pending = gpending.Pending;
const PKey = gprefill.PKey;
const GraphTestDriver = gdriver.GraphTestDriver;

test "exl3: sequence graph retirement removes mixed lanes and pending work without evicting a live owner" {
    gcache.clear();
    defer gcache.clear();
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    var driver = GraphTestDriver.driver();
    const exec: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0][2] = 100;
    const owner_key = key;
    gcache.entries[0] = .{ .key = key, .exec = exec, .used = 1 };
    key.seq[0][2] = 200;
    const live_key = key;
    gcache.entries[1] = .{ .key = key, .exec = exec, .used = 1 };
    key.rows = 2;
    key.seq[1][2] = 100;
    gcache.entries[2] = .{ .key = key, .exec = exec, .used = 1 };
    const pending = try std.heap.c_allocator.create(Pending);
    pending.* = .{ .graph = .{ .d = &driver, .handle = @ptrFromInt(1) }, .ctx = null, .keep_template = true };
    pending.thread = try std.Thread.spawn(.{}, GraphTestDriver.finishPending, .{ pending, &driver });
    gcache.entries[3] = .{ .key = owner_key, .pending = pending, .used = 1 };
    const live_pending = try std.heap.c_allocator.create(Pending);
    live_pending.* = .{ .graph = .{ .d = &driver, .handle = @ptrFromInt(1) }, .ctx = null, .keep_template = true };
    live_pending.thread = try std.Thread.spawn(.{}, GraphTestDriver.finishPending, .{ live_pending, &driver });
    var live_pending_key = live_key;
    live_pending_key.rows = 3;
    gcache.entries[4] = .{ .key = live_pending_key, .pending = live_pending, .used = 1 };
    const pkey: PKey = .{ .layer = 0, .rows = 32, .sub = 16, .pos0 = 0, .deep = 1, .base = 2, .base_cap = 32, .seq = .{ 1, 2, 100, 32, 3 }, .wait_count = 0, .wait_key = 0, .scores = 4, .tiles = false };
    gprefill.pentries[0] = .{ .key = pkey, .exec = exec, .used = 1 };
    var live_pkey = pkey;
    live_pkey.seq[2] = 200;
    gprefill.pentries[1] = .{ .key = live_pkey, .exec = exec, .used = 1 };
    gcache.seen[0] = .{ .key = owner_key, .n = 10, .used = 1 };
    gcache.seen[1] = .{ .key = live_key, .n = 10, .used = 1 };
    gprefill.pseen[0] = .{ .key = pkey, .n = 10 };
    gprefill.pseen[1] = .{ .key = live_pkey, .n = 10 };
    const Seq = struct { tail: struct { ptr: u64 } };
    gcache.releaseSeq(Seq{ .tail = .{ .ptr = 100 } });
    try std.testing.expect(gcache.find(owner_key) == null);
    try std.testing.expect(gcache.find(live_key) != null);
    try std.testing.expect(gcache.entries[2] == null and gcache.entries[3] == null);
    try std.testing.expect(gcache.entries[4].?.pending == live_pending);
    try std.testing.expect(gprefill.pentries[0] == null and gprefill.pentries[1] != null);
    try std.testing.expect(gcache.seen[0] == null and gcache.seen[1] != null);
    try std.testing.expect(gprefill.pseen[0] == null and gprefill.pseen[1] != null);
    try std.testing.expectEqual(@as(usize, 4), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.graph_drops);
    try std.testing.expectEqual(@as(u32, 1), gcache.sighting(owner_key));
    gcache.releaseSeq(Seq{ .tail = .{ .ptr = 100 } });
    try std.testing.expectEqual(@as(usize, 4), GraphTestDriver.exec_drops);
}

test "exl3: snapshot graph retirement keeps prefill templates but drops decode graphs" {
    gcache.clear();
    defer gcache.clear();
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    var driver = GraphTestDriver.driver();
    const exec: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0][2] = 100;
    gcache.entries[0] = .{ .key = key, .exec = exec, .used = 1 };
    const pkey: PKey = .{ .layer = 0, .rows = 32, .sub = 16, .pos0 = 0, .deep = 1, .base = 2, .base_cap = 32, .seq = .{ 1, 2, 100, 32, 3 }, .wait_count = 0, .wait_key = 0, .scores = 4, .tiles = false };
    gprefill.pentries[0] = .{ .key = pkey, .exec = exec, .used = 1 };
    gprefill.pseen[0] = .{ .key = pkey, .n = 2 };
    const Seq = struct { tail: struct { ptr: u64 } };
    gcache.releaseDecodeSeq(Seq{ .tail = .{ .ptr = 100 } });
    try std.testing.expect(gcache.entries[0] == null);
    try std.testing.expect(gprefill.pentries[0] != null);
    try std.testing.expect(gprefill.pseen[0] != null);
}

test "exl3: pending graph deadlines and failed instantiation never expose an executable" {
    var pending: Pending = .{ .graph = undefined, .ctx = null, .keep_template = true, .started_ns = wire.now() };
    var entry: Entry = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .pending = &pending, .used = 1, .cohort = 7 };
    try std.testing.expectEqual(collective.GraphState.pending, gtogether.graphVote(&entry, 1000).state);
    pending.started_ns = wire.now() -| 2 * std.time.ns_per_ms;
    try std.testing.expectEqual(collective.GraphState.timed_out, gtogether.graphVote(&entry, 1).state);
    entry.pending = null;
    try std.testing.expectEqual(collective.GraphState.failed, gtogether.graphVote(&entry, 1000).state);
    gcache.drop(&entry);
}

test "exl3: asynchronous TP instantiation preserves its NCCL template until owner retirement" {
    if (!gpending.asyncInstantiate()) return error.SkipZigTest;
    const previous_api = gpending.ctx_api;
    defer gpending.ctx_api = previous_api;
    gpending.ctx_api = .{ .get = GraphTestDriver.getContext, .set = GraphTestDriver.setContext };
    var driver = GraphTestDriver.driver();
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(1) };
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    const pending = (try gpending.startInstantiate(graph, true)).?;
    var entry: Entry = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .pending = pending, .used = 1, .cohort = 9 };
    defer gcache.drop(&entry);
    const until = wire.deadline(1000);
    while (!pending.done.load(.acquire)) {
        if (wire.now() >= until) return error.RankTimeout;
        wire.pause();
    }
    const vote = gtogether.graphVote(&entry, 1000);
    try std.testing.expectEqual(collective.GraphState.ready, vote.state);
    try std.testing.expectEqual(@as(u64, 9), vote.cohort);
    try std.testing.expect(entry.pending == null and entry.exec != null and entry.template != null);
    try std.testing.expectEqual(@as(usize, 0), GraphTestDriver.graph_drops);
}

test "exl3: a decode round captures on its second sighting and a window on its twelfth" {
    const round: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    const window: Key = .{ .kind = 2, .rows = 3, .scratch = 1, .scores = 2, .output = 3 };
    try std.testing.expectEqual(@as(u32, 2), gcache.afterKey(round));
    try std.testing.expectEqual(@as(u32, 12), gcache.afterKey(window));
    gcache.capture_after_override = 1;
    defer gcache.capture_after_override = null;
    try std.testing.expectEqual(@as(u32, 1), gcache.afterKey(window));
}

test "exl3: engine retirement drops only that owner's graphs and prevents cross-owner updates" {
    gcache.clear();
    defer gcache.clear();
    const Engine = @import("../../cuda_prompt.zig").Engine;
    var first: Engine = undefined;
    var second: Engine = undefined;
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_drops = 0;
    const a: Key = .{ .owner = @intFromPtr(&first), .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    var b = a;
    b.owner = @intFromPtr(&second);
    try std.testing.expect(!gkey.sameShape(a, b));
    gcache.entries[0] = .{ .key = a, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1 };
    gcache.entries[1] = .{ .key = b, .exec = .{ .d = &driver, .handle = @ptrFromInt(2) }, .used = 2 };
    gcache.seen[0] = .{ .key = a, .n = 2, .used = 1 };
    gcache.seen[1] = .{ .key = b, .n = 3, .used = 2 };
    gcache.releaseFor(&first);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_drops);
    try std.testing.expect(gcache.entries[0] == null and gcache.seen[0] == null);
    try std.testing.expect(gcache.entries[1] != null and gcache.seen[1] != null);
    gcache.releaseFor(&second);
    try std.testing.expectEqual(@as(usize, 2), GraphTestDriver.exec_drops);
}

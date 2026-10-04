//! 32 to 128 streams: what fits at 8k and 32k context, prompt rows a round, and 32 toy streams joining while others decode.
const std = @import("std");
const node = @import("node.zig");
const plan = @import("plan.zig");
const budget = @import("budget.zig");
const traffic = @import("traffic.zig");
const round = @import("round.zig");
const admission = @import("admission.zig");
const toy = @import("toy.zig");
const toy_cluster = @import("toy_cluster.zig");
const ct = @import("cost_test.zig");

const gib = node.gib;

test "K3 and GLM-5.3 hold 128 streams at 8k either way; at 32k only with MLA split by streams" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |glm| {
        const heads = try ct.setup(a, glm, .{ .streams = 128, .context = 32768 });
        const streams = try ct.setup(a, glm, .{ .streams = 128, .context = 32768, .mla = .streams });
        try std.testing.expect(admission.fitsAt(&heads.p, heads.ns, &heads.s, 128, 8192));
        try std.testing.expect(admission.fitsAt(&streams.p, streams.ns, &streams.s, 128, 8192));
        try std.testing.expect(!admission.fitsAt(&heads.p, heads.ns, &heads.s, 128, 32768));
        try std.testing.expect(admission.fitsAt(&streams.p, streams.ns, &streams.s, 128, 32768));
        try std.testing.expect(admission.maxStreams(&heads.p, heads.ns, &heads.s, 32768) >= 32);
        try std.testing.expect(admission.maxStreams(&streams.p, streams.ns, &streams.s, 32768) > 128);
        try std.testing.expect(admission.maxContext(&heads.p, heads.ns, &heads.s, 32) > 65536);
        try std.testing.expectEqual(@as(usize, 1), admission.best(&.{ .{ .p = &heads.p, .ns = heads.ns }, .{ .p = &streams.p, .ns = streams.ns } }, &heads.s, traffic.tb5_polled));
    }
    const k3 = try ct.setup(a, false, .{ .streams = 32, .context = 8192 });
    const k3s = try ct.setup(a, false, .{ .streams = 32, .context = 8192, .mla = .streams });
    const k3x = try ct.setup(a, false, .{ .streams = 32, .context = 8192, .layout = .{ .tensor = 4, .expert = 1 } });
    try std.testing.expectEqual(@as(usize, 0), admission.best(&.{ .{ .p = &k3.p, .ns = k3.ns }, .{ .p = &k3s.p, .ns = k3s.ns } }, &k3.s, traffic.tb5_polled));
    try std.testing.expectEqual(@as(usize, 2), admission.best(&.{ .{ .p = &k3.p, .ns = k3.ns }, .{ .p = &k3s.p, .ns = k3s.ns }, .{ .p = &k3x.p, .ns = k3x.ns } }, &k3.s, traffic.tb5_polled));
    const chunk = admission.chunkRows(&k3.p, k3.ns, &k3.s, 32, 8192, 0.5, traffic.tb5_polled);
    try std.testing.expect(chunk >= 32 and chunk <= 128);
}

test "a round takes every decode row, then prompt rows oldest first up to the chunk and the row limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const decode = [_]round.Row{ .{ .slot = 0, .token = 5, .index = 9, .sample = true }, .{ .slot = 1, .token = 6, .index = 4, .sample = true } };
    var pending = [_]admission.Pending{ .{ .slot = 7, .tokens = &.{ 1, 2, 3 } }, .{ .slot = 8, .tokens = &.{ 4, 5, 6, 7, 8 } } };
    const rows = try admission.compose(a, .{ .rows = 8, .chunk = 5 }, &decode, &pending);
    try std.testing.expectEqual(@as(usize, 7), rows.len);
    try std.testing.expectEqualSlices(round.Row, &decode, rows[0..2]);
    try std.testing.expectEqual(round.Row{ .slot = 7, .token = 3, .index = 2, .sample = true }, rows[4]);
    try std.testing.expectEqual(round.Row{ .slot = 8, .token = 5, .index = 1, .sample = false }, rows[6]);
    try std.testing.expectEqual(@as(u32, 3), pending[1].left());
    const tight = try admission.compose(a, .{ .rows = 3, .chunk = 5 }, &decode, &pending);
    try std.testing.expectEqual(@as(usize, 3), tight.len);
    try std.testing.expectEqual(@as(u32, 2), pending[1].left());
}

const Live = struct { prompt: []u32, out: std.ArrayList(u32) = .empty, pending: ?usize = null, joined: bool = false };

/// Each stream's greedy tokens when 32 streams join a few at a time, their prompts chunked into decode rounds.
fn converged(a: std.mem.Allocator, w: *const toy.Weights, n: u32, v: toy_cluster.Variant, streams: []Live, max_new: usize) !void {
    const c = try toy_cluster.Cluster.init(std.testing.allocator, w, n, v);
    defer c.deinit();
    var pending: std.ArrayList(admission.Pending) = .empty;
    var joined: usize = 0;
    var rounds: usize = 0;
    while (true) : (rounds += 1) {
        while (joined < streams.len and joined <= rounds * 3) : (joined += 1) {
            try pending.append(a, .{ .slot = @intCast(joined), .tokens = streams[joined].prompt });
            streams[joined].joined = true;
        }
        var decode: std.ArrayList(round.Row) = .empty;
        for (streams, 0..) |*st, i| {
            if (st.out.items.len == 0 or st.out.items.len >= max_new) continue;
            const index = st.prompt.len + st.out.items.len - 1;
            try decode.append(a, .{ .slot = @intCast(i), .token = st.out.items[st.out.items.len - 1], .index = @intCast(index), .sample = true });
        }
        if (decode.items.len == 0 and joined == streams.len and pending.items.len == 0) break;
        const rows = try admission.compose(a, .{ .rows = 40, .chunk = 9 }, decode.items, pending.items);
        while (pending.items.len > 0 and pending.items[0].left() == 0) _ = pending.orderedRemove(0);
        const logits = try c.leader.run(rows);
        defer std.testing.allocator.free(logits);
        var k: usize = 0;
        for (rows) |row| {
            if (!row.sample) continue;
            const vocab = w.dims.vocab;
            try streams[row.slot].out.append(a, try c.leader.draw(logits[k * vocab ..][0..vocab], null, 0));
            k += 1;
        }
    }
}

test "32 toy streams join while others decode, prompt chunks in the same rounds: each emits its solo decode, any variant" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try toy.Weights.init(a, .{}, 6);
    var prng: std.Random.DefaultPrng = .init(7);
    const prompts = try a.alloc([]u32, 32);
    for (prompts) |*p| {
        p.* = try a.alloc(u32, prng.random().intRangeAtMost(usize, 2, 23));
        for (p.*) |*t| t.* = prng.random().uintLessThan(u32, w.dims.vocab);
    }
    const max_new = 6;
    const bt = @import("backend_test.zig");
    for ([_]struct { u32, toy_cluster.Variant }{ .{ 4, .{} }, .{ 4, .{ .by_streams = true, .split_sums = true } }, .{ 3, .{ .by_streams = true } } }) |case| {
        const streams = try a.alloc(Live, prompts.len);
        for (streams, prompts) |*st, p| st.* = .{ .prompt = p };
        try converged(a, &w, case[0], case[1], streams, max_new);
        for (streams, prompts) |st, p| {
            const solo = try bt.decode(&w, p, null, max_new);
            defer std.testing.allocator.free(solo);
            try std.testing.expectEqualSlices(u32, solo, st.out.items);
        }
    }
}

test "the admission tables for CLUSTER.md (TF_CLUSTER_REPORT=1)" {
    if (std.testing.environ.getPosix("TF_CLUSTER_REPORT") == null) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |glm| for ([_]plan.MlaSplit{ .heads, .streams }) |mla| {
        const x = try ct.setup(a, glm, .{ .streams = 32, .context = 8192, .mla = mla });
        std.debug.print("\n== {s}, MLA by {s}: max streams {d} at 8k, {d} at 32k; max context at 32 streams {d}k\n", .{ if (glm) "GLM-5.3" else "K3", @tagName(mla), admission.maxStreams(&x.p, x.ns, &x.s, 8192), admission.maxStreams(&x.p, x.ns, &x.s, 32768), admission.maxContext(&x.p, x.ns, &x.s, 32) / 1024 });
        for ([_]u64{ 8192, 32768 }) |ctx| for ([_]u32{ 32, 64, 128 }) |n| {
            var q = x.p;
            q.opts.streams = n;
            q.opts.context = ctx;
            var worst: u64 = 0;
            var kv: u64 = 0;
            var st: u64 = 0;
            for (x.ns, 0..) |need, r| {
                const held = budget.caches(&q, &x.s, @intCast(r));
                const total = need.weightBytes() + need.drafter + budget.buffers(&q, &x.s, @intCast(r)) + held.kv + held.state + q.opts.margin;
                if (total < worst) continue;
                worst = total;
                kv = held.kv;
                st = held.state;
            }
            std.debug.print("  {d:>3} streams x {d:>2}k: kv {d:>6.1} GiB, state {d:>5.1} GiB, need {d:>6.1} of 453.1 GiB {s}\n", .{ n, ctx / 1024, budget.gb(kv), budget.gb(st), budget.gb(worst), if (admission.fitsAt(&x.p, x.ns, &x.s, n, ctx)) "fits" else "NO" });
        };
        std.debug.print("  prompt rows a round at 32 decode rows, 8k, +50% time: {d}\n", .{admission.chunkRows(&x.p, x.ns, &x.s, 32, 8192, 0.5, traffic.tb5_polled)});
    };
}

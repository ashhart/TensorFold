//! K3 and GLM-5.3 rounds on four 512 GiB Studios at 1 to 128 rows: dedup, caches, collectives, and the MLA split.
const std = @import("std");
const node = @import("node.zig");
const model = @import("model.zig");
const estimate = @import("estimate.zig");
const plan = @import("plan.zig");
const budget = @import("budget.zig");
const cost = @import("cost.zig");
const traffic = @import("traffic.zig");
const status = @import("status.zig");
const pt = @import("plan_test.zig");

const gib = node.gib;

pub const Setup = struct { s: model.Shape, p: plan.Plan, ns: []budget.Need };

/// K3 or GLM-5.3 on the four Studios with `opts`; caches sized for the streams and context given.
pub fn setup(a: std.mem.Allocator, glm: bool, opts: plan.Options) !Setup {
    const s = if (glm) model.glm53() else model.k3();
    const c = if (glm) try estimate.glm(a, &s) else try pt.k3(a);
    const nodes = try a.alloc(node.Inventory, 4);
    for (nodes, 0..) |*n, i| n.* = pt.studio(i, 512 * gib, 0);
    const p = try plan.plan(a, c, &s, nodes, opts);
    return .{ .s = s, .p = p, .ns = try budget.needs(a, &p, c, &s) };
}

fn rate(x: Setup, rows: u32, context: u64, link: traffic.Link) f64 {
    const c = cost.round(&x.p, x.ns, &x.s, .{ .rows = rows, .streams = rows, .context = context }, link, .{});
    return c.tokensPerSecond(@floatFromInt(rows));
}

test "K3 at 32, 64 and 128 streams: aggregate rate rises with dedup; polled hand-offs beat event wakes" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const x = try setup(arena.allocator(), false, .{ .streams = 128, .context = 8192 });
    const r32 = rate(x, 32, 8192, traffic.tb5_polled);
    const r64 = rate(x, 64, 8192, traffic.tb5_polled);
    const r128 = rate(x, 128, 8192, traffic.tb5_polled);
    try std.testing.expect(r32 > 75 and r32 < 110);
    try std.testing.expect(r64 > r32 and r128 > r64 and r128 < 300);
    try std.testing.expect(rate(x, 32, 8192, traffic.tb5_event) < r32);
    const one = cost.round(&x.p, x.ns, &x.s, .{ .rows = 1 }, traffic.tb5_polled, .{});
    const wide = cost.round(&x.p, x.ns, &x.s, .{ .rows = 128, .streams = 128, .context = 8192 }, traffic.tb5_polled, .{});
    try std.testing.expect(wide.distinct > 0.85 * 896 and one.distinct < 17);
    try std.testing.expect(wide.expert_bytes > 20 * one.expert_bytes);
    const chunk = cost.round(&x.p, x.ns, &x.s, .{ .rows = 32, .streams = 32, .context = 8192, .prompt = 512 }, traffic.tb5_polled, .{});
    try std.testing.expect(chunk.compute_ms > chunk.memoryMs());
}

test "GLM-5.3 in FP8 reads far less a token than K3, and 128 streams at 32k context fit only split by streams" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const glm = try setup(a, true, .{ .streams = 32, .context = 8192 });
    const k3 = try setup(a, false, .{ .streams = 32, .context = 8192 });
    try std.testing.expect(rate(glm, 1, 2048, traffic.tb5_polled) > 1.8 * rate(k3, 1, 2048, traffic.tb5_polled));
    try std.testing.expect(rate(glm, 32, 8192, traffic.tb5_polled) > 1.3 * rate(k3, 32, 8192, traffic.tb5_polled));
    const heads = try setup(a, true, .{ .streams = 128, .context = 32768 });
    const streams = try setup(a, true, .{ .streams = 128, .context = 32768, .mla = .streams });
    try std.testing.expect(!budget.allFit(try budget.fits(a, &heads.p, heads.ns)));
    try std.testing.expect(budget.allFit(try budget.fits(a, &streams.p, streams.ns)));
    try std.testing.expect(streams.ns[1].of(.attention) > 2 * heads.ns[1].of(.attention));
    const hc = cost.round(&heads.p, heads.ns, &heads.s, .{ .rows = 128, .streams = 128, .context = 32768 }, traffic.tb5_polled, .{});
    const sc = cost.round(&streams.p, streams.ns, &streams.s, .{ .rows = 128, .streams = 128, .context = 32768 }, traffic.tb5_polled, .{});
    try std.testing.expect(sc.cache_bytes * 3 < hc.cache_bytes);
}

test "the cost tables for CLUSTER.md (TF_CLUSTER_REPORT=1)" {
    if (std.testing.environ.getPosix("TF_CLUSTER_REPORT") == null) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out: std.Io.Writer.Allocating = .init(a);
    const cases = [_]struct { []const u8, bool, plan.MlaSplit, u64 }{
        .{ "K3 TP4xEP4, MLA by heads (prose)", false, .heads, 2048 },
        .{ "K3 TP4xEP4, MLA by heads", false, .heads, 8192 },
        .{ "K3 TP4xEP4, MLA by streams", false, .streams, 8192 },
        .{ "K3 TP4xEP4, MLA by streams", false, .streams, 32768 },
        .{ "GLM-5.3 TP4xEP4, MLA by heads", true, .heads, 8192 },
        .{ "GLM-5.3 TP4xEP4, MLA by streams", true, .streams, 8192 },
        .{ "GLM-5.3 TP4xEP4, MLA by streams", true, .streams, 32768 },
    };
    for (cases) |k| {
        const x = try setup(a, k[1], .{ .streams = 128, .context = k[3], .mla = k[2] });
        try out.writer.print("== {s}, {d}k context\n", .{ k[0], k[3] / 1024 });
        try status.costs(&out.writer, &x.p, x.ns, &x.s, k[3]);
    }
    for ([_]struct { []const u8, plan.Layout }{ .{ "EP4", .{ .tensor = 4, .expert = 4 } }, .{ "experts split", .{ .tensor = 4, .expert = 1 } } }) |lay| {
        const x = try setup(a, false, .{ .streams = 32, .context = 2048, .layout = lay[1] });
        for ([_]struct { []const u8, traffic.Link }{ .{ "polled", traffic.tb5_polled }, .{ "event", traffic.tb5_event } }) |l| {
            const c = cost.round(&x.p, x.ns, &x.s, .{ .rows = 32, .streams = 32, .context = 2048 }, l[1], .{});
            try out.writer.print("acceptance round ({s}), 32 prose rows at 2k, {s}: BF16 {d:.1} ms, experts {d:.1} ms ({d:.1} GB), caches {d:.1} ms, compute {d:.1} ms (hidden), collectives {d:.1} ms in {d} steps ({d:.1} MB a link), host {d:.1} ms: {d:.1} ms, {d:.1} tok/s\n", .{ lay[0], l[0], c.dense_ms, c.expert_ms, @as(f64, @floatFromInt(c.expert_bytes)) / 1e9, c.cache_ms, c.compute_ms, c.comm.ms, c.comm.steps, @as(f64, @floatFromInt(c.comm.link_bytes)) / 1e6, c.host_ms, c.total_ms, c.tokensPerSecond(32) });
        }
    }
    std.debug.print("\n{s}\n", .{out.written()});
}

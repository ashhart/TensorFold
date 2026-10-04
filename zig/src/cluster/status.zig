//! What `tensorfold cluster status` and `check` print: nodes, links, memory, the placement per rank, costs and load progress.
const std = @import("std");
const node = @import("node.zig");
const topology = @import("topology.zig");
const plan_mod = @import("plan.zig");
const budget = @import("budget.zig");
const cost = @import("cost.zig");
const traffic = @import("traffic.zig");
const model = @import("model.zig");
const membership = @import("membership.zig");

const W = std.Io.Writer;

fn gib(b: u64) f64 {
    return budget.gb(b);
}

pub fn nodes(w: *W, ns: []const node.Inventory) !void {
    try w.print("{s:<10} {s:<16} {s:>5} {s:>8} {s:>10} {s:>8} {s:>9} {s:>6}\n", .{ "node", "chip", "cores", "memory", "gpu limit", "free", "disk", "ports" });
    for (ns) |n| {
        var up: u32 = 0;
        for (n.ports()) |p| up += @intFromBool(p.up);
        var limit_buf: [24]u8 = undefined;
        const limit = if (n.backend == .metal and n.wired_limit_mb == 0) try std.fmt.bufPrint(&limit_buf, "{d:.1}*", .{gib(n.gpu_limit)}) else try std.fmt.bufPrint(&limit_buf, "{d:.1}", .{gib(n.gpu_limit)});
        try w.print("{s:<10} {s:<16} {d:>5} {d:>7.0}G {s:>9}G {d:>7.0}G {d:>8.2}T {d:>3}/{d:<2}\n", .{ n.name.str(), n.chip.str(), n.gpu_cores, gib(n.memory), limit, gib(n.free), @as(f64, @floatFromInt(n.disk_free)) / 1e12, up, n.ports().len });
    }
    const t = node.totals(ns);
    try w.print("{s:<10} {s:<16} {d:>5} {d:>7.0}G {d:>9.1}G {d:>7.0}G {d:>8.2}T {d:>3}\n", .{ "total", "", t.gpu_cores, gib(t.memory), gib(t.gpu_limit), gib(t.free), @as(f64, @floatFromInt(t.disk_free)) / 1e12, t.ports_up });
    try w.writeAll("* iogpu.wired_limit_mb is 0: macOS's default limit\n");
}

pub fn links(w: *W, ns: []const node.Inventory, es: []const topology.Edge) !void {
    for (es) |e| {
        const a = ns[e.a];
        const b = ns[e.b];
        try w.print("{s}:{s} <-> {s}:{s}  {d} Gb/s {s} {s}\n", .{ a.name.str(), a.ports()[e.a_port].device.str(), b.name.str(), b.ports()[e.b_port].device.str(), e.gbps, @tagName(e.kind), if (e.up) "up" else "DOWN" });
    }
    try w.print("shape: {s}, {d} links\n", .{ @tagName(topology.shape(ns.len, es)), es.len });
}

/// One row per rank: weights by kind, caches, buffers, the total against the GPU limit, and the verdict.
pub fn placement(w: *W, p: *const plan_mod.Plan, ns: []const budget.Need, fs: []const budget.Fit) !void {
    try w.print("layout: tensor {d} x pipeline {d}, experts over {d} per stage; {d} reduction slices, {d} expert groups; plan {x:0>16}\n", .{ p.layout.tensor, p.layout.pipeline, p.layout.expert, p.opts.slices, p.opts.groups, p.digest });
    try w.print("{s:<10} {s:>9} {s:>9} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8} {s:>9} {s:>9} {s}\n", .{ "rank", "experts", "attention", "shared", "latent", "router", "emb+head", "kv+state", "other", "total", "limit", "fit" });
    for (ns, fs, p.nodes) |n, f, inv| {
        const other = n.activations + n.drafter + n.of(.dense) + n.of(.norms) + n.of(.vision) + n.of(.draft) + n.of(.other);
        try w.print("{s:<10} {d:>8.1}G {d:>8.1}G {d:>7.1}G {d:>7.1}G {d:>7.1}G {d:>7.1}G {d:>7.1}G {d:>7.1}G {d:>8.1}G {d:>8.1}G {s}\n", .{ inv.name.str(), gib(n.of(.experts)), gib(n.of(.attention)), gib(n.of(.shared)), gib(n.of(.latent)), gib(n.of(.router)), gib(n.of(.embed) + n.of(.head)), gib(n.kv + n.state), gib(other), gib(f.need), gib(f.limit), if (f.ok) "ok" else "NO" });
    }
    try budget.refusal(w, p, fs);
}

/// Round costs at 1 to 128 decode rows (one a stream, each with `context` cached), a prompt chunk, and event wakes.
pub fn costs(w: *W, p: *const plan_mod.Plan, ns: []const budget.Need, s: *const model.Shape, context: u64) !void {
    try w.print("{s:<18} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8} {s:>6} {s:>8} {s:>8} {s:>8} {s:>9} {s:>8}\n", .{ "round", "experts", "dense GB", "exp GB", "cache GB", "dense ms", "exp ms", "cache ms", "steps", "link MB", "sent MB", "comm ms", "round ms", "tok/s" });
    const shapes = [_]struct { []const u8, cost.Round, traffic.Link }{
        .{ "1 row", .{ .rows = 1, .streams = 1, .context = context }, traffic.tb5_polled },
        .{ "16 rows", .{ .rows = 16, .streams = 16, .context = context }, traffic.tb5_polled },
        .{ "32 rows", .{ .rows = 32, .streams = 32, .context = context }, traffic.tb5_polled },
        .{ "32 rows, event", .{ .rows = 32, .streams = 32, .context = context }, traffic.tb5_event },
        .{ "64 rows", .{ .rows = 64, .streams = 64, .context = context }, traffic.tb5_polled },
        .{ "128 rows", .{ .rows = 128, .streams = 128, .context = context }, traffic.tb5_polled },
        .{ "32 + 512 prompt", .{ .rows = 32, .streams = 33, .context = context, .prompt = 512 }, traffic.tb5_polled },
    };
    for (shapes) |x| {
        const c = cost.round(p, ns, s, x[1], x[2], .{});
        const bound = if (c.compute_ms > c.memoryMs()) " compute" else "";
        try w.print("{s:<18} {d:>8.0} {d:>8.1} {d:>8.1} {d:>8.1} {d:>8.1} {d:>8.1} {d:>8.1} {d:>6} {d:>8.1} {d:>8.0} {d:>8.1} {d:>9.1} {d:>8.1}{s}\n", .{ x[0], c.distinct, gb10(c.dense_bytes), gb10(c.expert_bytes), gb10(c.cache_bytes), c.dense_ms, c.expert_ms, c.cache_ms, c.comm.steps, @as(f64, @floatFromInt(c.comm.link_bytes)) / 1e6, @as(f64, @floatFromInt(c.comm.sent)) / 1e6, c.comm.ms, c.total_ms, c.tokensPerSecond(@floatFromInt(x[1].rows)), bound });
    }
}

fn gb10(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// Each member's phase and load progress as the membership gossips them.
pub fn progress(w: *W, m: *membership.Membership) !void {
    try w.print("{s:<10} {s:<8} {s:<8} {s:>7}\n", .{ "node", "state", "phase", "loaded" });
    try w.print("{s:<10} {s:<8} {s:<8} {d:>6.1}%\n", .{ m.inv.name.str(), @tagName(m.me.state), @tagName(m.me.phase), @as(f64, @floatFromInt(m.me.progress)) / 100 });
    for (m.members.items) |x| {
        try w.print("{s:<10} {s:<8} {s:<8} {d:>6.1}%\n", .{ x.inv.name.str(), @tagName(x.entry.state), @tagName(x.entry.phase), @as(f64, @floatFromInt(x.entry.progress)) / 100 });
    }
    try w.print("leader {x}, epoch {d}, {s}\n", .{ m.leader, m.me.epoch, if (m.settled()) "settled" else "settling" });
}

test "the K3 status on four Studios prints every rank, fits, and the round costs" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pt = @import("plan_test.zig");
    const c = try pt.k3(a);
    const s = model.k3();
    var ns: [4]node.Inventory = undefined;
    for (&ns, 0..) |*n, i| n.* = pt.studio(i, 512 * node.gib, 0);
    const p = try plan_mod.plan(a, c, &s, &ns, .{ .drafter_bytes = 4 * node.gib });
    const needs = try budget.needs(a, &p, c, &s);
    const fs = try budget.fits(a, &p, needs);
    var out: W.Allocating = .init(a);
    try nodes(&out.writer, &ns);
    try placement(&out.writer, &p, needs, fs);
    try costs(&out.writer, &p, needs, &s, 8192);
    const text = try a.dupe(u8, out.written());
    if (std.testing.environ.getPosix("TF_CLUSTER_REPORT") != null) {
        std.debug.print("\n{s}\n", .{text});
        const others = [_]struct { []const u8, plan_mod.Options }{
            .{ "experts split by tensor parallelism", .{ .layout = .{ .tensor = 4, .expert = 1 } } },
            .{ "latent input replicated", .{ .latent_in = .replicate } },
            .{ "pipeline 4", .{ .layout = .{ .pipeline = 4 } } },
        };
        for (others) |o| {
            const q = try plan_mod.plan(a, c, &s, &ns, o[1]);
            const qn = try budget.needs(a, &q, c, &s);
            out.clearRetainingCapacity();
            try placement(&out.writer, &q, qn, try budget.fits(a, &q, qn));
            try costs(&out.writer, &q, qn, &s, 8192);
            std.debug.print("== {s}\n{s}\n", .{ o[0], out.written() });
        }
    }
    try std.testing.expect(std.mem.indexOf(u8, text, "node-d") != null and std.mem.indexOf(u8, text, " NO") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "512 prompt") != null and std.mem.indexOf(u8, text, "128 rows") != null);
}

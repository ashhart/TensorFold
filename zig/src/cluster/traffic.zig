//! A converged round's collectives: what each layer exchanges, and one-shot or reduce-scatter + all-gather (both exact).
const std = @import("std");
const fabric = @import("fabric");
const model = @import("model.zig");
const plan_mod = @import("plan.zig");

/// One collective step's price: a message's one-way latency, the GPU-host handoff around it, one link's payload rate.
pub const Link = struct {
    write_ns: f64,
    handoff_ns: f64,
    bandwidth: f64,

    pub fn from(link: fabric.cost.Link, h: fabric.cost.Handoff) Link {
        return .{ .write_ns = link.latency_ns, .handoff_ns = h.gpu_to_host_ns + h.host_to_gpu_ns, .bandwidth = link.bandwidth };
    }
};

/// Thunderbolt 5 as the fabric measured it (3.5 us one way, 9.38 GB/s); the GPU polls its receive ring (gate Q3, unverified).
pub const tb5_polled: Link = .from(fabric.cost.tb5, fabric.cost.polled_handoff);
/// The same link when every step wakes the GPU through a shared event (90-115 us measured on the M5).
pub const tb5_event: Link = .from(fabric.cost.tb5, fabric.cost.event_handoff);

/// The largest Thunderbolt message: 4,095 packets of 4 KiB (the send queue counts packets).
pub const max_write: u64 = 4095 * 4096;

pub const Traffic = struct {
    /// Exchange steps, each one GPU-host handoff on every rank.
    steps: u32 = 0,
    /// Bytes on the busiest mesh link, one direction, over the round.
    link_bytes: u64 = 0,
    /// Bytes one rank sends over the round, all links.
    sent: u64 = 0,
    /// The largest single write; it must fit one registration.
    largest: u64 = 0,
    ms: f64 = 0,
    reduce_scatters: u32 = 0,

    /// `steps` steps, each a message of `largest` bytes or less to each peer; a message past `max_write` goes in pieces.
    fn add(t: *Traffic, link: Link, peers: u32, steps: u32, peer_bytes: u64, largest: u64) void {
        const pieces = std.math.divCeil(u64, @max(largest, 1), max_write) catch unreachable;
        t.steps += steps;
        t.link_bytes += peer_bytes;
        t.sent += peer_bytes * peers;
        t.largest = @max(t.largest, @min(largest, max_write));
        const fixed = @as(f64, @floatFromInt(steps)) * (link.write_ns * @as(f64, @floatFromInt(pieces)) + link.handoff_ns);
        t.ms += (fixed + @as(f64, @floatFromInt(peer_bytes)) / link.bandwidth * 1e9) / 1e6;
    }

    /// An exact sum of fp32 partials over `rows` x `width`: one-shot, or reduce-scatter then all-gather, whichever is faster.
    fn allReduce(t: *Traffic, link: Link, ranks: u32, rows: u64, width: u64, residual: u64) void {
        const one = rows * width * 4;
        const part = rows * (width / ranks);
        const split = part * 4 + part * residual;
        const one_ns = link.write_ns + link.handoff_ns + @as(f64, @floatFromInt(one)) / link.bandwidth * 1e9;
        const split_ns = 2 * (link.write_ns + link.handoff_ns) + @as(f64, @floatFromInt(split)) / link.bandwidth * 1e9;
        if (split_ns < one_ns) {
            t.add(link, ranks - 1, 2, split, part * 4);
            t.reduce_scatters += 1;
        } else t.add(link, ranks - 1, 1, one, one);
    }
};

pub const Shape = struct {
    /// Sampled decode rows, whose logits go to the leader.
    rows: u32,
    /// Prompt chunk rows riding the same round.
    prompt: u32 = 0,
    /// Bytes of a residual-stream value (bf16); all-gathers carry values every rank would have computed itself.
    residual: u64 = 2,
};

/// Every collective of one round, stage by stage, plus the hand-offs between stages, the logits and the descriptor.
pub fn round(p: *const plan_mod.Plan, s: *const model.Shape, x: Shape, link: Link) Traffic {
    var t: Traffic = .{};
    const rows: u64 = x.rows + x.prompt;
    const h = s.hidden;
    for (p.stages) |st| {
        if (st.tensor < 2) continue;
        const n = st.tensor;
        for (st.layers.begin..st.layers.end) |l| {
            const layer: u32 = @intCast(l);
            if (s.kind(layer) == .mla and p.opts.mla == .streams) {
                const own = std.math.divCeil(u64, rows, n) catch unreachable;
                t.add(link, n - 1, 1, own * h * x.residual, own * h * x.residual);
            } else t.allReduce(link, n, rows, h, x.residual);
            if (s.moe(layer) and s.latent != s.hidden) {
                const part = rows * (s.latent / n);
                if (p.opts.latent_in == .column) t.add(link, n - 1, 1, part * x.residual, part * x.residual);
                t.add(link, n - 1, 1, part * 4, part * 4);
            }
            t.allReduce(link, n, rows, h, x.residual);
        }
    }
    if (p.stages.len > 1) {
        const hop = rows * h * x.residual;
        t.add(link, 1, @intCast(p.stages.len - 1), hop * (p.stages.len - 1), hop);
    }
    if (p.ranks() > 1) {
        const last = p.stages[p.stages.len - 1];
        const logits = @as(u64, x.rows) * (s.vocab / last.tensor) * 4;
        t.add(link, 1, 1, logits, logits);
        t.add(link, p.ranks() - 1, 1, 21 + rows * 13, 21 + rows * 13);
    }
    return t;
}

test "K3 at 32 rows: reduce-scatter and all-gather win with polled hand-offs, one-shot with event wakes" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pt = @import("plan_test.zig");
    const node = @import("node.zig");
    const c = try pt.k3(a);
    const s = model.k3();
    var nodes: [4]node.Inventory = undefined;
    for (&nodes, 0..) |*n, i| n.* = pt.studio(i, 512 * node.gib, 0);
    const p = try plan_mod.plan(a, c, &s, &nodes, .{});
    const polled = round(&p, &s, .{ .rows = 32 }, tb5_polled);
    const evented = round(&p, &s, .{ .rows = 32 }, tb5_event);
    try std.testing.expectEqual(@as(u32, 93 * 2), polled.reduce_scatters);
    try std.testing.expectEqual(@as(u32, 0), evented.reduce_scatters);
    try std.testing.expectEqual(@as(u32, 93 * 4 + 92 * 2 + 2), polled.steps);
    try std.testing.expect(polled.ms < 20 and evented.ms > 50 and polled.sent < evented.sent);
    const one = round(&p, &s, .{ .rows = 1 }, tb5_polled);
    try std.testing.expectEqual(@as(u32, 0), one.reduce_scatters);
    const chunk = round(&p, &s, .{ .rows = 32, .prompt = 512 }, tb5_polled);
    try std.testing.expect(chunk.largest <= max_write and chunk.ms > polled.ms);
    const wide = round(&p, &s, .{ .rows = 128 }, tb5_polled);
    try std.testing.expect(wide.largest == max_write);
    const streams = try plan_mod.plan(a, c, &s, &nodes, .{ .mla = .streams });
    const dp = round(&streams, &s, .{ .rows = 32 }, tb5_polled);
    try std.testing.expect(dp.sent < polled.sent and dp.steps < polled.steps);
}

//! A decode window's L2 prefetch (TF_DSV41_L2PF): each site's sources, the table rows, the side-stream launches.

const std = @import("std");
const block = @import("block.zig");
const Emitter = block.Emitter;
const L2Src = block.Emitter.L2Src;
const L2Site = block.Emitter.L2Site;

/// l2pf._lin_tensors: every projection's suh and svh, then their trellis words.
pub fn linSrcs(e: *Emitter, out: *std.ArrayList(L2Src), ws: []const []const u8, ks: []const i64, ns: []const i64) !void {
    for (ws, ks, ns) |w, k, n| try out.appendSlice(e.a, &.{
        .{ .kind = .weight, .name = try e.fmt("{s}.suh", .{w}), .bytes = @intCast(2 * k) },
        .{ .kind = .weight, .name = try e.fmt("{s}.svh", .{w}), .bytes = @intCast(2 * n) },
    });
    for (ws, ks, ns) |w, k, n| try out.append(e.a, .{ .kind = .words, .name = w, .bytes = try e.trellisBytes(w, k, n) });
}

/// Site `site`'s sources: "o" and "f" of block `L`, "x" of the block before `L` (its table is "s.L<L>.pf.x").
pub fn l2pfSources(e: *Emitter, L: u32, site: L2Site) ![]const L2Src {
    var out: std.ArrayList(L2Src) = .empty;
    switch (site) {
        .o => {
            const D: i64 = e.cfg.hidden;
            const g = e.cfg.o_groups / e.o.world;
            const ki: i64 = @divExact(@as(i64, e.cfg.heads) * e.cfg.head_dim, @as(i64, e.cfg.o_groups));
            const ws = try e.a.alloc([]const u8, g);
            const ks = try e.a.alloc(i64, g);
            const ns = try e.a.alloc(i64, g);
            for (ws, ks, ns, 0..) |*w, *k, *n, i| {
                w.* = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i });
                k.* = ki;
                n.* = e.cfg.o_lora;
            }
            try e.linSrcs(&out, ws, ks, ns);
            try e.linSrcs(&out, &.{try e.fmt("L{d}.attn.wo_b", .{L})}, &.{@as(i64, g) * e.cfg.o_lora}, &.{D});
        },
        .f => {
            const D: i64 = e.cfg.hidden;
            const I: i64 = (try @import("partition.zig").share(e.cfg.expert_width, e.o.rank, e.o.world)).len;
            const ex = e.cfg.expertsOf(L);
            const sh = try e.fmt("L{d}.moe.shared.0.w1", .{L});
            const k2: u64 = try e.w.k2(sh);
            const shared: u64 = @as(u64, @intCast(@divExact(D, 16) * @divExact(I, 16))) * 8 * k2 * 2;
            try out.appendSlice(e.a, &.{
                .{ .kind = .weight, .name = try e.fmt("L{d}.moe.gate", .{L}), .bytes = @as(u64, ex.count) * @as(u64, @intCast(D)) * 2 },
                .{ .kind = .expert, .name = try e.fmt("s.L{d}.ex.tp_g", .{L}), .index = ex.count, .bytes = shared },
                .{ .kind = .expert, .name = try e.fmt("s.L{d}.ex.tp_u", .{L}), .index = ex.count, .bytes = shared },
            });
        },
        .x => {
            const xg = try e.xGroup(L);
            try e.linSrcs(&out, xg.ws, xg.ks, xg.ns);
            const qg = try e.qGroup(L);
            try e.linSrcs(&out, qg.ws, qg.ks, qg.ns);
        },
    }
    return out.items;
}

pub fn srcSizes(e: *Emitter, srcs: []const L2Src) ![]const u64 {
    const out = try e.a.alloc(u64, srcs.len);
    for (srcs, out) |x, *y| y.* = x.bytes;
    return out;
}

/// l2pf.take: spans of `sizes` in order within the budget, the last one's head.
pub fn take(e: *const Emitter, sizes: []const u64) i64 {
    var left = e.o.l2pf_budget;
    var rows: i64 = 0;
    for (sizes) |s| {
        if (left < 16) break;
        const n = @min(left, s) / 16 * 16;
        if (n == 0) continue;
        rows += 1;
        left -= n;
    }
    return rows;
}

/// Site "o" (after wq_b): wo_a's groups, then wo_b; l2pf segments.
pub fn prefetchO(e: *Emitter, L: u32) !void {
    const sizes = try e.srcSizes(try e.l2pfSources(L, .o));
    var total: u64 = 0;
    var left = e.o.l2pf_budget;
    for (sizes) |s| {
        if (left < 16) break;
        const n = @min(left, s) / 16 * 16;
        total += n;
        left -= n;
    }
    const pieces = std.math.divCeil(u64, total, @intCast(e.o.l2pf_chunk)) catch 1;
    const grid: i64 = @max(1, @min(e.o.sms, @as(i64, @intCast(std.math.divCeil(u64, pieces, 128) catch 1))));
    const rows = e.take(sizes);
    try e.ext("tf_dsv41_l2pf_v1.segments", &.{ .{ .opaque_table = (try e.buf("s.L{d}.pf.o", .{L}, .i64, &.{ rows, 2 })).t }, .{ .i = grid }, .{ .i = 128 }, .{ .i = e.o.l2pf_chunk }, .{ .b = true } });
}

pub fn paced(e: *Emitter, role: []const u8, rows: i64) !void {
    const ns: i64 = @max(1, @as(i64, @intFromFloat(@round(@as(f64, @floatFromInt(e.o.l2pf_chunk)) / e.o.l2pf_pace_gbps))));
    try e.ext("tf_dsv41_l2pace_v1.paced", &.{ .{ .opaque_table = (try e.buf("{s}", .{role}, .i64, &.{ rows, 2 })).t }, .{ .i = e.o.l2pf_pace_ctas }, .{ .i = e.o.l2pf_chunk }, .{ .i = ns }, .{ .i = 0 } });
}

/// Site "f" (after the attention sublayer): the router's gate, then the shared expert's gate and up trellises.
pub fn prefetchF(e: *Emitter, L: u32) !void {
    try e.paced(try e.fmt("s.L{d}.pf.f", .{L}), e.take(try e.srcSizes(try e.l2pfSources(L, .f))));
}

/// Site "x" of the block before `nxt` (after its MoE): `nxt`'s x group, then its q group.
pub fn prefetchX(e: *Emitter, nxt: u32) !void {
    try e.paced(try e.fmt("s.L{d}.pf.x", .{nxt}), e.take(try e.srcSizes(try e.l2pfSources(nxt, .x))));
}

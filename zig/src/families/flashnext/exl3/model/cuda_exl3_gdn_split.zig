//! The DeltaNet heads split across the two expert-parallel ranks, for decode rows only (rounds).

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const load = @import("../pack/cuda_exl3_load.zig");
const projection = @import("../../cuda_projection.zig");
const gdn = @import("../../cuda_gdn.zig");

/// A rank's heads: key heads, value heads and the width of its gated rows.
pub const rank_nk: usize = gdn.nk / 2;
pub const rank_nv: usize = gdn.nv / 2;
pub const rank_value: usize = rank_nv * gdn.dv;

comptime {
    std.debug.assert(rank_value == 3072 and rank_value % 128 == 0);
    std.debug.assert((gdn.nk * gdn.dk / 2) % 128 == 0 and (gdn.nv * gdn.dv / 2) % 128 == 0);
}

pub fn enabled(world: usize) bool {
    if (world != 2) return false;
    const v = std.c.getenv("TF_FLASHNEXT_GDN_SPLIT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The rank's decode views of a layer's projections.
pub const Views = struct {
    proj: projection.Projection,
    out: projection.Projection,
    hk0: u32,
    hv0: u32,
};

/// Columns [c0, c0 + n) of a weight (both whole 128-column blocks).
fn columns(w: cuda.exl3.Weight, c0: usize, n: usize) cuda.exl3.Weight {
    var v = w;
    v.trellis = w.trellis + @as(u64, c0 / 128) * w.stride_nb * 4;
    v.svh = w.svh + c0 * 2;
    if (w.bias != 0) v.bias = w.bias + c0 * 2;
    v.n = @intCast(n);
    return v;
}

/// Input rows [k0, k0 + k) of a weight (whole 128-row blocks).
fn inputRows(w: cuda.exl3.Weight, k0: usize, k: usize) cuda.exl3.Weight {
    var v = w;
    v.trellis = w.trellis + @as(u64, k0 / 16) * w.stride_k * 4;
    v.suh = w.suh + k0 * 2;
    v.k = @intCast(k);
    return v;
}

fn wrap(d: *cuda.Driver, p: *load.Proj) projection.Projection {
    var out = projection.Projection.empty(d);
    out.n = p.n;
    out.k = p.k;
    out.exl3 = p;
    return out;
}

/// Rank `rank`'s views of a linear-attention layer's stitched q|k|v|z|b|a projection and its out projection.
pub fn build(gpa: std.mem.Allocator, d: *cuda.Driver, proj: projection.Projection, out: projection.Projection, rank: usize) !?Views {
    if (rank > 1) return error.BadRank;
    const pe = proj.exl3 orelse return null;
    const oe = out.exl3 orelse return null;
    if (pe.n != gdn.proj_width or pe.k != wt.dims or oe.n != wt.dims or oe.k != gdn.value_dim or pe.parts.len != 3 or oe.parts.len != 1) return null;
    const qkv = switch (pe.parts[0].kind) {
        .exl3 => |w| w,
        else => return null,
    };
    const z = switch (pe.parts[1].kind) {
        .exl3 => |w| w,
        else => return null,
    };
    const o = switch (oe.parts[0].kind) {
        .exl3 => |w| w,
        else => return null,
    };
    if (pe.parts[0].col != 0 or pe.parts[0].n != gdn.conv_dim or pe.parts[1].col != gdn.conv_dim or pe.parts[1].n != gdn.value_dim) return null;
    if (pe.parts[2].col != gdn.conv_dim + gdn.value_dim or pe.parts[2].n != 2 * gdn.nv) return null;
    if (qkv.stride_nb == 0 or z.stride_nb == 0 or o.stride_k == 0) return null;

    const q_half = gdn.nk * gdn.dk / 2;
    const v_half = gdn.nv * gdn.dv / 2;
    const p = try gpa.create(load.Proj);
    p.* = .{ .gpa = gpa, .ops = pe.ops.retain(), .k = pe.k, .n = pe.n, .parts = &.{} };
    errdefer p.destroy();
    p.parts = try gpa.alloc(load.Part, 5);
    p.parts[0] = .{ .col = rank * q_half, .n = q_half, .kind = .{ .exl3 = columns(qkv, rank * q_half, q_half) } };
    p.parts[1] = .{ .col = gdn.nk * gdn.dk + rank * q_half, .n = q_half, .kind = .{ .exl3 = columns(qkv, gdn.nk * gdn.dk + rank * q_half, q_half) } };
    p.parts[2] = .{ .col = 2 * gdn.nk * gdn.dk + rank * v_half, .n = v_half, .kind = .{ .exl3 = columns(qkv, 2 * gdn.nk * gdn.dk + rank * v_half, v_half) } };
    p.parts[3] = .{ .col = gdn.conv_dim + rank * v_half, .n = v_half, .kind = .{ .exl3 = columns(z, rank * v_half, v_half) } };
    p.parts[4] = pe.parts[2];
    for (p.parts[0..4]) |pt| p.ops.noteWeight(pt.kind.exl3, pe.n); // the views' plans (split-K, partials) size the shared scratch too

    const q = try gpa.create(load.Proj);
    q.* = .{ .gpa = gpa, .ops = oe.ops.retain(), .k = rank_value, .n = oe.n, .parts = &.{} };
    errdefer q.destroy();
    q.parts = try gpa.alloc(load.Part, 1);
    q.parts[0] = .{ .col = 0, .n = oe.n, .kind = .{ .exl3 = inputRows(o, rank * rank_value, rank_value) } };
    q.ops.noteWeight(q.parts[0].kind.exl3, oe.n);
    return .{ .proj = wrap(d, p), .out = wrap(d, q), .hk0 = @intCast(rank * rank_nk), .hv0 = @intCast(rank * rank_nv) };
}

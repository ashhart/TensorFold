//! Native Gemma learner glue has checked typed extents and one deterministic owner per output cell.
const std = @import("std");
const rows = @import("living_rows.zig");
const wo = @import("weight_only.zig");
pub const Params = rows.Params;
pub const Op = enum { copy, zero, move, rms, geglu, softcap, scale, rope };
pub const names = [_][:0]const u8{ "tf_lw_g4_copy", "tf_lw_g4_zero", "tf_lw_g4_move", "tf_lw_g4_rms_back", "tf_lw_g4_geglu_back", "tf_lw_g4_softcap_back", "tf_lw_g4_scale", "tf_lw_g4_rope_back" };
pub const threads = 128;
fn mul(xs: []const usize) !usize {
    var n: usize = 1;
    for (xs) |x| n = try std.math.mul(usize, n, x);
    return n;
}
pub fn needs(op: Op, p: Params) !rows.Needs {
    const r: usize = p.u[0];
    const w: usize = p.u[1];
    if (r == 0 or r > 512 or w == 0 or !std.math.isFinite(p.f[0])) return error.LivingGemmaShape;
    var n: rows.Needs = .{ .cells = try mul(&.{ r, w }) };
    switch (op) {
        .copy => {
            if (p.u[3] > 2) return error.LivingGemmaShape;
            const stride: usize = if (p.u[2] == 0) w else p.u[2];
            const cols = if (p.u[3] == 2) (if (w % 16 == 0) 2 * w - 16 else return error.LivingGemmaShape) else w;
            const elements = if (p.u[3] == 1) n.cells else try std.math.add(usize, try mul(&.{ r - 1, stride }), cols);
            if (p.u[3] == 1 and (p.u[4] == 0 or w % p.u[4] != 0)) return error.LivingGemmaShape;
            n.bytes[0] = try mul(&.{ elements, 2 });
            n.bytes[8] = try mul(&.{ n.cells, 2 });
        },
        .zero => {
            n.bytes[8] = try mul(&.{ n.cells, 4 });
        },
        .move => {
            if (p.u[2] > 1) return error.LivingGemmaShape;
            n.bytes[0] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[0];
        },
        .rms => {
            const groups: usize = p.u[2];
            if (groups == 0 or groups > 64 or p.u[3] > 1 or p.f[0] <= 0) return error.LivingGemmaShape;
            n.cells = try mul(&.{ r, w, groups });
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            if (p.u[3] != 0) n.bytes[1] = try mul(&.{ w, 2 });
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[2];
        },
        .geglu => {
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = n.bytes[0];
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[2];
            n.bytes[9] = n.bytes[2];
        },
        .softcap => {
            if (p.f[0] <= 0) return error.LivingGemmaShape;
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[1];
        },
        .scale => {
            if (p.u[2] > 1) return error.LivingGemmaShape;
            n.bytes[0] = try mul(&.{ n.cells, 4 });
            n.bytes[1] = 2;
            n.bytes[8] = n.bytes[0];
        },
        .rope => {
            const dim: usize = p.u[2];
            if (dim == 0 or dim % 2 != 0 or w % dim != 0 or p.u[3] > 262144) return error.LivingGemmaShape;
            n.bytes[0] = try mul(&.{ n.cells, 4 });
            n.bytes[1] = try mul(&.{ dim / 2, 4 });
            n.bytes[8] = n.bytes[0];
        },
    }
    for (n.bytes) |b| if (b > std.math.maxInt(u32)) return error.LivingGemmaShape;
    if (n.cells > std.math.maxInt(u32)) return error.LivingGemmaShape;
    return n;
}
pub fn finish(op: Op, p: Params) !Params {
    var out = p;
    out.u[15] = @intCast((try needs(op, p)).cells);
    return out;
}
pub fn validate(op: Op, p: Params, ranges: [11]wo.Range) !void {
    const n = try needs(op, p);
    if (n.cells != p.u[15]) return error.LivingGemmaShape;
    for (ranges, n.bytes, 0..) |v, size, i| {
        if (size == 0) continue;
        const alignment: usize = if (op == .copy or ((op == .rms or op == .geglu) and i < 2) or (op == .softcap and i == 0) or (op == .scale and i == 1)) 2 else 4;
        if (v.address == 0 or v.address % alignment != 0 or v.len < size) return error.LivingGemmaBuffer;
        _ = try std.math.add(usize, v.address, size);
    }
    for (8..11) |i| for (0..i) |j| {
        if (n.bytes[i] == 0 or n.bytes[j] == 0) continue;
        const x = ranges[i].address;
        const y = ranges[j].address;
        const overlap = if (x <= y) y - x < n.bytes[i] else x - y < n.bytes[j];
        if (overlap and !(i == 8 and j == 0 and x == y and (op == .move or op == .scale))) return error.LivingGemmaAlias;
    };
}

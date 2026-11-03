//! Checked buffer extents for native learner row pullbacks shared by the CUDA and Metal launchers.
const std = @import("std");
const wo = @import("weight_only.zig");
pub const Control = enum(u3) { production, nan, repeat, tail, sums };
pub const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
pub const Op = enum { linear, rms, hc_act, hc_mix, hc_branch, swiglu, loss, nudge, gate, rope_rms, pocket_act, pocket_add, pocket_back, pocket_coeff };
pub const Params = extern struct { u: [16]u32 = @splat(0), f: [4]f32 = @splat(0) };
pub const names = [_][:0]const u8{
    "tf_lw_linear_back", "tf_lw_rms_back",   "tf_lw_hc_act_back", "tf_lw_hc_mix_back",  "tf_lw_hc_branch_back",
    "tf_lw_swiglu_back", "tf_lw_loss",       "tf_lw_nudge",       "tf_lw_gate_back",    "tf_lw_rope_rms_back",
    "tf_lw_pocket_act",  "tf_lw_pocket_add", "tf_lw_pocket_back", "tf_lw_pocket_coeff",
};
pub const threads = 128;
pub const Needs = struct { bytes: [11]usize = @splat(0), cells: usize };
fn mul(values: []const usize) !usize {
    var n: usize = 1;
    for (values) |v| n = try std.math.mul(usize, n, v);
    return n;
}
pub fn needs(op: Op, p: Params) !Needs {
    const r: usize = p.u[0];
    const w: usize = p.u[1];
    const s: usize = p.u[2];
    if (r == 0 or r > 512 or w == 0 or !std.math.isFinite(p.f[0])) return error.LivingRowsShape;
    var n: Needs = .{ .cells = try mul(&.{ r, w }) };
    switch (op) {
        .linear => {
            if (s == 0 or p.u[3] > 1 or (p.u[3] == 1 and (p.u[4] != 64 or s % 64 != 0))) return error.LivingRowsFormat;
            const weights = try mul(&.{ w, s });
            if (weights > std.math.maxInt(u32)) return error.LivingRowsShape;
            n.cells = try mul(&.{ r, s });
            const stride: usize = if (p.u[6] == 0) w else p.u[6];
            const end = try std.math.add(usize, p.u[7], w);
            if (end > stride) return error.LivingRowsShape;
            n.bytes[0] = try mul(&.{ try std.math.add(usize, try mul(&.{ r - 1, stride }), end), 4 });
            n.bytes[1] = try mul(&.{ weights, if (p.u[3] == 0) @as(usize, 2) else 1 }) / (if (p.u[3] == 0) @as(usize, 1) else 2);
            if (p.u[3] == 1) {
                n.bytes[2] = try mul(&.{ weights / 64, 2 });
                n.bytes[3] = n.bytes[2];
            }
            n.bytes[8] = try mul(&.{ n.cells, 4 });
        },
        .rms => {
            if (s == 0 or s > 8 or p.f[0] <= 0) return error.LivingRowsShape;
            n.cells = try mul(&.{ r, w, s });
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ w, s, 4 });
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[2];
        },
        .hc_act => {
            if (s < 2 or s > 8 or p.u[3] > 1) return error.LivingRowsShape;
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[1];
        },
        .hc_mix => {
            if (s < 2 or s > 8) return error.LivingRowsShape;
            n.cells = try mul(&.{ r, w, s });
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = n.bytes[0];
            n.bytes[2] = try mul(&.{ r, w, 4 });
            n.bytes[8] = try mul(&.{ n.cells, 4 });
            n.bytes[9] = n.bytes[8];
        },
        .hc_branch => {
            if (s < 2 or s > 8) return error.LivingRowsShape;
            n.cells = try mul(&.{ r, @max(w, s) });
            n.bytes[0] = try mul(&.{ r, w, s, 4 });
            n.bytes[1] = try mul(&.{ r, w, 2 });
            n.bytes[2] = try mul(&.{ r, s, 2 });
            n.bytes[8] = try mul(&.{ r, w, 4 });
            n.bytes[9] = try mul(&.{ r, s, 4 });
        },
        .swiglu => {
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = n.bytes[0];
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[2];
            n.bytes[9] = n.bytes[2];
        },
        .loss => {
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ r, 4 });
            n.bytes[2] = n.bytes[1];
            n.bytes[8] = try mul(&.{ n.cells, 4 });
            n.bytes[9] = try mul(&.{ r, 8 });
            n.cells = r;
        },
        .nudge => {
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[0];
        },
        .gate => {
            if (s == 0 or w % s != 0 or p.u[3] < 2 * w) return error.LivingRowsShape;
            n.bytes[0] = try mul(&.{ n.cells, 2 });
            n.bytes[1] = try mul(&.{ r, p.u[3], 2 });
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[8] = n.bytes[2];
            n.bytes[9] = try mul(&.{ r, p.u[3], 4 });
        },
        .rope_rms => {
            const rot: usize = p.u[3];
            const stride: usize = p.u[4];
            const offset: usize = p.u[5];
            const head_stride: usize = p.u[6];
            if (s == 0 or rot > s or rot % 2 != 0 or p.f[0] <= 0 or head_stride < s or
                p.u[7] > std.math.maxInt(u32) - r or
                try std.math.add(usize, offset, try std.math.add(usize, try mul(&.{ w - 1, head_stride }), s)) > stride)
                return error.LivingRowsShape;
            n.cells = try mul(&.{ r, w, s });
            n.bytes[0] = try mul(&.{ r, stride, 2 });
            n.bytes[1] = try mul(&.{ s, 4 });
            n.bytes[2] = try mul(&.{ n.cells, 4 });
            n.bytes[3] = try mul(&.{ rot / 2, 4 });
            n.bytes[8] = try mul(&.{ r, stride, 4 });
        },
        .pocket_act, .pocket_add, .pocket_back, .pocket_coeff => {
            if (p.u[12] > 1) return error.LivingRowsShape;
            if (s == 0 or s > 64 or p.u[4] != 64) return error.LivingRowsShape;
            n.bytes[1] = try mul(&.{ 3, 64, w, 4 });
            if (op == .pocket_act or op == .pocket_coeff) {
                n.cells = try mul(&.{ r, s });
                n.bytes[0] = try mul(&.{ r, w, 2 });
                n.bytes[8] = try mul(&.{ n.cells, if (op == .pocket_coeff) @as(usize, 8) else 4 });
                if (op == .pocket_coeff) n.bytes[2] = try mul(&.{ r, w, 4 });
            } else {
                n.bytes[0] = try mul(&.{ r, s, if (op == .pocket_back) @as(usize, 8) else 4 });
                n.bytes[8] = try mul(&.{ n.cells, if (op == .pocket_back) @as(usize, 4) else 2 });
                if (op == .pocket_add) n.bytes[2] = n.bytes[8];
            }
        },
    }
    if (n.cells > std.math.maxInt(u32)) return error.LivingRowsShape;
    for (n.bytes) |bytes| if (bytes > std.math.maxInt(u32)) return error.LivingRowsShape;
    return n;
}
pub fn finish(op: Op, p: Params) !Params {
    var out = p;
    out.u[15] = @intCast((try needs(op, p)).cells);
    return out;
}
pub fn validate(op: Op, p: Params, ranges: [11]wo.Range) !void {
    const n = try needs(op, p);
    if (p.u[15] != n.cells) return error.LivingRowsShape;
    for (ranges, n.bytes, 0..) |v, len, i| {
        if (len == 0) continue;
        const a_float = op == .linear or op == .hc_branch or op == .pocket_add or op == .pocket_back;
        const alignment: usize = if (i == 0 and !a_float) 2 else 4;
        if (v.address == 0 or v.address % alignment != 0 or v.len < len) return error.LivingRowsBuffer;
        _ = try std.math.add(usize, v.address, len);
    }
    for (8..11) |i| {
        if (n.bytes[i] == 0) continue;
        for (0..i) |j| {
            if (n.bytes[j] == 0) continue;
            const x = ranges[i].address;
            const y = ranges[j].address;
            const overlap = if (x <= y) y - x < n.bytes[i] else x - y < n.bytes[j];
            const declared = (op == .nudge and i == 8 and j == 0) or (op == .pocket_add and i == 8 and j == 2);
            if (overlap and !(declared and x == y)) return error.LivingRowsAlias;
        }
    }
}

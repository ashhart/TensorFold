//! Explicit branch/gate rounding policies and buffer bounds for BF16 pointwise HC operators.
const std = @import("std");
pub const Range = @import("weight_only.zig").Range;
pub const Op = enum(c_int) { branch, output_gate };
pub const Rounding = enum(c_int) { native, hf };
pub const Params = extern struct { rows: c_int, hidden: c_int, streams: c_int, op: Op, round_mid: Rounding = .native, gate_stride: c_int = 0, gate_dim: c_int = 0, repeat: c_int = 0 };
pub fn sizes(p: Params) ![4]usize {
    if (p.rows < 1 or p.rows > 128 or p.hidden < 1 or p.streams < 1 or p.streams > 8 or (p.op == .branch and p.streams < 2) or (p.op == .output_gate and p.streams != 1)) return error.HcPointwiseShape;
    const n = try std.math.mul(usize, @intCast(p.rows), @intCast(p.hidden));
    const streams: usize = @intCast(p.streams);
    if (p.hidden > @divTrunc(std.math.maxInt(c_int), p.streams)) return error.HcPointwiseShape;
    const total = try std.math.mul(usize, n, if (p.op == .branch) streams else 1);
    var branch = n;
    if (p.gate_stride != 0 or p.gate_dim != 0) {
        if (p.op != .output_gate or p.gate_stride <= 0 or p.gate_dim <= 0 or p.gate_dim > p.hidden or @mod(p.hidden, p.gate_dim) != 0) return error.HcPointwiseShape;
        const wide = try std.math.mul(usize, @as(usize, @intCast(p.hidden)), 2);
        if (p.gate_stride < wide) return error.HcPointwiseShape;
        branch = try std.math.add(usize, try std.math.mul(usize, @as(usize, @intCast(p.rows - 1)), @intCast(p.gate_stride)), wide);
    }
    return .{ try std.math.mul(usize, total, 2), try std.math.mul(usize, branch, 2), if (p.op == .branch) try std.math.mul(usize, @as(usize, @intCast(p.rows)) * streams, 2) else 2, try std.math.mul(usize, total, 2) };
}
pub fn validate(p: Params, ranges: [4]Range) !usize {
    const needs = try sizes(p);
    for (ranges, needs) |range, n| {
        if (range.address == 0 or range.len < n) return error.HcPointwiseBuffer;
        if (range.address % 2 != 0) return error.HcPointwiseAlignment;
        _ = try std.math.add(usize, range.address, range.len);
    }
    for (0..3) |j| {
        const x = ranges[3];
        const y = ranges[j];
        if (j == 0 and x.address == y.address and x.len == y.len) continue;
        if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.HcPointwiseAlias;
    }
    return needs[3] / 2;
}

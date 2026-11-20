//! Mode-less additive GLPv3 replay ABI; original paired-site concept and format: Matt Suiche/weightless.
const std = @import("std");
const glp = @import("glp_ops.zig");
const replay = @import("kernels_replay.zig");
const Call = replay.Call;
const Tensor = replay.Tensor;

fn typed(t: Tensor, want: replay.DType) !void {
    if (t.dtype != want) return error.WrongArgument;
}
fn integer(c: Call, at: usize) !c_int {
    return std.math.cast(c_int, try c.int(at)) orelse error.WrongArgument;
}
fn matrix(t: Tensor, rows: c_int, width: i64) !void {
    if (t.shape.len != 2 or t.stride.len != 2 or t.stride[1] != 1 or rows < 0 or
        rows > t.shape[0] or t.shape[1] != width) return error.WrongArgument;
}
fn direction(t: Tensor, width: c_int) !void {
    try typed(t, .f32);
    if (width <= 0 or t.shape.len != 1 or t.stride.len != 1 or
        t.shape[0] != width or t.stride[0] != 1) return error.WrongArgument;
}

pub fn glpExt(o: glp.Ops, func: []const u8, args: []const replay.Arg, kwargs: []const replay.Kw) !bool {
    if (replay.is(func, "stream")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "tap", "dir", "alpha", "R", "width" } };
        const x = try c.tensor(0);
        const tap = try c.tensor(1);
        const dir = try c.tensor(2);
        const rows = try integer(c, 4);
        const wide = try integer(c, 5);
        try typed(x, .bf16);
        try direction(dir, wide);
        try matrix(x, rows, 4 * @as(i64, wide));
        const has_tap = replay.numel(tap) != 0;
        if (has_tap) {
            try typed(tap, .bf16);
            try matrix(tap, rows, wide);
        }
        try o.stream(.{ .x = x.ptr, .xs = x.stride[0], .tap = if (has_tap) tap.ptr else 0, .ts = if (has_tap) tap.stride[0] else 0, .dir = dir.ptr, .alpha = @floatCast(try c.float(3)), .R = rows, .width = wide });
        return true;
    }
    if (replay.is(func, "writer")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "g", "dir", "alpha", "R", "width" } };
        const g = try c.tensor(0);
        const dir = try c.tensor(1);
        const rows = try integer(c, 3);
        const wide = try integer(c, 4);
        try typed(g, .bf16);
        try direction(dir, wide);
        if (g.shape.len != 3 or g.stride.len != 3 or g.stride[2] != 1 or rows < 0 or
            rows > g.shape[1] or wide != g.shape[2] or g.shape[0] < 1 or g.shape[0] > glp.max_world) return error.WrongArgument;
        try o.writer(.{ .g = g.ptr, .gr = g.stride[0], .gs = g.stride[1], .dir = dir.ptr, .alpha = @floatCast(try c.float(2)), .R = rows, .width = wide, .world = @intCast(g.shape[0]) });
        return true;
    }
    if (replay.is(func, "final")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "pre", "nw", "out", "eps", "R", "width" } };
        const x = try c.tensor(0);
        const pre = try c.tensor(1);
        const nw = try c.tensor(2);
        const out = try c.tensor(3);
        const rows = try integer(c, 5);
        const wide = try integer(c, 6);
        try typed(x, .bf16);
        try typed(pre, .f32);
        try typed(out, .bf16);
        if (wide <= 0 or (nw.dtype != .f32 and nw.dtype != .bf16)) return error.WrongArgument;
        try matrix(x, rows, 4 * @as(i64, wide));
        try matrix(pre, rows, 4);
        try matrix(out, rows, wide);
        if (pre.stride[0] != 4 or out.stride[0] != wide or nw.shape.len != 1 or nw.stride.len != 1 or
            nw.shape[0] != wide or nw.stride[0] != 1) return error.WrongArgument;
        try o.final(.{ .x = x.ptr, .xs = x.stride[0], .pre = pre.ptr, .nw = nw.ptr, .out = out.ptr, .eps = @floatCast(try c.float(4)), .nwbf16 = @intFromBool(nw.dtype == .bf16), .R = rows, .width = wide });
        return true;
    }
    return false;
}

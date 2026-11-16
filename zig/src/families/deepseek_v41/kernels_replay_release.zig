//! Runs the release projections' calls (release_dense.zig's `tf_dsv41_rel_v1`) through kernels_release.zig's launches.

const std = @import("std");
const replay = @import("kernels_replay.zig");
const kr = @import("kernels_release.zig");

const Arg = replay.Arg;
const Kw = replay.Kw;
const Call = replay.Call;
const Tensor = replay.Tensor;

fn is(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn rows(t: Tensor) !c_int {
    if (t.shape.len != 2) return error.WrongArgument;
    return @intCast(t.shape[0]);
}

fn cols(t: Tensor) !c_int {
    if (t.shape.len != 2 or t.stride[1] != 1) return error.WrongArgument;
    return @intCast(t.shape[1]);
}

/// y's element type as the launch's flag: 1 for FP32, 0 for BF16.
fn f32Of(y: Tensor) !c_int {
    return switch (y.dtype) {
        .f32 => 1,
        .bf16 => 0,
        else => error.WrongArgument,
    };
}

/// One `tf_dsv41_rel_v1` call: actq (x, codes, scales), fp8 (codes, scales, w, ws, y, col, part) or bf16.
pub fn call(o: kr.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "actq")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "codes", "scales" } };
        const x = try c.tensor(0);
        if (x.dtype != .bf16) return error.WrongArgument;
        try o.actq(.{ .x = x.ptr, .x_ld = x.stride[0], .codes = try c.ptr(1), .scales = try c.ptr(2), .m = try rows(x), .k = try cols(x) });
        return true;
    }
    if (is(func, "fp8")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "codes", "scales", "w", "ws", "y", "col", "part" } };
        const a = try c.tensor(0);
        const sa = try c.tensor(1);
        const w = try c.tensor(2);
        const y = try c.tensor(4);
        try o.fp8(.{
            .a = a.ptr,
            .a_ld = a.stride[0],
            .sa = sa.ptr,
            .sa_ld = sa.stride[0],
            .w = w.ptr,
            .ws = try c.ptr(3),
            .y = y.ptr,
            .y_ld = y.stride[0],
            .y_col = try c.int(5),
            .part = try c.ptrOrNull(6),
            .m = try rows(a),
            .n = try rows(w),
            .k = try cols(w),
            .f32 = try f32Of(y),
        });
        return true;
    }
    if (is(func, "bf16")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "w", "y", "col", "part" } };
        const x = try c.tensor(0);
        const w = try c.tensor(1);
        const y = try c.tensor(2);
        if (x.dtype != .bf16 or w.dtype != .bf16) return error.WrongArgument;
        try o.bf16(.{
            .a = x.ptr,
            .a_ld = x.stride[0],
            .w = w.ptr,
            .y = y.ptr,
            .y_ld = y.stride[0],
            .y_col = try c.int(3),
            .part = try c.ptrOrNull(4),
            .m = try rows(x),
            .n = try rows(w),
            .k = try cols(w),
            .f32 = try f32Of(y),
        });
        return true;
    }
    return error.Unsupported;
}

//! Runs the release projections' calls (release_dense.zig's `tf_dsv41_rel_v1`) through kernels_release.zig's launches.

const std = @import("std");
const replay = @import("kernels_replay.zig");
const kr = @import("kernels_release.zig");

const Arg = replay.Arg;
const Kw = replay.Kw;
const Call = replay.Call;
const Tensor = replay.Tensor;
const numel = replay.numel;

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

/// One `tf_dsv41_rel_v1` call: actq, fp8, bf16 (release_dense.zig); group, gateup, down, swiglu, combine (release_moe).
pub fn call(o: kr.Ops, ex: kr.ex.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
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
    if (is(func, "group")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "ids", "slot_of", "offs", "pair_tok", "pair_of", "active", "nactive" } };
        const ids = try c.tensor(0);
        const slot_of = try c.tensor(1);
        const offs = try c.tensor(2);
        try ex.group(.{
            .ids = ids.ptr,
            .slot_of = slot_of.ptr,
            .offs = offs.ptr,
            .pair_tok = try c.ptr(3),
            .pair_of = try c.ptr(4),
            .active = try c.ptr(5),
            .nactive = try c.ptr(6),
            .m = try rows(ids),
            .topk = try cols(ids),
            .nslots = @intCast(offs.shape[0] - 1),
            .nids = @intCast(slot_of.shape[0]),
        });
        return true;
    }
    if (is(func, "gateup")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "codes", "scales", "w1", "s1", "w3", "s3", "wts", "pair_tok", "offs", "active", "nactive", "out", "out_scales" } };
        const a = try c.tensor(0);
        const sa = try c.tensor(1);
        const w1 = try c.tensor(2);
        const wts = try c.tensor(6);
        if (w1.shape.len != 3) return error.WrongArgument;
        try ex.gateUp(.{
            .x = a.ptr,
            .x_ld = a.stride[0],
            .xs = sa.ptr,
            .xs_ld = sa.stride[0],
            .w1 = w1.ptr,
            .s1 = try c.ptr(3),
            .w3 = try c.ptr(4),
            .s3 = try c.ptr(5),
            .wts = wts.ptr,
            .pair_tok = try c.ptr(7),
            .offs = try c.ptr(8),
            .active = try c.ptr(9),
            .nactive = try c.ptr(10),
            .codes = try c.ptr(11),
            .scales = try c.ptr(12),
            .n = @intCast(w1.shape[1]),
            .k = @intCast(w1.shape[2] * 2),
            .topk = try cols(wts),
        }, @intCast(a.shape[0]), @intCast(w1.shape[0]), .served);
        return true;
    }
    if (is(func, "down")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "codes", "scales", "w2", "s2", "offs", "active", "nactive", "part" } };
        const a = try c.tensor(0);
        const w2 = try c.tensor(2);
        if (w2.shape.len != 3) return error.WrongArgument;
        // pairs = rows of the codes: the routed launch sizes by pairs, so one "row" a pair at top-k 1
        try ex.down(.{
            .x = a.ptr,
            .xs = try c.ptr(1),
            .w2 = w2.ptr,
            .s2 = try c.ptr(3),
            .offs = try c.ptr(4),
            .active = try c.ptr(5),
            .nactive = try c.ptr(6),
            .part = try c.ptr(7),
            .n = @intCast(w2.shape[1]),
            .k = @intCast(w2.shape[2] * 2),
        }, @intCast(a.shape[0]), 1, @intCast(w2.shape[0]), .served);
        return true;
    }
    if (is(func, "swiglu")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "gate", "up", "codes", "scales" } };
        const g = try c.tensor(0);
        const u = try c.tensor(1);
        try ex.swigluActq(.{ .gate = g.ptr, .gate_ld = g.stride[0], .up = u.ptr, .up_ld = u.stride[0], .codes = try c.ptr(2), .scales = try c.ptr(3), .m = try rows(g), .n = try cols(g) });
        return true;
    }
    if (is(func, "combine")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "part", "pair_of", "shared", "y", "col" } };
        const part = try c.tensor(0);
        const pair_of = try c.tensor(1);
        const shared = try c.tensor(2);
        const y = try c.tensor(3);
        const has_shared = numel(shared) != 0;
        try ex.combine(.{
            .part = part.ptr,
            .pair_of = pair_of.ptr,
            .shared = if (has_shared) shared.ptr else 0,
            .shared_ld = if (has_shared) shared.stride[0] else 0,
            .y = y.ptr,
            .y_ld = y.stride[0],
            .y_col = try c.int(4),
            .m = try rows(pair_of),
            .n = @intCast(part.shape[1]),
            .topk = try cols(pair_of),
            .f32 = try f32Of(y),
        });
        return true;
    }
    return error.Unsupported;
}

//! Chunked SSD arithmetic from the pinned mlx-lm models/ssm.py.
const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const Ops = @import("prefill_ops.zig").Ops;

fn repeat(s: *mx.Scope, x: A, count: i32, axis: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_repeat_axis(&out, x, count, axis, mx.stream);
    return s.result(rc, out);
}
fn tril(s: *mx.Scope, x: A, diagonal: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_tril(&out, x, diagonal, mx.stream);
    return s.result(rc, out);
}
fn cumulative(s: *mx.Scope, x: A, axis: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_cumsum(&out, x, axis, false, true, mx.stream);
    return s.result(rc, out);
}
fn shapeIs(x: A, shape: []const i32) bool {
    return x.ctx != null and std.mem.eql(i32, mx.shape(x), shape);
}

/// Unmasked prompt chunk. State is [batch, heads, head_dim, state_dim].
pub fn forward(ops: *Ops, s: *mx.Scope, x: A, a_log: A, b: A, cc: A, d: A, dt: A, dt_bias: A, initial: A, limits: [2]f32) ![2]A {
    if (x.ctx == null or mx.shape(x).len != 4 or b.ctx == null or mx.shape(b).len != 4) return error.InvalidTensorShape;
    const batch = mx.dim(x, 0);
    const length = mx.dim(x, 1);
    const heads = mx.dim(x, 2);
    const dims = mx.dim(x, 3);
    const groups = mx.dim(b, 2);
    const state_dim = mx.dim(b, 3);
    if (batch < 1 or length < 1 or length > 2048 or heads < 1 or dims < 1 or groups < 1 or state_dim < 1 or @mod(heads, groups) != 0) return error.InvalidTensorShape;
    if (!shapeIs(b, &.{ batch, length, groups, state_dim }) or !shapeIs(cc, mx.shape(b)) or !shapeIs(a_log, &.{heads}) or !shapeIs(d, &.{heads}) or !shapeIs(dt_bias, &.{heads}) or !shapeIs(dt, &.{ batch, length, heads })) return error.InvalidTensorShape;
    if (initial.ctx != null and !shapeIs(initial, &.{ batch, heads, dims, state_dim })) return error.InvalidTensorShape;
    if (std.math.isNan(limits[0]) or std.math.isNan(limits[1]) or limits[0] < 0 or limits[0] > limits[1]) return error.InvalidTimeStepLimits;
    const copies = @divExact(heads, groups);
    const delta = try ops.call(s, .ssm_dt, &.{ dt, dt_bias, try s.scalar(limits[0]), try s.scalar(limits[1]) });
    const a = try s.cast(try s.unary(c.mlx_negative, try s.unary(c.mlx_exp, a_log)), mx.dtype(delta));
    const delta_a = try s.binary(c.mlx_multiply, delta, try s.reshape(a, &.{ 1, 1, heads }));
    const delta_x = try s.binary(c.mlx_multiply, try s.reshape(delta, &.{ batch, length, heads, 1 }), x);
    var state = initial;
    var outputs: [8]A = undefined;
    var count: usize = 0;
    var start: i32 = 0;
    while (start < length) : (start += 256) {
        const end = @min(length, start + 256);
        const rows = end - start;
        const dx = try s.slice(delta_x, 1, start, end);
        const da = try s.slice(delta_a, 1, start, end);
        const cb = try s.slice(cc, 1, start, end);
        var bb = try s.transpose(try s.slice(b, 1, start, end), &.{ 0, 2, 3, 1 });
        const product = try repeat(s, try s.binary(c.mlx_matmul, try s.transpose(cb, &.{ 0, 2, 1, 3 }), bb), copies, 1);
        const transitions = try s.transpose(da, &.{ 0, 2, 1 });
        const tiled = try repeat(s, try s.reshape(transitions, &.{ batch, heads, rows, 1 }), rows, -1);
        const decay = try s.unary(c.mlx_exp, try cumulative(s, try tril(s, tiled, -1), -2));
        const attention = try tril(s, try s.binary(c.mlx_multiply, product, decay), 0);
        var y = try s.transpose(try s.binary(c.mlx_matmul, attention, try s.transpose(dx, &.{ 0, 2, 1, 3 })), &.{ 0, 2, 1, 3 });
        const last_decay = try s.transpose(try s.slice(decay, 2, rows - 1, rows), &.{ 0, 3, 1, 2 });
        bb = try s.transpose(try repeat(s, bb, copies, 1), &.{ 0, 1, 3, 2 });
        const decayed = try s.transpose(try s.binary(c.mlx_multiply, dx, last_decay), &.{ 0, 2, 3, 1 });
        var next = try s.binary(c.mlx_matmul, decayed, bb);
        if (state.ctx != null) {
            const accumulated = try s.unary(c.mlx_exp, try cumulative(s, da, -2));
            const last = try s.reshape(try s.slice(accumulated, 1, rows - 1, rows), &.{ batch, heads, 1, 1 });
            next = try s.binary(c.mlx_add, next, try s.binary(c.mlx_multiply, last, state));
            const prior = try s.binary(c.mlx_matmul, try s.reshape(state, &.{ batch, 1, groups, copies, dims, state_dim }), try s.reshape(cb, &.{ batch, rows, groups, 1, state_dim, 1 }));
            y = try s.binary(c.mlx_add, y, try s.binary(c.mlx_multiply, try s.reshape(accumulated, &.{ batch, rows, heads, 1 }), try s.reshape(prior, &.{ batch, rows, heads, dims })));
        }
        outputs[count] = try s.cast(y, mx.dtype(x));
        count += 1;
        state = next;
    }
    const skip = try s.binary(c.mlx_multiply, x, try s.reshape(d, &.{ 1, 1, heads, 1 }));
    return .{ try s.binary(c.mlx_add, try s.cat(outputs[0..count], 1), skip), state };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var weights = @import("checkpoint.zig").Store.init(64);
    defer weights.deinit();
    var path: [4096]u8 = undefined;
    try weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/ssm.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/ssm.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { key: []const u8, state: bool, limits: [2]f32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var ops = Ops{};
    defer ops.deinit();
    for (cases.value) |case| {
        errdefer std.debug.print("SSD fixture {s}\n", .{case.key});
        var s = mx.Scope{};
        defer s.deinit();
        var inputs: [8]A = undefined;
        for (&inputs, 0..) |*v, j| v.* = try weights.field(case.key, try std.fmt.bufPrint(&path, "input{d}", .{j}));
        const result = try forward(&ops, &s, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], if (case.state) inputs[7] else mx.empty, case.limits);
        try @import("sampling_checks.zig").equal(&s, result[0], try weights.field(case.key, "output"));
        try @import("sampling_checks.zig").equal(&s, result[1], try weights.field(case.key, "state"));
    }
    std.debug.print("PASS: {d} chunked SSD prompt outputs and recurrent states match pinned mlx-lm exactly.\n", .{cases.value.len});
}

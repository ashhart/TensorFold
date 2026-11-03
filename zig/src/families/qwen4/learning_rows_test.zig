//! The actual shared device body is instrumented on the host and compared with independent binary64 slopes.
const std = @import("std");
const rows = @import("core").living_rows;
const bf = @import("living").affine;
const a = std.testing.allocator;
extern fn tf_living_rows_cpu(c_uint, rows.Params, [*][*]u8) void;
const Case = struct {
    p: rows.Params,
    op: rows.Op,
    data: [11][]align(16) u8,
    fn init(op: rows.Op, p: rows.Params) !Case {
        var out: Case = .{ .op = op, .p = try rows.finish(op, p), .data = undefined };
        const needed = try rows.needs(op, p);
        var count: usize = 0;
        errdefer for (out.data[0..count]) |v| a.free(v);
        for (&out.data, needed.bytes) |*v, len| {
            v.* = try a.alignedAlloc(u8, .@"16", @max(len, 16));
            @memset(v.*, 0);
            count += 1;
        }
        return out;
    }
    fn deinit(c: *Case) void {
        for (c.data) |v| a.free(v);
    }
    fn run(c: *Case) !void {
        var ranges: [11]@import("core").weight_only.Range = undefined;
        var ptrs: [11][*]u8 = undefined;
        for (c.data, &ranges, &ptrs) |v, *r, *p| {
            r.* = .{ .address = @intFromPtr(v.ptr), .len = v.len };
            p.* = v.ptr;
        }
        try rows.validate(c.op, c.p, ranges);
        tf_living_rows_cpu(@intCast(@backingInt(c.op)), c.p, &ptrs);
    }
    fn floats(c: *Case, i: usize) []f32 {
        return std.mem.bytesAsSlice(f32, c.data[i]);
    }
    fn putBf(c: *Case, i: usize, values: []const f32) void {
        const out = std.mem.bytesAsSlice(u16, c.data[i]);
        for (values, out[0..values.len]) |v, *x| x.* = bf.bf16of(v);
    }
};
fn near(got: f32, want: f64) !void {
    if (!std.math.isFinite(got) or @abs(@as(f64, got) - want) > 2e-5 + 2e-5 * @abs(want)) return error.NativeRowGradientMismatch;
}
fn sig(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
fn silu(x: f64) f64 {
    return x * sig(x);
}
fn rounded(x: f64) f64 {
    return bf.f32of(bf.bf16of(@floatCast(x)));
}
fn value(bytes: []const u8, at: usize) f64 {
    return bf.f32of(std.mem.readInt(u16, bytes[2 * at ..][0..2], .little));
}
test "Qwen4 learning: native row body dense and affine input pullbacks sum every output row" {
    inline for (.{ false, true }) |affine| {
        var c = try Case.init(.linear, .{ .u = .{ 2, 3, 64, @intFromBool(affine), 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } });
        defer c.deinit();
        @memcpy(c.floats(0)[0..6], &[_]f32{ 0.5, -0.75, 1.25, -0.5, 1, 0.25 });
        if (affine) {
            const words = std.mem.bytesAsSlice(u32, c.data[1]);
            for (words, 0..) |*w, i| w.* = @as(u32, @intCast(i % 16)) * 0x11111111;
            c.putBf(2, &.{ 0.125, 0.25, 0.5 });
            c.putBf(3, &.{ -0.5, 0.75, -1 });
        } else {
            const w = std.mem.bytesAsSlice(u16, c.data[1]);
            for (w[0..192], 0..) |*v, i| v.* = bf.bf16of(@as(f32, @floatFromInt(i % 13)) / 16 - 0.25);
        }
        try c.run();
        for (0..128) |i| {
            var want: f64 = 0;
            for (0..3) |o| {
                const at = o * 64 + i % 64;
                const weight = if (affine) blk: {
                    const word = std.mem.readInt(u32, c.data[1][at / 8 * 4 ..][0..4], .little);
                    const code = word >> @as(u5, @intCast(at % 8 * 4)) & 15;
                    break :blk @as(f64, @floatFromInt(code)) * value(c.data[2], o) + value(c.data[3], o);
                } else value(c.data[1], at);
                want += weight * @as(f64, c.floats(0)[i / 64 * 3 + o]);
            }
            try near(c.floats(8)[i], want);
        }
    }
}
test "Qwen4 learning: native HC activation and dense SwiGLU derivatives keep their forward cuts" {
    const gates = [_]f32{ -3, -0.5, 0.25, 2 };
    const ups = [_]f32{ 0.5, -1, 2, 0.75 };
    inline for (.{ false, true }) |injection| {
        var c = try Case.init(.hc_act, .{ .u = .{ 1, 4, 2, @intFromBool(injection), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } });
        defer c.deinit();
        c.putBf(0, &gates);
        @memcpy(c.floats(1)[0..4], &ups);
        try c.run();
        for (gates, ups, 0..) |gate, grad, i| {
            const x: f64 = gate;
            const offset = rounded(x / 2) - x / 2;
            inline for (.{ 1e-3, 1e-4 }) |eps| {
                const hi = (x + eps) / 2 + offset;
                const lo = (x - eps) / 2 + offset;
                const slope = @as(f64, grad) * (if (injection) 2 * (sig(hi) - sig(lo)) else silu(hi) - silu(lo)) / (2 * eps);
                try near(c.floats(8)[i], slope);
            }
        }
    }
    var c = try Case.init(.swiglu, .{ .u = .{ 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } });
    defer c.deinit();
    c.putBf(0, &gates);
    c.putBf(1, &ups);
    @memset(c.floats(2), 1);
    try c.run();
    for (gates, ups, 0..) |gate, up, i| {
        const x: f64 = gate;
        inline for (.{ 1e-3, 1e-4 }) |eps| {
            try near(c.floats(8)[i], @as(f64, up) * (silu(x + eps) - silu(x - eps)) / (2 * eps));
            try near(c.floats(9)[i], rounded(silu(x)));
        }
    }
}
fn rmsLoss(xs: []const f64, scales: []const f32, gradient: []const f32, width: usize) f64 {
    var result: f64 = 0;
    for (0..xs.len / width) |row| {
        var sq: f64 = 0;
        for (xs[row * width ..][0..width]) |x| sq += x * x;
        const inv = 1 / @sqrt(sq / @as(f64, @floatFromInt(width)) + 1e-6);
        for (0..width) |i| result += xs[row * width + i] * inv * @as(f64, scales[row * width + i]) * @as(f64, gradient[row * width + i]);
    }
    return result;
}
test "Qwen4 learning: native grouped HC norm adjoint agrees with binary64 central slopes" {
    var c = try Case.init(.rms, .{ .u = .{ 1, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, .f = .{ 1e-6, 0, 0, 0 } });
    defer c.deinit();
    const xs = [_]f32{ -0.5, 1, -2, 0.75, 2, 0.25, -0.5, 1.5 };
    c.putBf(0, &xs);
    @memcpy(c.floats(1)[0..8], &[_]f32{ 0.5, 1, 1.25, -0.5, 1.5, 0.5, 0.75, 1 });
    @memcpy(c.floats(2)[0..8], &[_]f32{ 1, -0.5, 0.75, 0.25, -1, 0.75, 0.5, 1.25 });
    try c.run();
    var perturbed: [8]f64 = undefined;
    for (&perturbed, xs) |*x, v| x.* = v;
    for (0..8) |i| inline for (.{ 1e-3, 1e-4 }) |eps| {
        perturbed[i] = @as(f64, xs[i]) + eps;
        const hi = rmsLoss(&perturbed, c.floats(1)[0..8], c.floats(2)[0..8], 4);
        perturbed[i] = @as(f64, xs[i]) - eps;
        const lo = rmsLoss(&perturbed, c.floats(1)[0..8], c.floats(2)[0..8], 4);
        perturbed[i] = xs[i];
        try near(c.floats(8)[i], (hi - lo) / (2 * eps));
    };
}

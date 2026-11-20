//! Control vectors: the operator's parameters, its float64 reference, the bounds the gates check and its launch.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");
const bf16 = @import("bf16_round");

/// The hidden width this family's residual stream has.
pub const hidden = 5120;

/// The elements a thread of the fused norm holds, so the row's squares are spread over `hidden / elements` threads.
pub const elements = 16;

/// The threads a row is spread over, as the kernel launches it.
pub const threads = hidden / elements;

/// The threadgroup memory the kernel's one set of partials takes, in the 16-byte units the Metal compiler allocates.
pub const group_memory = std.mem.alignForward(usize, (threads / 32) * 4, 16);

/// The operator's parameters: the strength, the rows and the hidden width.
pub const Params = struct {
    alpha: f32 = 1.0,
    rows: usize = 1,
    hidden: usize = hidden,
};

/// One call's inputs: the residual rows in bf16 and the direction in fp32, with the float64 reference beside them.
pub const Steer = struct {
    a: std.mem.Allocator,
    p: Params,
    x: []u16,
    dir: []f32,

    /// A call's rows and a direction as wide as they are.
    pub fn init(a: std.mem.Allocator, p: Params) !Steer {
        if (p.hidden == 0 or p.rows == 0) return error.BadDirectionWidth;
        const x = try a.alloc(u16, p.rows * p.hidden);
        errdefer a.free(x);
        return .{ .a = a, .p = p, .x = x, .dir = try a.alloc(f32, p.hidden) };
    }

    /// A call whose direction is the caller's width: a width that does not match the rows is refused.
    pub fn withDirection(a: std.mem.Allocator, p: Params, dir: []const f32) !Steer {
        if (p.hidden == 0 or p.rows == 0) return error.BadDirectionWidth;
        const x = try a.alloc(u16, p.rows * p.hidden);
        errdefer a.free(x);
        return .{ .a = a, .p = p, .x = x, .dir = try a.dupe(f32, dir) };
    }

    pub fn deinit(s: Steer) void {
        s.a.free(s.x);
        s.a.free(s.dir);
    }

    /// The outputs a call writes: the rows the hidden width.
    pub fn outputs(p: Params) usize {
        return p.rows * p.hidden;
    }

    /// The inputs a gate asks for: ordinary rows with a direction, or rows that hold NaN and infinities.
    pub fn fill(s: *Steer, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (s.dir) |*d| d.* = rng.float(f32) * 2 - 1;
        for (s.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random, .stress => {},
            .nonfinite => s.plant(),
        }
    }

    /// A NaN and both infinities, where float64 and the kernel must agree on the class.
    fn plant(s: *Steer) void {
        const k = s.p.hidden;
        if (k < 4) return;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        for (at, put) |p, v| if (p[0] < s.p.rows) {
            s.x[p[0] * k + p[1]] = v;
        };
    }

    /// The float64 outputs and, per output, the magnitude a sum's error would carry into it: there are no sums.
    pub fn reference(s: Steer, want: []f64, mag: []f64) !void {
        if (s.dir.len != s.p.hidden) return error.BadDirectionWidth;
        const k = s.p.hidden;
        for (0..s.p.rows) |r| {
            for (0..k) |i| {
                const x = c.fromBf16(s.x[r * k + i]);
                const g: f64 = s.dir[i];
                want[r * k + i] = x + @as(f64, s.p.alpha) * g;
                mag[r * k + i] = 0;
            }
        }
    }

    /// The kernel's own arithmetic: the product and the sum in fp32, then the stream's bf16 rounding.
    pub fn kernelOut(s: Steer, out: []f64) void {
        const k = s.p.hidden;
        for (0..s.p.rows) |r| {
            for (0..k) |i| {
                const x = c.fromBf16(s.x[r * k + i]);
                const v: f32 = x + s.p.alpha * s.dir[i];
                out[r * k + i] = bf16.round32(v);
            }
        }
    }

    /// The allowance per output: the product's and the sum's fp32 roundings, then the output's own rounding.
    pub fn bounds(s: Steer, want: []const f64, allow: []f64) void {
        const u = 0x1p-24;
        const k = s.p.hidden;
        const scale = @abs(@as(f64, s.p.alpha));
        for (0..s.p.rows) |r| {
            for (0..k) |i| {
                const w = want[r * k + i];
                if (!std.math.isFinite(w)) {
                    allow[r * k + i] = 0; // the nonfinite gate compares classes, not distances
                    continue;
                }
                const prod = scale * @abs(@as(f64, s.dir[i]));
                const work = u * (prod + @abs(w));
                allow[r * k + i] = work + 0x1p-8 * (@abs(w) + work) + 0x1p-133;
            }
        }
    }
};

/// The kernel has no sums to round.
pub fn depthNone(_: c.Shape) usize {
    return 0;
}

/// The kernel's registration: the served shapes and the bound of its one product and one sum.
pub fn spec() c.Spec {
    return .{
        .name = "qwen27 steer",
        .accum = .none,
        .depth = depthNone,
        .ops = 2,
        .out = .bf16,
        .in_place = false,
        .tiles = .{
            .{ .size = 8, .step = 8, .min = 8 },
            .{ .size = hidden, .step = hidden, .min = hidden },
            .{ .size = hidden, .step = hidden, .min = hidden },
        },
        .served = &.{
            .{ .rows = 1, .n = hidden, .k = hidden },
            .{ .rows = 8, .n = hidden, .k = hidden },
            .{ .rows = 64, .n = hidden, .k = hidden },
        },
        .sums = .{
            .why = "Each output is one product and one sum of two values, with no iterative accumulation.",
        },
    };
}

test "the kernel's own arithmetic lands inside the bound the spec declares" {
    const a = std.testing.allocator;
    var s = try Steer.init(a, .{ .alpha = 0.75, .rows = 8 });
    defer s.deinit();
    s.fill(.random, 11);
    const n = Steer.outputs(s.p);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const got = try a.alloc(f64, n);
    defer a.free(got);
    const allow = try a.alloc(f64, n);
    defer a.free(allow);
    try s.reference(want, mag);
    s.kernelOut(got);
    s.bounds(want, allow);
    for (got, want, allow) |g, w, l| try std.testing.expect(@abs(g - w) <= l);
}

test "the reference adds the strength times the direction, whatever the direction's length" {
    const a = std.testing.allocator;
    var s = try Steer.withDirection(a, .{ .alpha = 2, .rows = 1, .hidden = 2 }, &.{ 3, 4 });
    defer s.deinit();
    s.x[0] = c.toBf16(1);
    s.x[1] = c.toBf16(0);
    var want: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    try s.reference(&want, &mag);
    try std.testing.expectEqual(@as(f64, 7), want[0]); // 1 + 2 * 3
    try std.testing.expectEqual(@as(f64, 8), want[1]); // 0 + 2 * 4
}

test "a NaN in a row stays in its own element, as float64 does" {
    const a = std.testing.allocator;
    var s = try Steer.init(a, .{ .alpha = 1, .rows = 5, .hidden = 64 });
    defer s.deinit();
    s.fill(.nonfinite, 3);
    const n = Steer.outputs(s.p);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    const got = try a.alloc(f64, n);
    defer a.free(got);
    try s.reference(want, mag);
    s.kernelOut(got);
    try std.testing.expect(std.math.isNan(want[21]));
    try std.testing.expect(std.math.isFinite(want[0]));
    for (got, want) |g, w| try std.testing.expectEqual(c.classOf(w), c.classOf(g));
}

test "a zero-length or mismatched direction is refused" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BadDirectionWidth, Steer.init(a, .{ .rows = 1, .hidden = 0 }));
    var s = try Steer.withDirection(a, .{ .rows = 1, .hidden = 4 }, &.{ 1, 2, 3 });
    defer s.deinit();
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    try std.testing.expectError(error.BadDirectionWidth, s.reference(&want, &mag));
}

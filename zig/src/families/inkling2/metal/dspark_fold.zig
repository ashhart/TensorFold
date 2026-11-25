//! One-time binary64 folding of DSpark's rank-256 hidden head into the target unembedding.
const std = @import("std");
const bf = @import("bf16_round");

pub const hidden = 2304;
pub const rank = 256;
pub const vocabulary = 201024;

pub fn value(h: u16) f64 {
    const x: f32 = @bitCast(@as(u32, h) << 16);
    return x;
}

/// The reference's binary64 dot, rounded once through binary32 to BF16.
pub fn dot(u: []const f64, w: []const f64) u16 {
    std.debug.assert(u.len == w.len);
    var sum: f64 = 0;
    for (u, w) |x, y| sum += x * y;
    return bf.bits(@floatCast(sum));
}

const Work = struct {
    unembed: []const u16,
    hidden_w: []const f64,
    hidden_b: []const f64,
    phi: []u16,
    beta: []u16,
    v_stride: usize,
    begin: usize,
    end: usize,

    fn run(w: *const Work) void {
        var u: [hidden]f64 = undefined;
        for (w.begin..w.end) |v| {
            for (&u, w.unembed[v * hidden ..][0..hidden]) |*x, h| x.* = value(h);
            for (0..rank) |r| w.phi[r * w.v_stride + v] = dot(&u, w.hidden_w[r * hidden ..][0..hidden]);
            w.beta[v] = dot(&u, w.hidden_b);
        }
    }
};

/// Φ[r,v] = bf16(f32(f64(U[v]) · f64(hidden_out[:,r]))), β[v] likewise for its bias.
pub fn fold(
    a: std.mem.Allocator,
    unembed: []const u16,
    hidden_weight: []const u16,
    hidden_bias: []const u16,
    phi: []u16,
    beta: []u16,
) !void {
    const v = beta.len;
    if (unembed.len != v * hidden or hidden_weight.len != hidden * rank or hidden_bias.len != hidden or phi.len != v * rank or v == 0)
        return error.DSparkFoldShape;
    const hw = try a.alloc(f64, hidden * rank);
    defer a.free(hw);
    const hb = try a.alloc(f64, hidden);
    defer a.free(hb);
    for (hidden_weight, 0..) |h, i| hw[(i % rank) * hidden + i / rank] = value(h);
    for (hb, hidden_bias) |*x, h| x.* = value(h);

    const count = @min(@as(usize, 8), v);
    const chunk = (v + count - 1) / count;
    var work: [8]Work = undefined;
    var threads: [8]?std.Thread = @splat(null);
    for (0..count) |i| {
        const begin = @min(i * chunk, v);
        const end = @min(begin + chunk, v);
        work[i] = .{ .unembed = unembed, .hidden_w = hw, .hidden_b = hb, .phi = phi, .beta = beta, .v_stride = v, .begin = begin, .end = end };
        threads[i] = std.Thread.spawn(.{}, Work.run, .{&work[i]}) catch null;
        if (threads[i] == null) work[i].run();
    }
    for (threads) |thread| if (thread) |t| t.join();
}

test "the binary64 fold does not collapse a discriminating binary32 intermediate" {
    var u: [4]f64 = @splat(0);
    u[0] = 1;
    u[1] = 0x1p-8;
    u[2] = 0x1p-24;
    u[3] = 0x1p-24;
    const w: [4]f64 = @splat(1);
    const got = bf.value(dot(&u, &w));
    try std.testing.expectEqual(@as(f32, 1.0078125), got);
    var sum: f32 = 0;
    for (u, w) |x, y| sum += @as(f32, @floatCast(x * y));
    try std.testing.expectEqual(@as(f32, 1), bf.value(bf.bits(sum)));
}

test "fold checks every published shape" {
    var u: [hidden]u16 = @splat(0);
    var w: [hidden * rank]u16 = @splat(0);
    var p: [rank]u16 = undefined;
    var b: [1]u16 = undefined;
    try std.testing.expectError(error.DSparkFoldShape, fold(std.testing.allocator, &.{}, &w, &u, &p, &b));
    try fold(std.testing.allocator, &u, &w, &u, &p, &b);
    for (p) |x| try std.testing.expectEqual(@as(u16, 0), x);
    try std.testing.expectEqual(@as(u16, 0), b[0]);
    u[1] = bf.bits(1);
    w[rank + 2] = bf.bits(1); // hidden row 1, rank column 2 in the checkpoint's row-major layout
    try fold(std.testing.allocator, &u, &w, &u, &p, &b);
    try std.testing.expectEqual(bf.bits(1), p[2]);
    try std.testing.expectEqual(bf.bits(1), b[0]);
    for (p, 0..) |x, r| if (r != 2) try std.testing.expectEqual(@as(u16, 0), x);
    var u_two: [2 * hidden]u16 = @splat(0);
    var p_two: [2 * rank]u16 = undefined;
    var b_two: [2]u16 = undefined;
    u_two[hidden + 1] = bf.bits(1);
    try fold(std.testing.allocator, &u_two, &w, &u, &p_two, &b_two);
    try std.testing.expectEqual(bf.bits(1), p_two[2 * 2 + 1]);
    try std.testing.expectEqual(@as(u16, 0), p_two[2 * 2]);
    try std.testing.expectEqual(bf.bits(1), b_two[1]);
}

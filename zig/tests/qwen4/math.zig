//! CPU arithmetic fixtures for the shared graph, independent of the binary64 Python oracle.
const std = @import("std");
pub fn bf(x: f32) u16 {
    return @import("core").quant.convert.bf16(x);
}
pub fn f(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn round(x: f32) f32 {
    return f(bf(x));
}
pub fn sig(x: f32) f32 {
    return if (x >= 0) 1 / (1 + @exp(-x)) else @exp(x) / (1 + @exp(x));
}
pub fn silu(x: f32) f32 {
    return x * sig(x);
}
pub const Tensor = struct {
    bits: []u16,
    rows: usize,
    width: usize,
    pub fn row(t: Tensor, r: usize) []u16 {
        return t.bits[r * t.width ..][0..t.width];
    }
};
pub fn tensor(a: std.mem.Allocator, rows: usize, width: usize) !Tensor {
    return .{ .bits = try a.alloc(u16, rows * width), .rows = rows, .width = width };
}
pub fn source(a: std.mem.Allocator, name: []const u8, n: usize, norm_weight: bool, gated_norm: bool) ![]u16 {
    const out = try a.alloc(u16, n);
    var hash: u32 = 2166136261;
    for (name) |byte| hash = (hash ^ byte) *% 16777619;
    for (out, 0..) |*v, i| {
        const code = (hash + 17 * @as(u64, @intCast(i)) + 3 * @as(u64, @intCast(i)) * @as(u64, @intCast(i))) % 31;
        const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(code)) - 15)) / (if (norm_weight) @as(f32, 512) else 128);
        v.* = bf(x + (if (gated_norm) @as(f32, 1) else 0));
    }
    return out;
}
pub fn project(a: std.mem.Allocator, x: Tensor, stem: []const u8, n: usize) !Tensor {
    const w = try source(a, try std.fmt.allocPrint(a, "{s}.weight", .{stem}), n * x.width, false, false);
    const y = try tensor(a, x.rows, n);
    for (0..x.rows) |r| for (0..n) |i| {
        var sum: f32 = 0;
        for (x.row(r), w[i * x.width ..][0..x.width]) |v, q| sum += f(v) * f(q);
        y.row(r)[i] = bf(sum);
    };
    return y;
}
pub fn norm(a: std.mem.Allocator, x: Tensor, stem: []const u8, group: usize, eps: f32) !Tensor {
    const gamma = try source(a, try std.fmt.allocPrint(a, "{s}.weight", .{stem}), x.width, true, false);
    const out = try tensor(a, x.rows, x.width);
    for (0..x.rows) |r| {
        var start: usize = 0;
        while (start < x.width) : (start += group) {
            var ss: f32 = 0;
            for (x.row(r)[start..][0..group]) |v| ss += f(v) * f(v);
            const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(group)) + eps);
            for (start..start + group) |i| out.row(r)[i] = bf(f(x.row(r)[i]) * inv * (1 + f(gamma[i])));
        }
    }
    return out;
}
pub fn rope(a: std.mem.Allocator, x: []const u16, rotary: usize, theta: f64, pos: usize) ![]u16 {
    const out = try a.dupe(u16, x);
    const half = rotary / 2;
    for (0..half) |i| {
        const frequency: f32 = @floatCast(std.math.pow(f64, theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rotary))));
        const angle = @as(f32, @floatFromInt(pos)) * frequency;
        const c = @cos(angle);
        const s = @sin(angle);
        out[i] = bf(f(x[i]) * c - f(x[i + half]) * s);
        out[i + half] = bf(f(x[i + half]) * c + f(x[i]) * s);
    }
    return out;
}

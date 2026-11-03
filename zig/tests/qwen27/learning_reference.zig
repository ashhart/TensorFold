//! Independent binary64 local adjoints on saved BF16/F32 nodes; no trainer or shader helpers are called.
const std = @import("std");
pub const Which = enum { pocket, affine, prep, scan };
pub const Buffer = struct { bytes: []u8, want: ?[]f64 = null, input: bool = true };
pub const Fixture = struct { buffers: []Buffer, params: []u8, groups: [3]usize, threads: usize = 256, bound: f64 };
pub const Shape = extern struct { u: [4]u32 align(16), f: [4]f32 align(16) };
pub const Prep = extern struct { pos: i32, heads: i32, kv: i32, dim: i32, half: i32, eps: f32 };
pub const Scan = extern struct { rows: i32, nk: i32, nv: i32, conv: i32, proj: i32, from: i32 };

pub fn floats(a: std.mem.Allocator, n: usize, salt: usize, divisor: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + salt * 7) % 29)) - 14)) / divisor;
    return out;
}
pub fn input(a: std.mem.Allocator, values: anytype) !Buffer {
    return .{ .bytes = try a.dupe(u8, std.mem.sliceAsBytes(values)) };
}
pub fn bfInput(a: std.mem.Allocator, values: []const f32) !Buffer {
    const words = try a.alloc(u16, values.len);
    for (words, values) |*to, x| {
        const bits: u32 = @bitCast(x);
        to.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }
    return input(a, words);
}
pub fn output(a: std.mem.Allocator, n: usize, initial: f32) !Buffer {
    const values = try a.alloc(f32, n);
    @memset(values, initial);
    const want = try a.alloc(f64, n);
    @memset(want, initial);
    return .{ .bytes = std.mem.sliceAsBytes(values), .want = want, .input = false };
}
pub fn scratch(a: std.mem.Allocator, n: usize) !Buffer {
    const out = try output(a, n, 0);
    return .{ .bytes = out.bytes, .input = false };
}
pub fn params(a: std.mem.Allocator, value: anytype) ![]u8 {
    return a.dupe(u8, std.mem.asBytes(&value));
}
pub fn scalar(b: Buffer, comptime T: type, i: usize) f64 {
    const x = std.mem.bytesAsSlice(T, @as([]align(@alignOf(T)) const u8, @alignCast(b.bytes)))[i];
    if (T == u16) return @as(f32, @bitCast(@as(u32, x) << 16));
    return x;
}

pub fn make(a: std.mem.Allocator, which: Which) !Fixture {
    return switch (which) {
        .pocket => pocket(a),
        .affine => affine(a),
        .prep => prep(a),
        .scan => @import("learning_scan_reference.zig").make(a),
    };
}
fn pocket(a: std.mem.Allocator) !Fixture {
    const rows = 3;
    const d = 65;
    const cap = 64;
    const neurons = 3;
    const b = try a.alloc(Buffer, 4);
    b[0] = try bfInput(a, try floats(a, rows * d, 0, 16));
    const weights = try a.alloc(f32, 3 * cap * d);
    @memset(weights, 0);
    for (0..3) |part| for (0..neurons) |j| for (0..d) |c| {
        weights[(part * cap + j) * d + c] = @as(f32, @floatFromInt(@as(i32, @intCast((part * 7 + j * 13 + c * 17) % 23)) - 11)) / 128;
    };
    b[1] = try input(a, weights);
    b[2] = try input(a, try floats(a, rows * d, 1, 32));
    b[3] = try output(a, rows * d, 0.125);
    for (0..rows) |r| for (0..neurons) |j| {
        var gate: f64 = 0;
        var up: f64 = 0;
        var error_dot: f64 = 0;
        for (0..d) |c| {
            const x = scalar(b[0], u16, r * d + c);
            gate += x * weights[j * d + c];
            up += x * weights[(cap + j) * d + c];
            error_dot += scalar(b[2], f32, r * d + c) * weights[(2 * cap + j) * d + c];
        }
        const sigmoid = 1 / (1 + @exp(-gate));
        const scale: f64 = if (j >= 1) 0.25 else 1;
        const cg = scale * error_dot * up * sigmoid * (1 + gate * (1 - sigmoid));
        const cu = scale * error_dot * gate * sigmoid;
        for (0..d) |c| b[3].want.?[r * d + c] += cg * weights[j * d + c] + cu * weights[(cap + j) * d + c];
    };
    return .{ .buffers = b, .params = try params(a, Shape{ .u = .{ 1, d, neurons, cap }, .f = .{ 0.25, 0, 0, 0 } }), .groups = .{ rows, 1, 1 }, .bound = 2e-6 };
}
fn affine(a: std.mem.Allocator) !Fixture {
    const rows = 5;
    const n = 17;
    const k = 128;
    const b = try a.alloc(Buffer, 5);
    b[0] = try input(a, try floats(a, rows * n, 2, 1024));
    const words = try a.alloc(u32, n * k / 8);
    for (words, 0..) |*word, i| word.* = @as(u32, @intCast(i)) *% 1550719 +% 191;
    b[1] = try input(a, words);
    const scales = try a.alloc(f32, n * k / 64);
    const biases = try a.alloc(f32, scales.len);
    for (scales, biases, 0..) |*s, *z, i| {
        s.* = @as(f32, @floatFromInt(i % 7 + 1)) / 64;
        z.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 32;
    }
    b[2] = try bfInput(a, scales);
    b[3] = try bfInput(a, biases);
    b[4] = try output(a, rows * k, 0.125);
    for (0..rows) |r| for (0..k) |c| for (0..n) |out| {
        const code = (words[out * k / 8 + c / 8] >> @as(u5, @intCast(4 * (c % 8)))) & 15;
        const group = out * (k / 64) + c / 64;
        b[4].want.?[r * k + c] += scalar(b[0], f32, r * n + out) * (scalar(b[2], u16, group) * @as(f64, @floatFromInt(code)) + scalar(b[3], u16, group));
    };
    return .{ .buffers = b, .params = try params(a, Shape{ .u = .{ rows, n, k, 1 }, .f = @splat(0) }), .groups = .{ ((rows + 3) / 4 * k + 255) / 256, 1, 1 }, .bound = 0 };
}
fn prep(a: std.mem.Allocator) !Fixture {
    const rows = 3;
    const heads: usize = 2;
    const kv: usize = 1;
    const dim = 256;
    const half = 32;
    const b = try a.alloc(Buffer, 9);
    b[0] = try bfInput(a, try floats(a, rows * 2 * heads * dim, 3, 16));
    b[1] = try bfInput(a, try floats(a, rows * kv * dim, 4, 16));
    for (2..4) |i| {
        const w = try a.alloc(f32, dim);
        for (w, 0..) |*x, d| x.* = @as(f32, @floatFromInt((d + i) % 7 + 1)) / 8;
        b[i] = try input(a, w);
    }
    const inv = try a.alloc(f32, half);
    for (inv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) / 1024;
    b[4] = try input(a, inv);
    b[5] = try input(a, try floats(a, rows * heads * dim, 5, 32));
    b[6] = try input(a, try floats(a, rows * kv * dim, 6, 32));
    b[7] = try output(a, rows * 2 * heads * dim, 0.125);
    b[8] = try output(a, rows * kv * dim, 0.125);
    for (0..rows) |r| for (0..heads + kv) |h| {
        const query = h < heads;
        const head = if (query) h else h - heads;
        const raw = if (query) b[0] else b[1];
        const grad = if (query) b[5] else b[6];
        const weights = if (query) b[2] else b[3];
        const raw_at = if (query) r * 2 * heads * dim + head * 2 * dim else (r * kv + head) * dim;
        const grad_at = (r * (if (query) heads else kv) + head) * dim;
        var square: f64 = 0;
        var dot: f64 = 0;
        var weighted: [dim]f64 = undefined;
        for (0..dim) |d| {
            var upstream = scalar(grad, f32, grad_at + d);
            if (d < 2 * half) {
                const angle = @as(f64, @floatFromInt(17 + r)) * scalar(b[4], f32, d % half);
                const other = if (d < half) d + half else d - half;
                upstream = upstream * @cos(angle) + (if (d < half) @as(f64, 1) else -1) * scalar(grad, f32, grad_at + other) * @sin(angle);
            }
            const x = scalar(raw, u16, raw_at + d);
            weighted[d] = scalar(weights, f32, d) * upstream;
            square += x * x;
            dot += x * weighted[d];
        }
        const eps: f32 = 1e-6;
        const inv_rms = 1 / @sqrt(square / dim + eps);
        const out = if (query) b[7].want.? else b[8].want.?;
        for (0..dim) |d| out[raw_at + d] = inv_rms * (weighted[d] - scalar(raw, u16, raw_at + d) * inv_rms * inv_rms * dot / dim);
    };
    return .{ .buffers = b, .params = try params(a, Prep{ .pos = 17, .heads = heads, .kv = kv, .dim = dim, .half = half, .eps = 1e-6 }), .groups = .{ heads + kv, rows, 1 }, .bound = 2e-6 };
}

test "Qwen27 local adjoints: every fixture has finite reference outputs and a fixed narrow ABI" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Shape));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Prep));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Scan));
    for (std.enums.values(Which)) |which| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const fixture = try make(arena.allocator(), which);
        var nonzero: usize = 0;
        for (fixture.buffers) |b| if (b.want) |want| for (want) |x| {
            try std.testing.expect(std.math.isFinite(x));
            nonzero += @intFromBool(x != 0);
        };
        try std.testing.expect(nonzero > 0);
    }
}

test {
    _ = @import("learning_reference_fd.zig");
}

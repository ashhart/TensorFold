//! The two bicubic resizes a picture meets, both separable fixed-point uint8 passes, width first: Pillow's and torch's.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    pillow, // Image.resize(BICUBIC): 22-bit coefficients
    torch, // torchvision's antialiased uint8 bicubic on the CPU: int16 coefficients at the most bits the largest allows
};

/// RGB `src` (w x h) resized to `out_w` x `out_h`; a pass whose size is unchanged is skipped. Caller frees.
pub fn resize(a: Allocator, kind: Kind, src: []const u8, w: usize, h: usize, out_w: usize, out_h: usize) Allocator.Error![]u8 {
    const wide = if (out_w != w) try horizontal(a, kind, src, w, h, out_w) else null;
    if (out_h == h) return wide orelse a.dupe(u8, src);
    defer if (wide) |p| a.free(p);
    return vertical(a, kind, wide orelse src, out_w * 3, h, out_h);
}

fn horizontal(a: Allocator, kind: Kind, src: []const u8, w: usize, h: usize, out_w: usize) Allocator.Error![]u8 {
    var t = try Taps.init(a, kind, w, out_w);
    defer t.deinit(a);
    const out = try a.alloc(u8, out_w * h * 3);
    for (0..h) |y| for (0..out_w) |x| {
        const first, const k = t.at(x);
        for (0..3) |c| {
            var acc: i32 = @as(i32, 1) << (t.shift - 1);
            for (k, 0..) |kv, j| acc += @as(i32, src[(y * w + first + j) * 3 + c]) * kv;
            out[(y * out_w + x) * 3 + c] = clamp(acc, t.shift);
        }
    };
    return out;
}

/// Rows of `row` bytes resized from `h` to `out_h`.
fn vertical(a: Allocator, kind: Kind, src: []const u8, row: usize, h: usize, out_h: usize) Allocator.Error![]u8 {
    var t = try Taps.init(a, kind, h, out_h);
    defer t.deinit(a);
    const out = try a.alloc(u8, out_h * row);
    for (0..out_h) |y| {
        const first, const k = t.at(y);
        for (0..row) |i| {
            var acc: i32 = @as(i32, 1) << (t.shift - 1);
            for (k, 0..) |kv, j| acc += @as(i32, src[(first + j) * row + i]) * kv;
            out[y * row + i] = clamp(acc, t.shift);
        }
    }
    return out;
}

fn clamp(acc: i32, shift: u5) u8 {
    return @intCast(std.math.clamp(acc >> shift, 0, 255));
}

/// Each output index's first input index, tap count and fixed-point weights.
const Taps = struct {
    first: []usize,
    count: []usize,
    weights: []i32, // `size` slots an output, the first `count` used
    size: usize,
    shift: u5,

    fn init(a: Allocator, kind: Kind, in: usize, out: usize) Allocator.Error!Taps {
        const scale = @as(f64, @floatFromInt(in)) / @as(f64, @floatFromInt(out));
        const filterscale = @max(scale, 1.0);
        const support = 2.0 * filterscale;
        const size: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const first = try a.alloc(usize, out);
        errdefer a.free(first);
        const count = try a.alloc(usize, out);
        errdefer a.free(count);
        const real = try a.alloc(f64, out * size);
        defer a.free(real);
        const inv = 1.0 / filterscale;
        var most: f64 = 0;
        for (0..out) |i| {
            const center = (@as(f64, @floatFromInt(i)) + 0.5) * scale;
            const lo: usize = @intFromFloat(@max(center - support + 0.5, 0));
            const hi = @min(@as(usize, @intFromFloat(center + support + 0.5)), in);
            first[i], count[i] = .{ lo, @min(hi -| lo, size) };
            const k = real[i * size ..][0..size];
            @memset(k, 0);
            var total: f64 = 0;
            for (k[0..count[i]], 0..) |*v, j| {
                v.* = filter(kind, (@as(f64, @floatFromInt(j + lo)) - center + 0.5) * inv);
                total += v.*;
            }
            if (total != 0) for (k[0..count[i]]) |*v| {
                v.* /= total;
                most = @max(most, v.*);
            };
        }
        const shift: u5 = switch (kind) {
            .pillow => 22,
            .torch => blk: {
                var p: u5 = 0;
                while (p < 22 and @as(i64, @intFromFloat(0.5 + most * @as(f64, @floatFromInt(@as(u32, 2) << p)))) < 1 << 15) p += 1;
                break :blk p;
            },
        };
        const weights = try a.alloc(i32, out * size);
        for (weights, real) |*q, v| {
            const x = v * @as(f64, @floatFromInt(@as(u32, 1) << shift));
            q.* = @intFromFloat(if (x < 0) -0.5 + x else 0.5 + x);
        }
        return .{ .first = first, .count = count, .weights = weights, .size = size, .shift = shift };
    }

    fn deinit(t: *Taps, a: Allocator) void {
        a.free(t.first);
        a.free(t.count);
        a.free(t.weights);
    }

    fn at(t: *const Taps, i: usize) struct { usize, []const i32 } {
        return .{ t.first[i], t.weights[i * t.size ..][0..t.count[i]] };
    }
};

/// Keys' cubic at a = -0.5, written as each library evaluates it.
fn filter(kind: Kind, x: f64) f64 {
    const a = -0.5;
    const ax = @abs(x);
    if (ax < 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1;
    if (ax < 2.0) return switch (kind) {
        .pillow => (((ax - 5) * ax + 8) * ax - 4) * a,
        .torch => ((a * ax - 5 * a) * ax + 8 * a) * ax - 4 * a,
    };
    return 0;
}

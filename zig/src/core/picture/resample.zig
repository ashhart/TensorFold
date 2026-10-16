//! Resizes of 8-bit RGB: Keys' bicubic kernel (a = -0.5), or Pillow's triangle, widened by the shrink factor, a width
//! pass then a height pass.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Where the fixed-point weights keep their fraction bits.
pub const Kind = enum {
    pillow, // Pillow's Image.resize(BICUBIC): 22 bits
    torch, // torch's antialiased uint8 bicubic: int16 weights at the most bits (at most 22) the largest allows
};

/// The kernel: Keys' cubic, or the triangle of Pillow's BILINEAR (Step3-VL's processor).
pub const Filter = enum {
    bicubic,
    bilinear,

    fn support(f: Filter) f64 {
        return switch (f) {
            .bicubic => 2,
            .bilinear => 1,
        };
    }

    fn at(f: Filter, x: f64) f64 {
        return switch (f) {
            .bicubic => cubic(x),
            .bilinear => if (@abs(x) < 1) 1 - @abs(x) else 0,
        };
    }
};

/// RGB rows `stride` bytes apart.
pub const View = struct {
    pixels: []const u8,
    width: usize,
    height: usize,
    stride: usize,

    pub fn of(pixels: []const u8, width: usize, height: usize) View {
        return .{ .pixels = pixels, .width = width, .height = height, .stride = width * 3 };
    }
};

/// `src` resized to `w` x `h` (bicubic) in a new buffer the caller frees.
pub fn resize(a: Allocator, kind: Kind, src: View, w: usize, h: usize) Allocator.Error![]u8 {
    return resizeWith(a, kind, .bicubic, src, w, h);
}

/// `resize` with `filter`'s kernel.
pub fn resizeWith(a: Allocator, kind: Kind, filter: Filter, src: View, w: usize, h: usize) Allocator.Error![]u8 {
    const out = try a.alloc(u8, w * h * 3);
    errdefer a.free(out);
    try intoWith(a, kind, filter, src, w, h, out);
    return out;
}

/// `src` resized to `w` x `h` (bicubic) into `out` (packed rows); a pass whose size is kept is skipped, so equal sizes copy.
pub fn into(a: Allocator, kind: Kind, src: View, w: usize, h: usize, out: []u8) Allocator.Error!void {
    return intoWith(a, kind, .bicubic, src, w, h, out);
}

/// `into` with `filter`'s kernel.
pub fn intoWith(a: Allocator, kind: Kind, filter: Filter, src: View, w: usize, h: usize, out: []u8) Allocator.Error!void {
    std.debug.assert(out.len == w * h * 3);
    var mid = src;
    var wide: ?[]u8 = null;
    defer if (wide) |p| a.free(p);
    if (w != src.width) {
        const dst = if (h == src.height) out else blk: {
            wide = try a.alloc(u8, w * src.height * 3);
            break :blk wide.?;
        };
        var t = try Taps.init(a, kind, filter, src.width, w);
        defer t.deinit(a);
        for (0..src.height) |y| {
            const row = src.pixels[y * src.stride ..][0 .. src.width * 3];
            const to = dst[y * w * 3 ..][0 .. w * 3];
            for (0..w) |x| {
                const first, const k = t.at(x);
                for (0..3) |c| {
                    var acc = t.half();
                    for (k, 0..) |kv, j| acc += @as(i64, row[(first + j) * 3 + c]) * kv;
                    to[x * 3 + c] = t.round(acc);
                }
            }
        }
        mid = View.of(dst, w, src.height);
    }
    if (h != src.height) {
        var t = try Taps.init(a, kind, filter, src.height, h);
        defer t.deinit(a);
        for (0..h) |y| {
            const first, const k = t.at(y);
            const to = out[y * w * 3 ..][0 .. w * 3];
            for (to, 0..) |*o, i| {
                var acc = t.half();
                for (k, 0..) |kv, j| acc += @as(i64, mid.pixels[(first + j) * mid.stride + i]) * kv;
                o.* = t.round(acc);
            }
        }
    } else if (w == src.width) {
        for (0..h) |y| @memcpy(out[y * w * 3 ..][0 .. w * 3], src.pixels[y * src.stride ..][0 .. w * 3]);
    }
}

/// Keys' cubic convolution kernel at a = -0.5, by Horner's rule (the one value both libraries compute).
fn cubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0;
}

/// Each destination index's first source index and its normalized weights, before any rounding to fixed point.
pub const Weights = struct {
    first: []usize,
    count: []usize,
    real: []f64, // `size` slots a destination index, the first `count` in use
    size: usize,

    /// One axis's weights from its sizes: torch's antialiased float resize reads them as they are.
    pub fn init(a: Allocator, filter: Filter, in: usize, out: usize) Allocator.Error!Weights {
        const scale = @as(f64, @floatFromInt(in)) / @as(f64, @floatFromInt(out));
        const stretch = @max(scale, 1.0); // shrinking widens the kernel by the factor: the antialiased form
        const support = filter.support() * stretch;
        const inv = 1 / stretch; // both libraries scale by the reciprocal
        const size = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const first = try a.alloc(usize, out);
        errdefer a.free(first);
        const count = try a.alloc(usize, out);
        errdefer a.free(count);
        const real = try a.alloc(f64, out * size);
        @memset(real, 0);
        for (0..out) |i| {
            const center = (@as(f64, @floatFromInt(i)) + 0.5) * scale;
            const lo: usize = @intFromFloat(@max(center - support + 0.5, 0));
            const hi = @min(@as(usize, @intFromFloat(center + support + 0.5)), in);
            first[i] = lo;
            count[i] = @min(hi -| lo, size);
            const k = real[i * size ..][0..count[i]];
            var total: f64 = 0;
            for (k, 0..) |*v, j| {
                v.* = filter.at((@as(f64, @floatFromInt(lo + j)) - center + 0.5) * inv);
                total += v.*;
            }
            if (total != 0) for (k) |*v| {
                v.* /= total;
            };
        }
        return .{ .first = first, .count = count, .real = real, .size = size };
    }

    pub fn deinit(w: *Weights, a: Allocator) void {
        a.free(w.first);
        a.free(w.count);
        a.free(w.real);
    }

    pub fn at(w: *const Weights, i: usize) struct { usize, []const f64 } {
        return .{ w.first[i], w.real[i * w.size ..][0..w.count[i]] };
    }
};

/// Each destination index's first source index and fixed-point weights, from one axis's sizes.
const Taps = struct {
    first: []usize,
    count: []usize,
    weights: []i32, // `size` slots a destination index, the first `count` in use
    size: usize,
    bits: u5,

    fn init(a: Allocator, kind: Kind, filter: Filter, in: usize, out: usize) Allocator.Error!Taps {
        const f = try Weights.init(a, filter, in, out);
        errdefer a.free(f.first);
        errdefer a.free(f.count);
        defer a.free(f.real);
        const first = f.first;
        const count = f.count;
        const real = f.real;
        const size = f.size;
        var most: f64 = 0;
        for (real) |v| most = @max(most, @abs(v));
        const bits: u5 = switch (kind) {
            .pillow => 22,
            .torch => blk: {
                var b: u5 = 0;
                while (b < 22 and @as(i64, @intFromFloat(0.5 + most * @as(f64, @floatFromInt(@as(u32, 2) << b)))) < 1 << 15) b += 1;
                break :blk b;
            },
        };
        const weights = try a.alloc(i32, out * size);
        const one: f64 = @floatFromInt(@as(u32, 1) << bits);
        for (weights, real) |*q, v| q.* = @intFromFloat(if (v < 0) -0.5 + v * one else 0.5 + v * one);
        return .{ .first = first, .count = count, .weights = weights, .size = size, .bits = bits };
    }

    fn deinit(t: *Taps, a: Allocator) void {
        a.free(t.first);
        a.free(t.count);
        a.free(t.weights);
    }

    fn at(t: *const Taps, i: usize) struct { usize, []const i32 } {
        return .{ t.first[i], t.weights[i * t.size ..][0..t.count[i]] };
    }

    fn half(t: *const Taps) i64 {
        return if (t.bits == 0) 0 else @as(i64, 1) << (t.bits - 1);
    }

    fn round(t: *const Taps, acc: i64) u8 {
        return @intCast(std.math.clamp(acc >> t.bits, 0, 255));
    }
};

/// torch's antialiased bicubic on float samples, sums fused as its arm64 build fuses them: RGB to f32 [3][h][w].
pub fn bicubicPlanes(a: Allocator, src: View, w: usize, h: usize, out: []f32) Allocator.Error!void {
    std.debug.assert(out.len == 3 * w * h);
    const mid = if (h == src.height) out else try a.alloc(f32, 3 * src.height * w);
    defer if (h != src.height) a.free(mid);
    if (w != src.width) {
        var t = try FloatTaps.init(a, src.width, w);
        defer t.deinit(a);
        for (0..src.height) |y| {
            const row = src.pixels[y * src.stride ..][0 .. src.width * 3];
            for (0..w) |x| {
                const first, const k = t.at(x);
                for (0..3) |c| {
                    var acc = sample(row[first * 3 + c]) * k[0]; // a window always holds a sample
                    for (k[1..], first + 1..) |kv, i| acc = @mulAdd(f32, sample(row[i * 3 + c]), kv, acc);
                    mid[(c * src.height + y) * w + x] = acc;
                }
            }
        }
    } else for (0..src.height) |y| for (0..w) |x| for (0..3) |c| {
        mid[(c * src.height + y) * w + x] = sample(src.pixels[y * src.stride + x * 3 + c]);
    };
    if (h == src.height) return;
    var t = try FloatTaps.init(a, src.height, h);
    defer t.deinit(a);
    for (0..3) |c| for (0..h) |y| {
        const first, const k = t.at(y);
        const plane = mid[c * src.height * w ..][0 .. src.height * w];
        for (out[(c * h + y) * w ..][0..w], 0..) |*o, x| {
            var acc = plane[first * w + x] * k[0];
            for (k[1..], first + 1..) |kv, i| acc = @mulAdd(f32, plane[i * w + x], kv, acc);
            o.* = acc;
        }
    };
}

fn sample(v: u8) f32 {
    return @floatFromInt(v);
}

/// Keys' kernel at a = -0.5 in float32 with fused multiply-adds, as the float path's arm64 build weighs a tap.
fn cubic32(x0: f32) f32 {
    const a: f32 = -0.5;
    const x = @abs(x0);
    if (x < 1) return @mulAdd(f32, @mulAdd(f32, a + 2, x, -(a + 3)) * x, x, 1);
    if (x < 2) return @mulAdd(f32, @mulAdd(f32, @mulAdd(f32, a, x, -5 * a), x, 8 * a), x, -4 * a);
    return 0;
}

/// One axis's float taps: a window widened by the shrink factor, centre and offsets in float32, weights summing to one.
const FloatTaps = struct {
    first: []usize,
    count: []usize,
    weights: []f32,
    size: usize,

    fn init(a: Allocator, in: usize, out: usize) Allocator.Error!FloatTaps {
        const scale = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
        const support: f32 = if (scale >= 1) 2 * scale else 2;
        const inv: f32 = if (scale >= 1) @floatCast(1 / @as(f64, scale)) else 1;
        const size = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const first = try a.alloc(usize, out);
        errdefer a.free(first);
        const count = try a.alloc(usize, out);
        errdefer a.free(count);
        const weights = try a.alloc(f32, out * size);
        @memset(weights, 0);
        for (0..out) |i| {
            const center: f32 = @floatCast(@as(f64, scale) * (@as(f64, @floatFromInt(i)) + 0.5));
            const lo_at = @as(f64, center - support) + 0.5;
            const lo: usize = if (lo_at > 0) @intFromFloat(lo_at) else 0;
            const hi = @min(@as(usize, @intFromFloat(@as(f64, center + support) + 0.5)), in);
            first[i] = lo;
            count[i] = @min(hi -| lo, size);
            const k = weights[i * size ..][0..count[i]];
            var total: f32 = 0;
            for (k, 0..) |*v, j| {
                const d = @as(f32, @floatFromInt(lo + j)) - center;
                v.* = cubic32(@floatCast((@as(f64, d) + 0.5) * inv));
                total += v.*;
            }
            if (total != 0) for (k) |*v| {
                v.* /= total;
            };
        }
        return .{ .first = first, .count = count, .weights = weights, .size = size };
    }

    fn deinit(t: *FloatTaps, a: Allocator) void {
        a.free(t.first);
        a.free(t.count);
        a.free(t.weights);
    }

    fn at(t: *const FloatTaps, i: usize) struct { usize, []const f32 } {
        return .{ t.first[i], t.weights[i * t.size ..][0..t.count[i]] };
    }
};

test "a flat picture stays flat at every size, both kinds" {
    const a = std.testing.allocator;
    const px = try a.alloc(u8, 7 * 5 * 3);
    defer a.free(px);
    @memset(px, 200);
    for ([_]Kind{ .pillow, .torch }) |kind| for ([_][2]usize{ .{ 13, 3 }, .{ 2, 9 }, .{ 7, 5 }, .{ 1, 1 } }) |s| {
        const out = try resize(a, kind, View.of(px, 7, 5), s[0], s[1]);
        defer a.free(out);
        for (out) |v| try std.testing.expectEqual(@as(u8, 200), v);
    };
}

test "a kept size copies, and strided rows read only their pixels" {
    const a = std.testing.allocator;
    var px: [2 * 8 * 3]u8 = undefined;
    for (&px, 0..) |*v, i| v.* = @intCast(i);
    const view: View = .{ .pixels = &px, .width = 2, .height = 2, .stride = 8 * 3 };
    const same = try resize(a, .pillow, view, 2, 2);
    defer a.free(same);
    try std.testing.expectEqualSlices(u8, px[0..6], same[0..6]);
    try std.testing.expectEqualSlices(u8, px[24..30], same[6..12]);
}

test "torch's weights keep the largest under 2^15 at the most fraction bits" {
    const a = std.testing.allocator;
    for ([_][2]usize{ .{ 10, 3 }, .{ 3, 10 }, .{ 1280, 512 }, .{ 900, 32 } }) |s| {
        var t = try Taps.init(a, .torch, .bicubic, s[0], s[1]);
        defer t.deinit(a);
        var most: i32 = 0;
        for (t.weights) |w| most = @max(most, @as(i32, @intCast(@abs(w))));
        try std.testing.expect(most < 1 << 15 and t.bits <= 22);
        if (t.bits < 22) try std.testing.expect(most >= 1 << 14);
    }
}

test "bilinear matches Pillow 12.3's Image.resize(BILINEAR), shrinking and enlarging" {
    const a = std.testing.allocator;
    var px: [7 * 5 * 3]u8 = undefined;
    for (&px, 0..) |*p, i| p.* = @intCast((i * 37 + 11) % 256);
    const small = try resizeWith(a, .pillow, .bilinear, .of(&px, 7, 5), 3, 2);
    defer a.free(small);
    try std.testing.expectEqualSlices(u8, &.{ 113, 97, 134, 154, 131, 111, 134, 169, 128, 93, 115, 152, 131, 108, 129, 152, 149, 83 }, small);
    const big = try resizeWith(a, .pillow, .bilinear, .of(px[0 .. 4 * 2 * 3], 4, 2), 6, 3);
    defer a.free(big);
    try std.testing.expectEqualSlices(u8, &.{
        11,  48,  85, 67,  104, 141, 141, 135, 172, 214, 38,  75,  161, 70,  107, 88, 125, 162,
        105, 142, 51, 97,  134, 107, 107, 123, 160, 180, 111, 148, 127, 100, 137, 54, 91,  128,
        199, 236, 17, 127, 164, 73,  73,  110, 147, 146, 183, 220, 93,  130, 167, 20, 57,  94,
    }, big);
}

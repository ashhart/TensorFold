//! Metal-free bit oracle for DeepSeek-V4.1's routed EXL3 slot mapping, compensated dots and activation.
const std = @import("std");
const fmt = @import("core/exl3_format.zig");
const decode = @import("core/exl3_projection_decode.zig");
const row = @import("core/exl3_projection_row_ref.zig");
const ref = @import("families/deepseek_v41/metal/experts_exl3_ref.zig");

fn word(tile: []const i16, i: usize) u32 {
    return @as(u32, @as(u16, @bitCast(tile[2 * i]))) | (@as(u32, @as(u16, @bitCast(tile[2 * i + 1]))) << 16);
}

/// The Metal two-word extraction after tf_x3_maps has located one fragment's stream value.
fn mappedState(tile: []const i16, map: u32) u16 {
    const wi = map & 63;
    const off = (map >> 6) & 31;
    const lo = word(tile, wi);
    const hi = word(tile, map >> 11);
    const carry: u32 = (hi >> @intCast((32 - off) & 31)) & (0 -% @as(u32, @intFromBool(off != 0)));
    return @truncate(((lo << @intCast(off)) | carry) >> 16);
}

fn fragment(col: u32, r: u32) u32 {
    return 8 * (4 * (col % 8) + (r % 8) / 2) + 4 * (col / 8) + 2 * (r / 8) + r % 2;
}

fn maps(col: u32, halves: u32) [16]u32 {
    const ring = 128 * halves;
    var out: [16]u32 = undefined;
    for (&out, 0..) |*map, r| {
        const p = fragment(col, @intCast(r));
        const end = ((p + 1) * halves - (if (halves & 1 != 0) (p + 1) & 1 else @as(u32, 0))) / 2;
        const start = (end + ring - 16) % ring;
        const wi = start / 32;
        const next = if (wi + 1 == 4 * halves) 0 else wi + 1;
        map.* = wi | ((start % 32) << 6) | (next << 11);
    }
    return out;
}

const Dot = enum { compensated, plain };

fn twoSum(comptime dot: Dot, high: *f32, low: *f32, product: f32) void {
    const next = high.* + product;
    const part = next - high.*;
    if (dot == .compensated) low.* += (high.* - (next - part)) + (product - part);
    high.* = next;
}

/// Host model of tf_x3_pass: NW matrices and RP rows, with each output's independent TwoSum chain.
fn pass(comptime nw: usize, comptime rp: usize, comptime dot: Dot, matrices: [nw]ref.Matrix, x: []const f16, pairs: usize, rows: [rp]u32, col: u32, nb: usize, out: []f32, k: usize, n: usize) void {
    var high: [nw][rp]f32 = @splat(@splat(0));
    var low: [nw][rp]f32 = @splat(@splat(0));
    for (0..k / 16) |kb| {
        for (0..16) |t| {
            for (0..nw) |v| {
                const m = matrices[v];
                const words = m.width.tileWords();
                const tile = m.trellis[(kb * (n / 16) + nb / 16) * words ..][0..words];
                const ms = maps(col, m.width.halves);
                const w: f32 = decode.value(m.book, mappedState(tile, ms[t]));
                for (0..rp) |r| {
                    const product = @as(f32, x[(v * pairs + rows[r]) * k + kb * 16 + t]) * w;
                    twoSum(dot, &high[v][r], &low[v][r], product);
                }
            }
        }
    }
    for (0..nw) |v| {
        for (0..rp) |r| {
            out[(v * pairs + rows[r]) * n + nb + col] = high[v][r] + low[v][r];
        }
    }
}

const Fill = enum { random, cancel };

fn checkPass(comptime nw: usize, comptime rp: usize, comptime dot: Dot, book: fmt.Codebook, halves: u16, k: usize, n: usize, fill: Fill, seed: u64) !void {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const pairs = 9;
    const selected = if (rp == 4) 7 else rp;
    const order = [7]u32{ 8, 0, 6, 2, 7, 1, 4 };
    var rows: [selected]u32 = undefined;
    for (&rows, 0..) |*r, i| r.* = order[i];
    var matrices: [nw]ref.Matrix = undefined;
    var words: [nw][]i16 = undefined;
    defer for (words) |w| a.free(w);
    for (0..nw) |v| {
        const vbook: fmt.Codebook = if (v == 0) book else switch (book) {
            .inst3 => .mcg,
            .mcg => .mul1,
            .mul1 => .inst3,
        };
        const vhalves: u16 = if (v == 0) halves else if (vbook == .mul1) 5 else 6;
        const tiles = k / 16 * (n / 16);
        words[v] = try a.alloc(i16, tiles * @as(usize, vhalves) * 8);
        const tile_words = @as(usize, vhalves) * 8;
        for (0..n / 16) |tn| {
            for (words[v][tn * tile_words ..][0..tile_words]) |*w| w.* = rng.int(i16);
            for (1..k / 16) |tk| {
                const at = (tk * (n / 16) + tn) * tile_words;
                if (fill == .cancel) @memcpy(words[v][at..][0..tile_words], words[v][tn * tile_words ..][0..tile_words]) else for (words[v][at..][0..tile_words]) |*w| w.* = rng.int(i16);
            }
        }
        matrices[v] = .{ .width = .{ .halves = vhalves }, .book = vbook, .trellis = words[v], .su = &.{}, .sv = &.{} };
    }
    const x = try a.alloc(f16, nw * pairs * k);
    defer a.free(x);
    for (x) |*v| {
        if (fill == .cancel) {
            v.* = 0x1p-12;
        } else {
            var bits = rng.int(u16);
            if (bits & 0x7c00 == 0x7c00) bits = 0x3c00;
            v.* = @bitCast(bits);
        }
    }
    if (fill == .cancel) for (0..nw * pairs) |r| {
        x[r * k] = 4096;
        x[r * k + 48] = -4096;
    };
    const special = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x7e55, 0x7dff, 0x7c00, 0xfc00, 0x3c00, 0x3555 };
    if (fill == .random) for (0..nw * pairs) |r| for (special, 0..) |bits, i| {
        x[r * k + (i * 17) % k] = @bitCast(bits);
    };
    const got = try a.alloc(f32, nw * pairs * n);
    defer a.free(got);
    const want = try a.alloc(f32, got.len);
    defer a.free(want);
    @memset(got, @bitCast(@as(u32, 0x7fc12345)));
    @memcpy(want, got);
    for (0..n / 16) |tn| for (0..16) |col| {
        if (rp == 4) {
            pass(nw, 4, dot, matrices, x, pairs, rows[0..4].*, @intCast(col), tn * 16, got, k, n);
            pass(nw, 2, dot, matrices, x, pairs, rows[4..6].*, @intCast(col), tn * 16, got, k, n);
            pass(nw, 1, dot, matrices, x, pairs, .{rows[6]}, @intCast(col), tn * 16, got, k, n);
        } else {
            pass(nw, rp, dot, matrices, x, pairs, rows, @intCast(col), tn * 16, got, k, n);
        }
    };
    for (0..nw) |v| for (rows) |r| {
        const offset = (v * pairs + r) * n;
        row.dot(matrices[v].width, matrices[v].book, matrices[v].trellis, n / 16, x[(v * pairs + r) * k ..][0..k], 0, want[offset..][0..n]);
    };
    var plain_differs = false;
    for (got, want, 0..) |g, w, i| {
        const same = @as(u32, @bitCast(g)) == @as(u32, @bitCast(w));
        if (dot == .plain) {
            plain_differs = plain_differs or !same;
            continue;
        }
        if (!same) std.debug.print("mismatch at flat output {d}, nw={d}, rp={d}, book={t}, halves={d}, k={d}\n", .{ i, nw, rp, book, halves, k });
        try std.testing.expectEqual(@as(u32, @bitCast(w)), @as(u32, @bitCast(g)));
    }
    if (dot == .plain) try std.testing.expect(plain_differs);
}

test "routed EXL3 fragment maps: every row, column, width and circular boundary agrees with the scalar decoder" {
    var prng = std.Random.DefaultPrng.init(0x4d415053);
    const rng = prng.random();
    var tile: [128]i16 = undefined;
    for (&tile, 0..) |*v, i| v.* = if (i % 3 == 0) rng.int(i16) else @bitCast(@as(u16, @truncate(i * 0x9e37)));
    for (2..17) |halves| {
        const words = @as(usize, halves) * 8;
        for (0..16) |col| {
            const ms = maps(@intCast(col), @intCast(halves));
            for (0..16) |r| {
                const want = decode.state(.{ .halves = @intCast(halves) }, tile[0..words], fragment(@intCast(col), @intCast(r)));
                try std.testing.expectEqual(want, mappedState(tile[0..words], ms[r]));
            }
        }
    }
}

test "routed EXL3 host model: random, exceptional and noncontiguous slot rows match the scalar decoder bit for bit" {
    const forms = [_]struct { fmt.Codebook, u16 }{ .{ .inst3, 4 }, .{ .mcg, 4 }, .{ .mul1, 5 }, .{ .mul1, 8 } };
    for (forms, 0..) |form, fi| for ([_]usize{ 128, 256 }) |k| {
        inline for (.{ 1, 2 }) |nw| inline for (.{ 1, 2, 4 }) |rp| try checkPass(nw, rp, .compensated, form[0], form[1], k, 128, .random, 0x58334d4154 + fi * 97 + k + nw * 11 + rp);
    };
}

test "routed EXL3 host model: large cancelling terms cannot hide the compensated small-term sum" {
    inline for (.{ 1, 2 }) |nw| inline for (.{ 1, 2, 4 }) |rp| try checkPass(nw, rp, .compensated, .mul1, 8, 128, 128, .cancel, 0x54575953 + nw * 11 + rp);
}

test "routed EXL3 planted control: removing TwoSum's error term disagrees with the scalar oracle" {
    try checkPass(1, 1, .plain, .mul1, 8, 128, 128, .cancel, 0x504c41494e544);
}

fn ftz(v: f32) f32 {
    return if (v != 0 and @abs(v) < std.math.floatMin(f32)) std.math.copysign(@as(f32, 0), v) else v;
}

fn mul(a: f32, b: f32) f32 {
    return ftz(ftz(a) * ftz(b));
}

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate(bits >> 16);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn clip(x: f32, lo: f32, hi: f32) f32 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, lo, hi);
}

/// Independent direct epilogue; the table isolates the Metal SiLU evaluation for the Mac contract.
fn act(g: u16, u: u16, w: f32, limit: f32, silu: *const ref.Silu) u16 {
    const gc = bf16(clip(fromBf16(g), -std.math.inf(f32), limit));
    return bf16(mul(mul(silu[gc], clip(fromBf16(u), -limit, limit)), w));
}

test "routed EXL3 post-SiLU-table activation oracle: every BF16 gate, special values and RN-even ties preserve output bits" {
    var silu: ref.Silu = undefined;
    for (&silu, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 257)) - 128)) / 32;
    silu[0] = 0;
    silu[1] = @bitCast(@as(u32, 1));
    silu[0x8000] = -0.0;
    silu[0x7f80] = std.math.inf(f32);
    silu[0xff80] = std.math.nan(f32);
    silu[0x7fc0] = std.math.nan(f32);
    const ups = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x7fc1, 0x7fa1, 0x7f80, 0xff80, 0x3f80, 0xbf80, 0x3f81, 0xbf81 };
    const weights = [_]u32{ 0, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 1, 0x7f800000, 0x7fc12345 };
    for (0..65536) |g| for (ups, 0..) |u, ui| {
        const w: f32 = @bitCast(weights[(g + ui) % weights.len]);
        const want = act(@intCast(g), u, w, 10, &silu);
        const got = ref.act(@intCast(g), u, w, 10, &silu);
        try std.testing.expectEqual(want, got);
    };
    // A NaN keeps its payload's high bits: 0x7fc10000 is bf16 0x7fc1, not the canonical 0x7fc0.
    try std.testing.expectEqual(@as(u16, 0x7fc1), ref.bf16(@bitCast(@as(u32, 0x7fc10000))));
    try std.testing.expectEqual(@as(u16, 0x7fc0), ref.bf16(@bitCast(@as(u32, 0x7fc00000))));
    try std.testing.expectEqual(@as(u16, 0xffc1), ref.bf16(@bitCast(@as(u32, 0xffc10000))));
    for ([_]u32{ 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x00000001, 0x80000001, 0x00004000, 0x00004001, 0x80004000, 0x80004001, 0x7fc10000, 0xffc10000 }) |bits| {
        const v: f32 = @bitCast(bits);
        try std.testing.expectEqual(bf16(v), ref.bf16(v));
    }
}

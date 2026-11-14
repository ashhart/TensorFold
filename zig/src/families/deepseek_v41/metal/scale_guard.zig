//! fp8_nax folds each E4M3 code times its block's E8M0 scale into one bf16 operand: a load-time scan keeps that exact.
const std = @import("std");
const weights = @import("weights.zig");
const fp8_mma = @import("fp8_mma.zig");

/// The least scale byte at which every nonzero code's product is a normal number (a code's lowest bit is 2^-9).
pub const least: u8 = 10;

/// The lowest byte of `s`, 64 bytes a step.
pub fn lowest(s: []const u8) u8 {
    var m: @Vector(64, u8) = @splat(255);
    var j: usize = 0;
    while (j + 64 <= s.len) : (j += 64) m = @min(m, @as(@Vector(64, u8), s[j..][0..64].*));
    var low = @reduce(.Min, m);
    for (s[j..]) |b| low = @min(low, b);
    return low;
}

/// Whether bf16 holds every code [n][k] times its 32x32 block's scale exactly: blocks under `least` hold only zeros.
pub fn exact(codes: []const u8, scales: []const u8, k: usize) bool {
    if (lowest(scales) >= least) return true;
    const groups = k / 32;
    for (scales, 0..) |b, i| {
        if (b >= least) continue;
        const first = i / groups * 32 * k + i % groups * 32;
        for (0..32) |r| for (codes[first + r * k ..][0..32]) |x| if (x & 0x7f != 0) return false;
    }
    return true;
}

fn bytes(r: weights.Ref, len: usize) []const u8 {
    return r.buf.contents()[r.off..][0..len];
}

/// A layer's dense FP8 projections, the ones prompt rows run through fp8_mma.encode.
fn dense(l: *const weights.Layer, out: *[11]weights.Fp8) []const weights.Fp8 {
    var n: usize = 0;
    for ([_]weights.Fp8{ l.wq_a, l.wkv, l.wq_b, l.wo_a, l.wo_b } ++ l.shared) |f| {
        out[n] = f;
        n += 1;
    }
    for ([_]?weights.Fp8{ l.ix_wq_b, if (l.engram) |g| g.wkv else null }) |o| if (o) |f| {
        out[n] = f;
        n += 1;
    };
    return out[0..n];
}

/// With tensor-unit FP8 rows on, every dense projection's scales are scanned; an inexact fold turns them off.
pub fn check(io: std.Io, p: *fp8_mma.Pipelines, layers: []const weights.Layer) void {
    const nax = p.nax orelse return;
    const t0 = std.Io.Clock.awake.now(io);
    var low: u8 = 255;
    var total: usize = 0;
    var buf: [11]weights.Fp8 = undefined;
    for (layers) |*l| for (dense(l, &buf)) |f| {
        const s = bytes(f.s, @as(usize, f.n / 32) * (f.k / 32));
        total += s.len;
        low = @min(low, lowest(s));
        if (exact(bytes(f.w, @as(usize, f.n) * f.k), s, f.k)) continue;
        std.log.warn("dsv41 metal: a dense FP8 block's scale byte under {d} has nonzero codes (bf16 cannot hold them exactly): tensor-unit FP8 rows off", .{least});
        for (nax) |x| x.deinit();
        p.nax = null;
        return;
    };
    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    std.log.info("dsv41 metal: dense FP8 scales {d} KiB checked in {d:.1} ms, lowest byte {d}: tensor-unit FP8 rows on", .{ total >> 10, ms, low });
}

test "the scan finds the lowest byte in bodies and tails, and a low block is exact only with zero codes" {
    const k = 96;
    var scales: [2 * 3]u8 = .{ 120, 119, 121, 118, 122, 117 };
    var codes: [64 * k]u8 = undefined;
    for (&codes, 0..) |*b, i| b.* = @intCast(i % 251);
    try std.testing.expectEqual(@as(u8, 117), lowest(&scales));
    try std.testing.expect(exact(&codes, &scales, k));
    var long: [200]u8 = @splat(200);
    for ([_]usize{ 5, 130, 199 }) |at| {
        long[at] = 3;
        try std.testing.expectEqual(@as(u8, 3), lowest(&long));
        long[at] = 200;
    }
    // block (row group 1, k group 2) under `least`: its codes are nonzero, then zeroed (signs alone still fold to zero)
    scales[5] = least - 1;
    try std.testing.expect(!exact(&codes, &scales, k));
    for (0..32) |r| @memset(codes[(32 + r) * k + 64 ..][0..32], 0x80);
    try std.testing.expect(exact(&codes, &scales, k));
    codes[(32 + 31) * k + 95] = 1;
    try std.testing.expect(!exact(&codes, &scales, k));
}

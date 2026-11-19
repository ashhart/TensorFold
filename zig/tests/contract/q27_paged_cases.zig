//! Paged prompt panels and source faults shared by the device fixture and its host witnesses.
const std = @import("std");
const c = @import("contract").spec;
const twins = @import("kernel_sources").qwen27_paged;

pub const second = "kt < kbeg + TF_PAGE_TOKENS ? v0 : v1";
pub const wrong = "kt < kbeg + TF_PAGE_TOKENS ? v0 : v0";
pub const scan = "if (simd_any(inf != 0) && lane == 0) atomic_fetch_or";
pub const no_scan = "if (false && lane == 0) atomic_fetch_or";

pub fn fault(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const count = std.mem.count(u8, text, second);
    if (count != 4 and count != 8) return error.ControlEdit;
    return std.mem.replaceOwned(u8, a, text, second, wrong);
}

pub fn served(comptime n: usize) []const c.Shape {
    return &.{ .{ .rows = 16, .n = n, .k = 255 }, .{ .rows = 16, .n = n, .k = 256 }, .{ .rows = 16, .n = n, .k = 511 }, .{ .rows = 16, .n = n, .k = 512 }, .{ .rows = 1, .n = n, .k = 255 }, .{ .rows = 1, .n = n, .k = 256 }, .{ .rows = 1, .n = n, .k = 511 }, .{ .rows = 1, .n = n, .k = 512 }, .{ .rows = 16, .n = n, .k = 0 }, .{ .rows = 128, .n = n, .k = 1920 }, .{ .rows = 16, .n = n, .k = 8176 } };
}

/// Reverse page order with page zero unused: no logical page has its contiguous address.
pub fn table(a: std.mem.Allocator, rows: usize) ![]u32 {
    const count = (rows + 255) / 256;
    const out = try a.alloc(u32, count);
    for (out, 0..) |*p, i| p.* = @intCast(count - i);
    return out;
}

pub fn index(ids: []const u32, heads: usize, h: usize, row: usize, d: usize) usize {
    return ((@as(usize, ids[row / 256]) * heads + h) * 256 + row % 256) * 256 + d;
}

/// Spare pages and partial-page padding stay NaN so a read outside the logical cache is visible.
pub fn pack(a: std.mem.Allocator, flat: []const u16, heads: usize, rows: usize, ids: []const u32) ![]u16 {
    const out = try a.alloc(u16, (ids.len + 2) * heads * 256 * 256);
    @memset(out, c.toBf16(std.math.nan(f32)));
    for (0..heads) |h| for (0..rows) |row| {
        @memcpy(out[index(ids, heads, h, row, 0)..][0..256], flat[(h * rows + row) * 256 ..][0..256]);
    };
    return out;
}

test "paged prompt controls mutate each real M5 and simdgroup step and scan" {
    const a = std.testing.allocator;
    for ([_][]const u8{ twins.attention_prompt, twins.attention_prompt_sg }) |text| {
        const bad = try fault(a, text);
        defer a.free(bad);
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, bad, second));
        try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, bad, wrong));
        try std.testing.expect(std.mem.indexOf(u8, bad, "q27_nonfinite_paged") != null);
    }
    const joined = try std.mem.concat(a, u8, &.{ twins.attention_prompt, twins.attention_prompt_sg });
    defer a.free(joined);
    const joined_bad = try fault(a, joined);
    defer a.free(joined_bad);
    try std.testing.expectEqual(@as(usize, 8), std.mem.count(u8, joined_bad, wrong));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, twins.attention_prompt, scan));
    const bad_scan = try std.mem.replaceOwned(u8, a, twins.attention_prompt, scan, no_scan);
    defer a.free(bad_scan);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, bad_scan, no_scan));
    try std.testing.expectError(error.ControlEdit, fault(a, "stale source"));
}

test "paged prompt shuffled storage keeps every head row and poisons unused final-page rows" {
    const a = std.testing.allocator;
    const rows = 529;
    const heads = 2;
    const ids = try table(a, rows);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 2, 1 }, ids);
    const flat = try a.alloc(u16, heads * rows * 256);
    defer a.free(flat);
    for (flat, 0..) |*v, i| v.* = @truncate(i ^ ((i >> 16) * 137));
    const arena = try pack(a, flat, heads, rows, ids);
    defer a.free(arena);
    for (0..heads) |h| for (0..rows) |row| {
        try std.testing.expectEqualSlices(u16, flat[(h * rows + row) * 256 ..][0..256], arena[index(ids, heads, h, row, 0)..][0..256]);
    };
    try std.testing.expect(std.math.isNan(c.fromBf16(arena[0])));
    try std.testing.expect(std.math.isNan(c.fromBf16(arena[index(ids, heads, 1, rows, 0)])));
}

test "paged prompt panels straddle page and chunk seams and expose a stale second page" {
    for ([_]usize{ 255, 256, 511, 512 }) |boundary| {
        var found = false;
        for (served(24)) |s| if (s.k == boundary and s.rows == 16) {
            found = true;
        };
        try std.testing.expect(found);
    }
    const ids = [_]u32{ 3, 2, 1 };
    const first = index(&ids, 1, 0, 0, 0);
    const actual = index(&ids, 1, 0, 256, 0);
    const stale = first + 256 * 256;
    try std.testing.expect(actual != stale);
    const a = std.testing.allocator;
    const flat = try a.alloc(u16, 529 * 256);
    defer a.free(flat);
    @memset(flat, 0);
    flat[256 * 256] = c.toBf16(1);
    const arena = try pack(a, flat, 1, 529, &ids);
    defer a.free(arena);
    const correct = c.fromBf16(arena[actual]) / 257;
    const faulted = c.fromBf16(arena[stale]) / 257;
    try std.testing.expect(correct > 0 and std.math.isFinite(correct));
    try std.testing.expect(std.math.isNan(faulted));
}

test "paged prompt nonfinite dispatch control has a class witness" {
    const visible: f64 = 1;
    const hidden: f64 = std.math.inf(f64);
    const unmasked = visible + 0 * hidden;
    try std.testing.expect(std.math.isFinite(visible));
    try std.testing.expect(std.math.isNan(unmasked));
}
